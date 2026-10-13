from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss


class CLIPEraseLoss(nn.Module):
    def __init__(self,
                 lambda_rm: float = 0.0,
                 lambda_cm: float = 0.0,
                 lambda_unif: float = 0.0,
                 lambda_dist: float = 0.0,  
                 temperature: float = 0.07):
        super().__init__()

        self.lambda_rm = lambda_rm
        self.lambda_cm = lambda_cm
        self.lambda_unif = lambda_unif     
        self.lambda_dist = lambda_dist 
        self.temperature = temperature
        self.is_cliperase = True
        self.noise_std = 0.08
        self.noise_type = "gaussian"

        self.ema_rank = None

    def add_text_noise(self, x, std=None):
        if std is None:
            std = self.noise_std
        
        original_norm = torch.norm(x, dim=-1, keepdim=True)
        
        if self.noise_type == "gaussian":
            noise = torch.randn_like(x) * std
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(x) - 0.5) * 2 * std * np.sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
        
        x_noisy = x + noise
        new_norm = torch.norm(x_noisy, dim=-1, keepdim=True)
        x_noisy = x_noisy * (original_norm / new_norm.clamp(min=1e-8))
        
        return x_noisy

    def hypersphere_uniformity(self, x, t=2, weights=None):
        
        x = F.normalize(x, dim=-1)
        pdist = torch.cdist(x, x, p=2) ** 2
        exp_vals = torch.exp(-t * pdist)
        
        loss_per_sample = torch.log(exp_vals.mean(dim=1) + 1e-12)  # [B]
        
        if weights is not None:
            weights = weights / weights.sum()
            loss = (loss_per_sample * weights).sum()
        else:
            loss = loss_per_sample.mean()
        
        return loss

    def energy_distance(self, f, g, t, f_weights=None):
        
        f = F.normalize(f, dim=-1)
        g = F.normalize(g, dim=-1)
        t = F.normalize(t, dim=-1)
        dist_fg = torch.cdist(f, g, p=2)
        mean_fg_per_sample = dist_fg.mean(dim=1) 
        
        dist_ft = torch.cdist(f, t, p=2)
        mean_ft_per_sample = dist_ft.mean(dim=1)  
        
        loss_per_sample = (mean_fg_per_sample - mean_ft_per_sample).abs()  
        
        if f_weights is not None:
            f_weights = f_weights / f_weights.sum()
            loss = (loss_per_sample * f_weights).sum()
        else:
            loss = loss_per_sample.mean()
        
        return loss

    def compute_similarity_matrix(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_img = (image_features @ text_features.t()) / self.temperature
        logits_txt = (text_features @ image_features.t()) / self.temperature
        return logits_img, logits_txt



    def retention_loss(self, img, txt):
        logit_i, logit_t = self.compute_similarity_matrix(img, txt)
        B = img.shape[0]
        labels = torch.arange(B, device=img.device)
        return (F.cross_entropy(logit_i, labels) +
                F.cross_entropy(logit_t, labels)) / 2

    def consistency_loss(self, orig_img, orig_txt, new_img, new_txt):
        orig_img = F.normalize(orig_img, dim=-1)
        orig_txt = F.normalize(orig_txt, dim=-1)
        new_img = F.normalize(new_img, dim=-1)
        new_txt = F.normalize(new_txt, dim=-1)

        o_i, o_t = self.compute_similarity_matrix(orig_img, orig_txt)
        n_i, n_t = self.compute_similarity_matrix(new_img, new_txt)

        return (
            F.kl_div(F.log_softmax(n_i, dim=-1), F.softmax(o_i, dim=-1), reduction="batchmean") +
            F.kl_div(F.log_softmax(n_t, dim=-1), F.softmax(o_t, dim=-1), reduction="batchmean")
        ) / 2
  
    
    

    def forward(self,
                batch_image_features, batch_text_features,
                original_batch_image_features, original_batch_text_features,
                flags,
                sample_weights=None):
        
        forget_mask = flags.bool()
        retain_mask = ~forget_mask
        device = flags.device

        total_loss = torch.zeros((), device=device, requires_grad=True)
        loss_dict = {}

        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
            forget_weights = sample_weights[forget_mask] if forget_mask.sum() > 0 else None
        else:
            forget_weights = None

        if forget_weights is not None:
            loss_dict["forget_weight_mean"] = forget_weights.mean().item()
            loss_dict["forget_weight_sum"] = forget_weights.sum().item()

        if self.lambda_rm != 0:
            if retain_mask.sum() > 0 and forget_mask.sum() > 0 :
                r_img = batch_image_features[retain_mask]
                r_txt = batch_text_features[retain_mask]
                loss_rm = self.retention_loss(r_img, r_txt)
                total_loss = total_loss + self.lambda_rm * loss_rm
                loss_dict["retention_loss"] = loss_rm.item()
            else:
                loss_dict["retention_loss"] = 0.0
        

        if self.lambda_unif != 0:
            if forget_mask.sum() > 0:
                f_img = batch_image_features[forget_mask]
                f_txt = batch_text_features[forget_mask]
                loss_unif_img = self.hypersphere_uniformity(f_img, weights=forget_weights)
                loss_unif_txt = self.hypersphere_uniformity(f_txt, weights=forget_weights)
                loss_uniform = loss_unif_img + loss_unif_txt
                loss_dict["uniform_loss"] = loss_uniform.item()
                total_loss = total_loss + self.lambda_unif * loss_uniform
            else:
                loss_dict["uniform_loss"] = 0.0

        if self.lambda_dist != 0:
            if retain_mask.sum() > 0 and forget_mask.sum() > 0:
                r_img = batch_image_features[retain_mask]
                r_txt = batch_text_features[retain_mask]
                f_img = batch_image_features[forget_mask]
                f_txt = batch_text_features[forget_mask]
                f_txt = self.add_text_noise(f_txt)

                loss_dist = (
                    self.energy_distance(f_img, r_txt, f_txt, f_weights=forget_weights) +
                    self.energy_distance(f_txt, r_img, f_img, f_weights=forget_weights)
                )
                loss_dict["dist_loss"] = loss_dist.item()
                total_loss = total_loss + self.lambda_dist * loss_dist
            else:
                loss_dict["dist_loss"] = 0.0
        

        if self.lambda_cm != 0:
            if retain_mask.sum() > 0 and forget_mask.sum() > 0:
                ri_o = original_batch_image_features[retain_mask]
                rt_o = original_batch_text_features[retain_mask]
                ri_n = batch_image_features[retain_mask]
                rt_n = batch_text_features[retain_mask]
                loss_cm = self.consistency_loss(ri_o, rt_o, ri_n, rt_n)
                total_loss = total_loss + self.lambda_cm * loss_cm
                loss_dict["consistency_loss"] = loss_cm.item()
            else:
                loss_dict["consistency_loss"] = 0.0


        loss_dict["total_loss"] = float(total_loss.detach())
        return total_loss, loss_dict