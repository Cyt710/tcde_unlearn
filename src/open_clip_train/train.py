import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP, create_model_and_transforms
from open_clip_train.distributed import is_master, broadcast_object
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self, device):
        if not dist.is_available() or not dist.is_initialized():
            return
         
        val_tensor = torch.tensor(float(self.val), device=device)
        sum_tensor = torch.tensor(float(self.sum), device=device)
        count_tensor = torch.tensor(float(self.count), device=device)
        
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        world_size = dist.get_world_size()
        self.val = val_tensor.item() / world_size
        self.sum = sum_tensor.item()
        self.count = count_tensor.item()
        if self.count > 0:
            self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def prepare_batch_data(batch, device, input_dtype):
    
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        images, texts, is_forget, labels = batch
        
        images = images.to(device=device, dtype=input_dtype)
        texts = texts.to(device=device)
        labels = labels.to(device=device)
        if is_forget is not None:
            is_forget = is_forget.to(device=device)
        
        return images, texts, is_forget, labels
    
    else:
        batch_size = 1
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if hasattr(batch[0], 'shape'):
                batch_size = batch[0].shape[0]
        
        images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=input_dtype)
        texts = torch.randint(0, 1000, (batch_size, 77), device=device)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        return images, texts, None, labels


def compute_unified_loss(model_out, loss_fn,original_model=None, images=None, texts=None, is_forget=None, args=None):
    
    if isinstance(model_out, dict):
        image_features = model_out["image_features"]
        text_features = model_out["text_features"]
        logit_scale = model_out.get("logit_scale")
    else:
        image_features = model_out[0]
        text_features = model_out[1]
        logit_scale = model_out[2] if len(model_out) > 2 else None
   
    is_cliperase = (hasattr(args, 'cliperase') and args.cliperase) or hasattr(loss_fn, 'is_cliperase')
    
    if is_cliperase:
        if original_model is not None and images is not None and texts is not None:
            with torch.no_grad():
                original_model_out = original_model(images, texts)
                if isinstance(original_model_out, dict):
                    original_image_features = original_model_out["image_features"]
                    original_text_features = original_model_out["text_features"]
                else:
                    original_image_features = original_model_out[0]
                    original_text_features = original_model_out[1]
        else:
            original_image_features = image_features.detach().clone()
            original_text_features = text_features.detach().clone()
        
        total_loss, loss_dict = loss_fn(
            image_features,
            text_features, 
            original_image_features,
            original_text_features,
            is_forget
        )
        return total_loss, loss_dict
    else:
        if isinstance(model_out, dict):
            losses = loss_fn(**model_out, output_dict=True)
        else:
            losses = loss_fn(image_features, text_features, logit_scale, output_dict=True)
        
        total_loss = sum(losses.values())
        losses["loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss, losses


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    original_model = None
    if hasattr(args, 'cliperase') and args.cliperase:
        try:
            if args.pretrained:
                original_model = create_model_and_transforms(
                    args.model,
                    args.pretrained,
                    precision=args.precision,
                    device=device,
                    output_dict=True,
                )[0]
            elif args.resume:
                original_model = create_model_and_transforms(
                    args.model,
                    args.resume,
                    precision=args.precision,
                    device=device,
                    output_dict=True,
                )[0]
            
            if original_model is not None:
                original_model.eval()
                if is_master(args):
                    logging.info("loaded original model for consistency loss caculation")
        except Exception as e:
            if is_master(args):
                logging.warning(f"failure to loading original model: {e}")
            original_model = None

    for i, batch in enumerate(dataloader):
        try:
            images, texts, is_forget, labels = prepare_batch_data(batch, device, input_dtype)
        except Exception as e:
            if is_master(args):
                logging.error(f"failure for batch data preparation: {e}")
                logging.error(f"batch type: {type(batch)}")
                if isinstance(batch, (list, tuple)):
                    logging.error(f"batch size: {len(batch)}")
            continue

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        current_losses = {}
        total_loss = 0

        if hasattr(args, 'cliperase') and args.cliperase:
            with autocast():
                model_out = model(images, texts)
             
                total_loss, current_losses = compute_unified_loss(
                    model_out, loss, original_model, images, texts, is_forget, args
                )

            if is_forget is not None:
                forget_count = is_forget.sum().item()
                retain_count = len(images) - forget_count
                current_losses["forget_samples"] = forget_count
                current_losses["retain_samples"] = retain_count

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
        else:
            if args.accum_freq == 1:
                with autocast():
                    model_out = model(images, texts)
                    
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        if isinstance(model_out, dict) and isinstance(dist_model_out, dict):
                            model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    
                    total_loss, current_losses = compute_unified_loss(
                        model_out, loss, None, None, None, None, args
                    )

                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
            else:
                with torch.no_grad():
                    with autocast():
                        model_out = model(images, texts)
                        
                        if args.distill:
                            with torch.no_grad():
                                dist_model_out = dist_model(images, texts)
                            if isinstance(model_out, dict) and isinstance(dist_model_out, dict):
                                model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})

                        for f in ("logit_scale", "logit_bias"):
                            if isinstance(model_out, dict):
                                model_out.pop(f, None)

                        for key, val in model_out.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                if ((i + 1) % args.accum_freq) > 0:
                    # FIXME this makes data time logging unreliable when accumulating
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                optimizer.zero_grad()
                for j in range(args.accum_freq):
                    images_j = accum_images[j]
                    texts_j = accum_texts[j]
                    with autocast():
                        model_out_j = model(images_j, texts_j)

                        
                        if args.distill:
                            with torch.no_grad():
                                dist_model_out_j = dist_model(images_j, texts_j)
                            if isinstance(model_out_j, dict) and isinstance(dist_model_out_j, dict):
                                model_out_j.update({f'dist_{k}': v for k, v in dist_model_out_j.items()})

                        inputs_no_accum = {}
                        if isinstance(model_out_j, dict):
                            inputs_no_accum["logit_scale"] = model_out_j.pop("logit_scale", None)
                            if "logit_bias" in model_out_j:
                                inputs_no_accum["logit_bias"] = model_out_j.pop("logit_bias")

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [model_out_j[key]] + accumulated[j + 1:])

                        total_loss, current_losses = compute_unified_loss(
                            {**inputs, **inputs_no_accum}, loss, None, None, None, None, args
                        )

                    if scaler is not None:
                        scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()

        should_update_optimizer = (args.accum_freq == 1) or ((i + 1) % args.accum_freq == 0)
        
        if should_update_optimizer:
            if scaler is not None:
                has_valid_gradients = False
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            has_valid_gradients = True
                            break
                    if has_valid_gradients:
                        break
                
                if not has_valid_gradients:
                    if is_master(args):
                        logging.warning(f"Step {step}: no valid gradients found, skipping optimizer step")
                    continue
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()

        if args.accum_freq > 1 and ((i + 1) % args.accum_freq == 0):
            accum_images, accum_texts, accum_features = [], [], []

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in current_losses.items():
                if isinstance(val, (int, float)):
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val, batch_size)

            if dist.is_available() and dist.is_initialized():
                batch_time_m.sync(device)
                data_time_m.sync(device)
                for loss_m in losses_m.values():
                    loss_m.sync(device)

            logit_scale = None
            if hasattr(unwrap_model(model), 'logit_scale'):
                logit_scale = unwrap_model(model).logit_scale
            logit_scale_scalar = logit_scale.item() if logit_scale is not None else 0.0
            
            loss_log_parts = []
            for loss_name, loss_m in losses_m.items():
                loss_log_parts.append(f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})")
            loss_log = " ".join(loss_log_parts)
            
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)
            
            batch_time_m.reset()
            data_time_m.reset()

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, loss_fn=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    
    if is_master(args):
        logging.info(f"evaluation,epoch={epoch}")

    if hasattr(args, 'zeroshot_frequency') and args.zeroshot_frequency and (epoch % args.zeroshot_frequency == 0):
        if is_master(args):
            logging.info(f"zero-shot evaluation, epoch={epoch}")
        zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
        if is_master(args):
            metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        if is_master(args):
            logging.info(f"evaluation, epoch={epoch}")
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        
        original_model = None
        if hasattr(args, 'cliperase') and args.cliperase and loss_fn is not None:
            try:
                if args.pretrained:
                    original_model = create_model_and_transforms(
                        args.model,
                        args.pretrained,
                        precision=args.precision,
                        device=device,
                        output_dict=True,
                    )[0]
                    original_model.eval()
                    if is_master(args):
                        logging.info("The original model has been loaded for computation.")
            except Exception as e:
                if is_master(args):
                    logging.warning(f"Unable to load the original model during evaluation: {e}")
        
        model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                try:
                    images, texts, is_forget, batch_labels = prepare_batch_data(batch, device, input_dtype)
                    batch_size = images.shape[0]
                    
                    if is_forget is not None:
                        if not isinstance(is_forget, torch.Tensor):
                            is_forget = torch.tensor(is_forget, device=device, dtype=torch.bool)
                        elif is_forget.device != device:
                            is_forget = is_forget.to(device)
                        
                except Exception as e:
                    if is_master(args):
                        logging.error(f"data failure: {e}")
                    continue

                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    
                    if loss_fn is not None:
                        total_loss, _ = compute_unified_loss(
                            model_out, loss_fn, original_model, images, texts, is_forget, args
                        )
                    else:
                        if isinstance(model_out, dict):
                            image_features = model_out["image_features"]
                            text_features = model_out["text_features"]
                            logit_scale = model_out["logit_scale"]
                        else:
                            image_features = model_out[0]
                            text_features = model_out[1]
                            logit_scale = model_out[2]
                        
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()

                        batch_size = images.shape[0]
                        labels_tensor = torch.arange(batch_size, device=device).long()
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels_tensor) +
                            F.cross_entropy(logits_per_text, labels_tensor)
                        ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)
                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

        if dist.is_available() and dist.is_initialized():
            cumulative_loss_tensor = torch.tensor(cumulative_loss, device=device)
            cumulative_gen_loss_tensor = torch.tensor(cumulative_gen_loss, device=device)
            num_samples_tensor = torch.tensor(num_samples, device=device)
            
            dist.all_reduce(cumulative_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(cumulative_gen_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
            
            cumulative_loss = cumulative_loss_tensor.item()
            cumulative_gen_loss = cumulative_gen_loss_tensor.item()
            num_samples = num_samples_tensor.item()

        if is_master(args) and num_samples > 0:
            val_loss_value = cumulative_loss / num_samples
            metrics["val_loss"] = val_loss_value
            
            if hasattr(val_loss_value, 'item'):
                metrics["val_loss"] = val_loss_value.item()
            elif torch.is_tensor(val_loss_value):
                metrics["val_loss"] = val_loss_value.cpu().item()
            else:
                metrics["val_loss"] = float(val_loss_value)
            
            if cumulative_gen_loss > 0:
                metrics["val_generative_loss"] = cumulative_gen_loss / num_samples
            

    if is_master(args) and metrics:
        val_loss_display = metrics.get("val_loss", "N/A")
        if val_loss_display != "N/A":
            val_loss_display = f"{val_loss_display:.4f}"
    
        logging.info(f"Epoch: {epoch} val_loss: {val_loss_display}")
    
        other_metrics = {k: v for k, v in metrics.items() if k != "val_loss"}
        if other_metrics:
            other_str = " ".join([f"{k}: {v:.4f}" for k, v in other_metrics.items() if isinstance(v, float)])
            logging.info(f" {other_str}")
        
        log_data = {"val/" + name: val for name, val in metrics.items()}

        if args.save_logs:
            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, epoch)

            with open(os.path.join(args.checkpoint_path, "results.json"), "a+") as f:
                serializable_metrics = {}
                for key, value in metrics.items():
                    if hasattr(value, 'item'):  
                        serializable_metrics[key] = value.item()
                    else:
                        serializable_metrics[key] = value
                f.write(json.dumps(serializable_metrics))
                f.write("\n")

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            if 'train' in data:
                dataloader = data['train'].dataloader
                num_batches_per_epoch = dataloader.num_batches // args.accum_freq
                step = num_batches_per_epoch * epoch
            else:
                step = None
            log_data['epoch'] = epoch
            wandb.log(log_data, step=step)

    if dist.is_available() and dist.is_initialized():
        metrics = broadcast_object(metrics, device)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
    return None