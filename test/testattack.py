import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from copy import deepcopy
import os
import sys
import math
import json
import numpy as np
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import random
import string
import open_clip
from collections import defaultdict
import datetime
import time
from tqdm import tqdm
import logging
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import UniversalDataset, UniversalDataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CIFAR100_LABEL_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def ensure_same_dtype(tensor1, tensor2):
    if tensor1.dtype != tensor2.dtype:
        if tensor1.dtype == torch.float32 or tensor2.dtype == torch.float32:
            return tensor1.to(torch.float32), tensor2.to(torch.float32)
        else:
            return tensor1.to(tensor2.dtype), tensor2
    return tensor1, tensor2


def load_openclip_model(checkpoint_path=None, model_name="ViT-B/32", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    if checkpoint_path is None:
        logger.info(f"Loading pretrained OpenCLIP model: {model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="laion2b_s32b_b82k"
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        return model, preprocess, tokenizer

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=None
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '').replace('_orig_mod.', '')
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    logger.info("Model loaded successfully")
    return model, preprocess, tokenizer


def compute_image_to_text_retrieval(model, preprocess, data_loader, device, class_names, forget_classes, output_dir="./similarity_results", asr_k=5):
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "openclip_image2text"

    logger.info("Preparing text features for all classes...")
    tokenizer = open_clip.get_tokenizer("ViT-B/32")
    text_inputs = tokenizer([f"a photo of a {c}" for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)

    logger.info("Collecting all image features...")
    all_image_features = []
    all_texts = []
    all_flags = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            images, texts, flags, labels = batch

            image_tensor = torch.stack([preprocess(img) for img in images]).to(device)
            flags_tensor = torch.tensor(flags).to(device)

            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)

            all_image_features.append(image_features.cpu())
            all_texts.extend(texts)
            all_flags.extend(flags_tensor.cpu().tolist())
            all_labels.extend(labels)

    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    total_samples = len(all_texts)
    logger.info(f"Total images collected: {total_samples}")

    forget_classes_set = set(forget_classes)
    forget_class_indices = [class_names.index(c) for c in forget_classes if c in class_names]
    forget_class_indices_set = set(forget_class_indices)
    retain_class_indices = [i for i in range(len(class_names)) if i not in forget_class_indices_set]

    logger.info("Computing Image -> Text retrieval metrics...")
    image_to_text_sim = all_image_features @ text_features.T

    forget_similarity_data = {c: {'true_similarities': [], 'other_similarities': [], 'num_samples': 0}
                              for c in forget_classes}
    retain_similarity_data = {class_names[idx]: {'true_similarities': [], 'other_similarities': [], 'num_samples': 0}
                              for idx in retain_class_indices}

    metrics = {
        'forget_set': {'total': 0, 'r@5': 0, 'r@10': 0, f'asr@{asr_k}': 0},
        'retain_set': {'total': 0, 'r@5': 0, 'r@10': 0, f'asr@{asr_k}': 0}
    }

    classification_metrics = {
        'forget_set': {'total': 0, 'correct': 0},
        'retain_set': {'total': 0, 'correct': 0}
    }

    with torch.no_grad():
        for i in range(total_samples):
            is_forget = all_flags[i]
            true_class = all_texts[i]
            true_idx = class_names.index(true_class)
            sim_vector = image_to_text_sim[i].cpu().numpy()

            sorted_indices = np.argsort(sim_vector)[::-1]
            sorted_indices_asc = np.argsort(sim_vector)

            r5_correct = true_idx in sorted_indices[:5]
            r10_correct = true_idx in sorted_indices[:10]

            bottom_k_indices = set(sorted_indices_asc[:asr_k])
            forget_in_bottom_k = bottom_k_indices & forget_class_indices_set
            asr_success = len(forget_in_bottom_k) > 0

            pred_idx = int(np.argmax(sim_vector))
            class_correct = (pred_idx == true_idx)

            if is_forget and true_class in forget_classes_set:
                set_type = 'forget_set'
                true_sim = float(sim_vector[true_idx])
                other_sims = np.delete(sim_vector, true_idx).tolist()

                forget_similarity_data[true_class]['true_similarities'].append(true_sim)
                forget_similarity_data[true_class]['other_similarities'].extend(other_sims)
                forget_similarity_data[true_class]['num_samples'] += 1
            else:
                set_type = 'retain_set'
                true_sim = float(sim_vector[true_idx])
                other_sims = np.delete(sim_vector, true_idx).tolist()

                retain_similarity_data[true_class]['true_similarities'].append(true_sim)
                retain_similarity_data[true_class]['other_similarities'].extend(other_sims)
                retain_similarity_data[true_class]['num_samples'] += 1

            metrics[set_type]['total'] += 1
            if r5_correct:
                metrics[set_type]['r@5'] += 1
            if r10_correct:
                metrics[set_type]['r@10'] += 1
            if asr_success:
                metrics[set_type][f'asr@{asr_k}'] += 1

            classification_metrics[set_type]['total'] += 1
            if class_correct:
                classification_metrics[set_type]['correct'] += 1

    for set_type in ['forget_set', 'retain_set']:
        total = metrics[set_type]['total']
        if total > 0:
            metrics[set_type][f'asr@{asr_k}'] = 100.0 * metrics[set_type][f'asr@{asr_k}'] / total
            metrics[set_type]['r@5'] = 100.0 * metrics[set_type]['r@5'] / total
            metrics[set_type]['r@10'] = 100.0 * metrics[set_type]['r@10'] / total

        total_cls = classification_metrics[set_type]['total']
        classification_metrics[set_type]['accuracy'] = 100.0 * classification_metrics[set_type]['correct'] / total_cls if total_cls > 0 else 0.0

    logger.info("Computing similarity statistics...")

    forget_set_stats = {}
    for class_name, class_data in forget_similarity_data.items():
        true_sims = class_data['true_similarities']
        other_sims = class_data['other_similarities']
        num_samples = class_data['num_samples']

        if num_samples > 0:
            mean_true = float(np.mean(true_sims))
            std_true = float(np.std(true_sims))
            mean_other = float(np.mean(other_sims)) if other_sims else 0.0
            std_other = float(np.std(other_sims)) if other_sims else 0.0
            discrimination = mean_true - mean_other
        else:
            mean_true = 0.0
            std_true = 0.0
            mean_other = 0.0
            std_other = 0.0
            discrimination = 0.0

        forget_set_stats[class_name] = {
            'num_samples': num_samples,
            'mean_similarity': mean_true,
            'std_similarity': std_true,
            'mean_other_similarity': mean_other,
            'std_other_similarity': std_other,
            'discrimination_score': discrimination,
            'true_similarities': true_sims,
            'other_similarities': other_sims
        }

    retain_set_stats = {}
    for class_name, class_data in retain_similarity_data.items():
        true_sims = class_data['true_similarities']
        other_sims = class_data['other_similarities']
        num_samples = class_data['num_samples']

        if num_samples > 0:
            mean_true = float(np.mean(true_sims))
            std_true = float(np.std(true_sims))
            mean_other = float(np.mean(other_sims)) if other_sims else 0.0
            std_other = float(np.std(other_sims)) if other_sims else 0.0
            discrimination = mean_true - mean_other
        else:
            mean_true = 0.0
            std_true = 0.0
            mean_other = 0.0
            std_other = 0.0
            discrimination = 0.0

        retain_set_stats[class_name] = {
            'num_samples': num_samples,
            'mean_similarity': mean_true,
            'std_similarity': std_true,
            'mean_other_similarity': mean_other,
            'std_other_similarity': std_other,
            'discrimination_score': discrimination,
            'true_similarities': true_sims,
            'other_similarities': other_sims
        }

    all_forget_true = []
    all_forget_other = []
    all_retain_true = []
    all_retain_other = []

    for class_name, stats in forget_set_stats.items():
        all_forget_true.extend(stats['true_similarities'])
        all_forget_other.extend(stats['other_similarities'])

    for class_name, stats in retain_set_stats.items():
        all_retain_true.extend(stats['true_similarities'])
        all_retain_other.extend(stats['other_similarities'])

    overall_statistics = {
        'forget_set': {
            'total_samples': len(all_forget_true),
            'num_classes': len(forget_classes),
            'classes': forget_classes,
            'true_similarities': all_forget_true,
            'other_similarities': all_forget_other,
            'mean_true': float(np.mean(all_forget_true)) if all_forget_true else 0.0,
            'std_true': float(np.std(all_forget_true)) if all_forget_true else 0.0,
            'mean_other': float(np.mean(all_forget_other)) if all_forget_other else 0.0,
            'std_other': float(np.std(all_forget_other)) if all_forget_other else 0.0,
        },
        'retain_set': {
            'total_samples': len(all_retain_true),
            'num_classes': len(retain_class_indices),
            'true_similarities': all_retain_true,
            'other_similarities': all_retain_other,
            'mean_true': float(np.mean(all_retain_true)) if all_retain_true else 0.0,
            'std_true': float(np.std(all_retain_true)) if all_retain_true else 0.0,
            'mean_other': float(np.mean(all_retain_other)) if all_retain_other else 0.0,
            'std_other': float(np.std(all_retain_other)) if all_retain_other else 0.0,
        }
    }

    logger.info("=" * 100)
    logger.info("Image -> Text Retrieval Results")
    logger.info("=" * 100)
    logger.info(f"ASR@{asr_k}: Forget class in bottom-{asr_k} least similar text labels")
    logger.info("-" * 100)
    logger.info(f"{'Metric':<25} {'Forget Set':<20} {'Retain Set':<20} {'Description':<30}")
    logger.info("-" * 100)

    f_metrics = metrics['forget_set']
    r_metrics = metrics['retain_set']

    logger.info(f"{'Total Samples':<25} {f_metrics['total']:<20} {r_metrics['total']:<20}")
    logger.info(f"{'R@5':<25} {f_metrics['r@5']:.2f}%{'':<15} {r_metrics['r@5']:.2f}%{'':<15} {'True label in top-5':<30}")
    logger.info(f"{'R@10':<25} {f_metrics['r@10']:.2f}%{'':<14} {r_metrics['r@10']:.2f}%{'':<14} {'True label in top-10':<30}")
    logger.info(f"{f'ASR@{asr_k}':<25} {f_metrics[f'asr@{asr_k}']:.2f}%{'':<15} {r_metrics[f'asr@{asr_k}']:.2f}%{'':<15} {f'Forget in bottom-{asr_k}':<30}")
    logger.info("-" * 100)

    acc_f = classification_metrics['forget_set']['accuracy']
    acc_r = classification_metrics['retain_set']['accuracy']
    logger.info(f"{'Classification Accuracy':<25} Acc.Df: {acc_f:.2f}%{'':<8} Acc.Dr: {acc_r:.2f}%")
    logger.info("=" * 100)

    logger.info(f"Forget Class Similarity Statistics:")
    logger.info(f"{'Class':<20} {'Mean True Sim':<15} {'Mean Other Sim':<15} {'Discrimination':<15}")
    logger.info("-" * 65)
    for class_name in forget_classes:
        stats = forget_set_stats[class_name]
        if stats['num_samples'] > 0:
            logger.info(f"{class_name:<20} {stats['mean_similarity']:<15.4f} {stats['mean_other_similarity']:<15.4f} {stats['discrimination_score']:<15.4f}")

    logger.info(f"Summary:")
    logger.info(f"  Forget classes: {forget_classes}")
    logger.info(f"  Acc.Df: {acc_f:.2f}%  |  Acc.Dr: {acc_r:.2f}%")
    logger.info(f"  R@5 (Forget/Retain): {f_metrics['r@5']:.2f}% / {r_metrics['r@5']:.2f}%")
    logger.info(f"  R@10 (Forget/Retain): {f_metrics['r@10']:.2f}% / {r_metrics['r@10']:.2f}%")
    logger.info(f"  ASR@{asr_k} (Forget/Retain): {f_metrics[f'asr@{asr_k}']:.2f}% / {r_metrics[f'asr@{asr_k}']:.2f}%")
    logger.info("=" * 100)

    results = {
        'config': {
            'forget_classes': forget_classes,
            'asr_k': asr_k,
            'timestamp': timestamp,
            'model_name': model_name
        },
        'metrics': {
            'forget_set': metrics['forget_set'],
            'retain_set': metrics['retain_set']
        },
        'classification': {
            'forget_set': classification_metrics['forget_set'],
            'retain_set': classification_metrics['retain_set']
        },
        'forget_set': forget_set_stats,
        'retain_set': retain_set_stats,
        'overall_statistics': overall_statistics
    }

    output_file = os.path.join(output_dir, f"{model_name}_{timestamp}_similarities.json")
    logger.info(f"Saving similarity results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    compact_results = {
        'config': results['config'],
        'metrics': results['metrics'],
        'classification': results['classification'],
        'forget_set': {k: {kk: vv for kk, vv in v.items() if kk not in ['true_similarities', 'other_similarities']}
                       for k, v in forget_set_stats.items()},
        'retain_set': {k: {kk: vv for kk, vv in v.items() if kk not in ['true_similarities', 'other_similarities']}
                       for k, v in retain_set_stats.items()},
        'overall_statistics': {
            'forget_set': {k: v for k, v in overall_statistics['forget_set'].items()
                          if k not in ['true_similarities', 'other_similarities']},
            'retain_set': {k: v for k, v in overall_statistics['retain_set'].items()
                          if k not in ['true_similarities', 'other_similarities']}
        }
    }
    compact_file = os.path.join(output_dir, f"{model_name}_{timestamp}_similarities_compact.json")
    logger.info(f"Saving compact results to: {compact_file}")
    with open(compact_file, 'w', encoding='utf-8') as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)

    logger.info("Image -> Text retrieval evaluation completed!")

    return results


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP Image-to-Text Retrieval Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="ViT-L-14", help="OpenCLIP model name")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--forget_classes", type=str, nargs="+", default=["apple"], help="List of forget classes")
    parser.add_argument("--asr_k", type=int, default=5, help="K for ASR@K")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./similarity_results", help="Output directory")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(device)

    logger.info(f"Using device: {device}")

    forget_classes = args.forget_classes
    asr_k = args.asr_k

    def forget_rule(sample):
        return sample["text"] in forget_classes

    logger.info("Loading test dataset...")
    test_dataset = UniversalDataset(
        "CIFAR100",
        root=args.data_root,
        train=False,
        download=True,
        label_names={i: name for i, name in enumerate(CIFAR100_LABEL_NAMES)}
    )
    test_dataset.set_forget_rule(forget_rule)

    test_loader = UniversalDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    forget_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i]["flag"])
    retain_count = len(test_dataset) - forget_count
    logger.info(f"Forget set samples: {forget_count}")
    logger.info(f"Retain set samples: {retain_count}")

    logger.info("=" * 80)
    logger.info("Loading OpenCLIP Model")
    logger.info("=" * 80)

    model, preprocess, tokenizer = load_openclip_model(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        device=device
    )
    model.eval()

    results = compute_image_to_text_retrieval(
        model=model,
        preprocess=preprocess,
        data_loader=test_loader,
        device=device,
        class_names=CIFAR100_LABEL_NAMES,
        forget_classes=forget_classes,
        output_dir=args.output_dir,
        asr_k=asr_k
    )

    logger.info("=" * 80)
    logger.info("All evaluations completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()