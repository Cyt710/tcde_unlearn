import torch
import torch.nn.functional as F
import os
import sys
import json
import clip
import numpy as np
import random
from typing import *
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time
import logging
import argparse
import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import UniversalDataset
from testcifar import CLIPModel

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


def ensure_same_dtype(t1, t2):
    if t1.dtype != t2.dtype:
        if t1.dtype == torch.float32 or t2.dtype == torch.float32:
            return t1.to(torch.float32), t2.to(torch.float32)
        else:
            return t1.to(t2.dtype), t2
    return t1, t2


def generate_random_text_tokens(num_tokens=50, length=10):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [''.join(random.choice(letters) for _ in range(length)) for _ in range(num_tokens)]


def generate_random_images_batch(num_images=50, shape=(224, 224, 3)):
    imgs = []
    for _ in range(num_images):
        arr = np.random.rand(*shape) * 255
        imgs.append(Image.fromarray(arr.astype(np.uint8)))
    return imgs


def generate_gaussian_noise_images_batch(num_images=50, shape=(224, 224, 3), mean=127, std=50):
    imgs = []
    for _ in range(num_images):
        noise = np.random.normal(mean, std, shape)
        noise = np.clip(noise, 0, 255)
        imgs.append(Image.fromarray(noise.astype(np.uint8)))
    return imgs


def generate_uniform_noise_images_batch(num_images=50, shape=(224, 224, 3)):
    imgs = []
    for _ in range(num_images):
        noise = np.random.rand(*shape) * 255
        imgs.append(Image.fromarray(noise.astype(np.uint8)))
    return imgs


class CrossModalSimilarityAnalyzerBatch:

    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
            self.text_features = model.encode_text(text_inputs)
            self.text_features = F.normalize(self.text_features, p=2, dim=-1)

    def collect_images_by_class(self, dataset, num_samples_per_class=10):
        class_images = defaultdict(list)

        for sample in dataset:
            name = sample['text']

            if len(class_images[name]) < num_samples_per_class:
                class_images[name].append(sample['image'])

                if all(len(imgs) >= num_samples_per_class for imgs in class_images.values()):
                    break

        max_samples = max(len(imgs) for imgs in class_images.values())

        for name in class_images:
            while len(class_images[name]) < max_samples:
                class_images[name].append(class_images[name][0])

        return class_images

    def analyze_random_text_on_real_images_batch(
        self,
        dataset,
        num_samples_per_class=10,
        num_random_texts=20,
        batch_size=64,
        save_path=None):

        logger.info("Random Text -> Real Image Similarity Analysis")
        logger.info("=" * 80)

        start = time.time()

        results = {
            'class_similarities': defaultdict(list),
            'random_text_similarities': [],
        }

        logger.info("Collecting samples...")

        class_images = self.collect_images_by_class(dataset, num_samples_per_class)

        for cname, images in tqdm(class_images.items()):
            tensors = []

            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                t = torch.stack([self.model.preprocess(img) for img in batch]).to(self.device)
                tensors.append(t)

            feats = []

            with torch.no_grad():
                for b in tensors:
                    f = self.model.encode_image(b)
                    f = F.normalize(f, p=2, dim=-1)
                    feats.append(f)

            img_features = torch.cat(feats, dim=0)

            rand_texts = generate_random_text_tokens(num_random_texts)
            txt_inputs = clip.tokenize(rand_texts).to(self.device)

            with torch.no_grad():
                txt_features = self.model.encode_text(txt_inputs)
                txt_features = F.normalize(txt_features, p=2, dim=-1)

            img_features, txt_features = ensure_same_dtype(img_features, txt_features)

            sim = img_features @ txt_features.T

            sim_np = sim.cpu().numpy()

            for t in range(num_random_texts):
                results['class_similarities'][cname].append(float(np.mean(sim_np[:, t])))

            results['random_text_similarities'].extend(sim_np.flatten().tolist())

        elapsed = time.time() - start

        logger.info(f"Finished in {elapsed:.2f} seconds")

        if save_path:
            path = os.path.join(save_path, "random_text_on_real_images_analysis.json")

            with open(path, "w") as f:
                json.dump(
                    {'random_text_similarities': [float(v) for v in results['random_text_similarities']]},
                    f,
                    indent=4)

            logger.info(f"Saved results to {path}")

        return results

    def analyze_random_images_on_real_texts_batch(
        self,
        num_random_images_per_class=10,
        batch_size=64,
        save_path=None):

        logger.info("Random Image -> Real Text Similarity Analysis")
        logger.info("=" * 80)

        start = time.time()

        results = {
            'class_similarities': defaultdict(list),
            'random_image_similarities': [],
        }

        logger.info("Generating random images...")

        imgs = []
        imgs += generate_gaussian_noise_images_batch(num_random_images_per_class)
        imgs += generate_uniform_noise_images_batch(num_random_images_per_class)
        imgs += generate_random_images_batch(num_random_images_per_class)

        feats = []

        for i in tqdm(range(0, len(imgs), batch_size)):
            batch = imgs[i:i+batch_size]
            t = torch.stack([self.model.preprocess(im) for im in batch]).to(self.device)

            with torch.no_grad():
                f = self.model.encode_image(t)
                f = F.normalize(f, p=2, dim=-1)

            feats.append(f)

        img_features = torch.cat(feats, dim=0)

        with torch.no_grad():
            img_features, txt_features = ensure_same_dtype(img_features, self.text_features)

            sim = img_features @ txt_features.T

        sim_np = sim.cpu().numpy()

        for i in range(sim_np.shape[0]):
            sims = sim_np[i]

            results['random_image_similarities'].extend(sims.tolist())

            for c, cls in enumerate(self.class_names):
                results['class_similarities'][cls].append(float(sims[c]))

        elapsed = time.time() - start

        logger.info(f"Finished in {elapsed:.2f} seconds")

        if save_path:
            path = os.path.join(save_path, "random_images_on_real_texts_analysis.json")

            with open(path, "w") as f:
                json.dump(
                    {'random_image_similarities': [float(v) for v in results['random_image_similarities']]},
                    f,
                    indent=4)

            logger.info(f"Saved results to {path}")

        return results


def load_model_from_checkpoint(model, checkpoint_path, device):
    logger.info(f"Loading model from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    new_sd = {}

    for k, v in sd.items():
        if k.startswith("clip_model."):
            new_sd[k] = v
        else:
            new_sd["clip_model." + k] = v

    model.load_state_dict(new_sd, strict=False)

    logger.info("Model loaded successfully")

    return model


def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Similarity Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--forget_classes", type=str, nargs="*", default=[], help="List of forget classes")
    parser.add_argument("--num_samples_per_class", type=int, default=100, help="Number of samples per class")
    parser.add_argument("--num_random_texts", type=int, default=20, help="Number of random texts")
    parser.add_argument("--num_random_images_per_class", type=int, default=10, help="Number of random images per class")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    logger.info(f"Using device: {device}")

    forget_classes = args.forget_classes

    def forget_rule(sample):
        return sample["text"] in forget_classes

    dataset = UniversalDataset(
        "CIFAR100",
        root=args.data_root,
        train=False,
        download=True,
        label_names={i: n for i, n in enumerate(CIFAR100_LABEL_NAMES)}
    )

    dataset.set_forget_rule(forget_rule)

    model = CLIPModel().to(device)

    try:
        model = load_model_from_checkpoint(model, args.checkpoint, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    analyzer = CrossModalSimilarityAnalyzerBatch(model, device, CIFAR100_LABEL_NAMES)

    os.makedirs(args.save_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(args.save_dir, f"batch_{ts}")
    os.makedirs(save_path, exist_ok=True)

    text_results = analyzer.analyze_random_text_on_real_images_batch(
        dataset,
        num_samples_per_class=args.num_samples_per_class,
        num_random_texts=args.num_random_texts,
        batch_size=args.batch_size,
        save_path=save_path)

    image_results = analyzer.analyze_random_images_on_real_texts_batch(
        num_random_images_per_class=args.num_random_images_per_class,
        batch_size=args.batch_size,
        save_path=save_path)

    logger.info("Analysis finished")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
