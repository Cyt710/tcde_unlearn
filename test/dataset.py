import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from PIL import Image
import os
import pandas as pd
import torchvision
import json
import re
from typing import *
import string
import numpy as np


class CustomizedDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.forget_rule = lambda x: False

    def is_in_forget_set(self, sample) -> bool:
        return self.forget_rule(sample)

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_rule = fn
        return


class Flickr30kDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, *args, **kwargs):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file, delimiter=",")
        self.img_dir = img_dir
        self.forget_set_image_names = set()

        print(f"CSV column names: {self.img_labels.columns.tolist()}")
        print(f"CSV file shape: {self.img_labels.shape}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0].strip()
        caption = self.img_labels.iloc[idx, 2]

        if isinstance(caption, str):
            caption = caption.lower()
            caption = re.sub(f'[{re.escape(string.punctuation)}]', '', caption)
            caption = re.sub(r'\s+', ' ', caption).strip()
            words = caption.split()[:60]
            caption = ' '.join(words)
        else:
            caption = ""

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        is_in_forget_set = img_name in self.forget_set_image_names

        return {
            "image": image,
            "text": caption,
            "label": -1,
            "flag": is_in_forget_set
        }

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        retain_set_image_names = set()

        for idx in range(len(self)):
            img_name = self.img_labels.iloc[idx, 0].strip()
            caption = self.img_labels.iloc[idx, 2]

            if isinstance(caption, str):
                caption = caption.strip()
            else:
                caption = ""

            if fn(caption):
                self.forget_set_image_names.add(img_name)
            else:
                retain_set_image_names.add(img_name)

        print(
            f"Statistics: Forget set {len(self.forget_set_image_names)} images, "
            f"Retain set {len(retain_set_image_names)} images."
        )

        return

    def get_forget_indices(self):
        forget_indices = []
        for idx in range(len(self)):
            img_name = self.img_labels.iloc[idx, 0].strip()
            if img_name in self.forget_set_image_names:
                forget_indices.append(idx)
        return forget_indices

    def get_retain_indices(self):
        retain_indices = []
        for idx in range(len(self)):
            img_name = self.img_labels.iloc[idx, 0].strip()
            if img_name not in self.forget_set_image_names:
                retain_indices.append(idx)
        return retain_indices


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        self.label_names = kwargs.get("label_names", {})
        if "label_names" in kwargs:
            del kwargs["label_names"]
        super().__init__(*args, **kwargs)
        self.forget_rule = lambda x: False

    def is_in_forget_set(self, sample) -> bool:
        return self.forget_rule(sample)

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_rule = fn

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        text = self.label_names[label]
        sample = {"image": image, "text": text, "label": label}
        sample["flag"] = self.is_in_forget_set(sample)
        return sample


class CC12MDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, *args, **kwargs):
        super().__init__()
        with open(annotations_file, "r") as f:
            self.metadata = json.load(f)
        self.img_dir = img_dir
        self.forget_indices_set = set()

        self.backup_image_name = "00191154.jpg"
        self.backup_text = (
            "Vector seamless pattern with flower Chinese plum. "
            "Floral pattern with leaves flowers and branches of the tree "
            "Chinese plum. Design paper wallpaper and fabrics. "
            "Black red gold."
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata[idx]["img_name"]
        text = self.metadata[idx]["text"]

        if isinstance(text, str):
            caption = text.lower()
            caption = re.sub(f'[{re.escape(string.punctuation)}]', '', caption)
            caption = re.sub(r'\s+', ' ', caption).strip()
            words = caption.split()[:60]
            caption = ' '.join(words)
            text = caption
        else:
            text = ""

        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")

            if image.size[0] < 5 or image.size[1] < 5:
                img_path = os.path.join(self.img_dir, self.backup_image_name)
                image = Image.open(img_path).convert("RGB")
                text = self.backup_text

        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        sample = {
            "image": image,
            "text": text,
            "label": -1,
            "flag": idx in self.forget_indices_set
        }

        return sample

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_indices_set.clear()

        for idx in range(len(self)):
            img_name = self.metadata[idx]["img_name"]
            text = self.metadata[idx]["text"]

            if isinstance(text, str):
                caption = text.lower()
                caption = re.sub(f'[{re.escape(string.punctuation)}]', '', caption)
                caption = re.sub(r'\s+', ' ', caption).strip()
                words = caption.split()[:60]
                caption = ' '.join(words)
                text = caption
            else:
                text = ""

            sample = {"image": None, "text": text, "img_name": img_name}

            if fn(sample):
                self.forget_indices_set.add(idx)

        retain_count = len(self) - len(self.forget_indices_set)

        print(
            f"CC12M Statistics: Forget set {len(self.forget_indices_set)} images, "
            f"Retain set {retain_count} images."
        )

        self.forget_rule = lambda sample: sample.get("flag", False)
        return

    def get_forget_indices(self):
        return list(self.forget_indices_set)

    def get_retain_indices(self):
        return [
            idx for idx in range(len(self))
            if idx not in self.forget_indices_set
        ]


class UniversalDataset(Dataset):
    def __init__(self, dataset_name: str, *args: Any, **kwargs: Any):
        self.dataset_name_to_class = {
            "CIFAR100": CIFAR100Dataset,
            "Flickr30k": Flickr30kDataset,
            "CC12M": CC12MDataset,
        }

        DatasetCls = self.dataset_name_to_class.get(dataset_name, None)

        if DatasetCls:
            self.dataset = DatasetCls(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: True) -> None:
        self.dataset.set_forget_rule(fn)
        return

    def get_forget_indices(self):
        if hasattr(self.dataset, 'get_forget_indices'):
            return self.dataset.get_forget_indices()
        return []

    def get_retain_indices(self):
        if hasattr(self.dataset, 'get_retain_indices'):
            return self.dataset.get_retain_indices()
        return []


class BalancedDataLoader:

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size

        forget_indices = dataset.get_forget_indices()
        retain_indices = dataset.get_retain_indices()

        print(
            f"Balanced dataloader: {len(forget_indices)} forget samples, "
            f"{len(retain_indices)} retain samples"
        )

        self.forget_batch_size = max(1, batch_size // 2)
        self.retain_batch_size = batch_size - self.forget_batch_size

        self.forget_loader = DataLoader(
            Subset(dataset, forget_indices),
            batch_size=self.forget_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

        self.retain_loader = DataLoader(
            Subset(dataset, retain_indices),
            batch_size=self.retain_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

        self.forget_iter = self.infinite_iterator(self.forget_loader)
        self.retain_iter = self.infinite_iterator(self.retain_loader)

    def infinite_iterator(self, dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        flags = [item["flag"] for item in batch]
        labels = [item["label"] for item in batch]
        return images, texts, flags, labels

    def __iter__(self):
        return self

    def __next__(self):
        forget_batch = next(self.forget_iter)
        retain_batch = next(self.retain_iter)

        images = forget_batch[0] + retain_batch[0]
        texts = forget_batch[1] + retain_batch[1]
        flags = forget_batch[2] + retain_batch[2]
        labels = forget_batch[3] + retain_batch[3]

        return images, texts, flags, labels

    def __len__(self):
        return max(len(self.forget_loader), len(self.retain_loader))


def UniversalDataLoader(dataset, *args, **kwargs):

    if hasattr(dataset, 'get_forget_indices') and hasattr(dataset, 'get_retain_indices'):

        forget_indices = dataset.get_forget_indices()
        retain_indices = dataset.get_retain_indices()

        if len(forget_indices) > 0 and len(retain_indices) > 0:
            print(
                f"Using balanced dataloader: {len(forget_indices)} forget samples, "
                f"{len(retain_indices)} retain samples"
            )
            return BalancedDataLoader(dataset, *args, **kwargs)

    def collate_fn(batch):

        images = []
        texts = []
        flags = []
        labels = []

        for sample in batch:
            images.append(sample["image"])
            texts.append(sample["text"])
            flags.append(sample["flag"])
            labels.append(sample["label"])

        return images, texts, flags, labels

    return DataLoader(dataset, collate_fn=collate_fn, *args, **kwargs)

# Usage example
if __name__ == "__main__":

    metadata_path = ""
    image_dir = ""

    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        exit(1)

    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        exit(1)

    dataset = UniversalDataset(
        "CC12M",
        metadata_path,
        image_dir,
    )

    dataset.set_forget_rule(
        lambda sample: "woman" in sample["text"].lower()
        or "women" in sample["text"].lower()
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Forget set size: {len(dataset.get_forget_indices())}")
    print(f"Retain set size: {len(dataset.get_retain_indices())}")

    dataloader = UniversalDataLoader(dataset, batch_size=32, shuffle=True)

    for i, (images, texts, flags, labels) in enumerate(dataloader):

        print(f"Batch {i}: {len(images)} images, {len(texts)} texts")
        print(f"Forget samples in batch: {sum(flags)}")

        if i >= 2:
            break