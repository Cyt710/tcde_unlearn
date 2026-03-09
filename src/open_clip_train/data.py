import ast
import json
import logging
import math
import os
import random
import re
import string
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info, Subset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


@dataclass
class DataSample:
    image: Any
    text: str
    is_forget: bool = False  
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self):
        return f"DataSample(image={self.image.shape}, text={self.text.shape}, is_forget={self.is_forget})"


def cliperase_collate_fn(batch):
    if not batch:
        return [torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])]
    
    first_elem = batch[0]
    batch_size = len(batch)
    
    try:
        if isinstance(first_elem, DataSample):
            images = torch.stack([sample.image for sample in batch])
            texts = torch.stack([sample.text for sample in batch])
            
            labels = []
            for sample in batch:
                if sample.metadata and isinstance(sample.metadata, dict):
                    label = sample.metadata.get('label', 0)
                    labels.append(label)
                else:
                    labels.append(0)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            is_forget = torch.tensor([sample.is_forget for sample in batch], dtype=torch.bool)
            
            return [images, texts, is_forget, labels_tensor]
        
        elif isinstance(first_elem, (tuple, list)):
            if len(first_elem) < 2:
                raise ValueError(f"Tuple/list format requires at least 2 elements, got {len(first_elem)}")
            
            images = torch.stack([item[0] for item in batch])
            texts = torch.stack([item[1] for item in batch])
            
            if len(first_elem) >= 3:
                is_forget = torch.tensor([item[2] for item in batch], dtype=torch.bool)
            else:
                is_forget = torch.zeros(batch_size, dtype=torch.bool)
            
            if len(first_elem) >= 4:
                labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
            else:
                labels = torch.zeros(batch_size, dtype=torch.long)
            
            return [images, texts, is_forget, labels]
        
        elif isinstance(first_elem, dict):
            images_list = []
            texts_list = []
            is_forget_list = []
            labels_list = []
            
            for item in batch:
                if 'image' in item:
                    if isinstance(item['image'], torch.Tensor):
                        images_list.append(item['image'])
                    else:
                        import torchvision.transforms as transforms
                        transform = transforms.ToTensor()
                        images_list.append(transform(item['image']))
                else:
                    images_list.append(torch.randn(3, 224, 224))
                
                if 'text' in item:
                    if isinstance(item['text'], torch.Tensor):
                        texts_list.append(item['text'])
                    else:
                        texts_list.append(torch.tensor([ord(c) for c in str(item['text'])][:77] + [0]*(77-len(str(item['text'])))))
                else:
                    texts_list.append(torch.zeros(77, dtype=torch.long))
                
                if 'is_forget' in item:
                    is_forget_list.append(bool(item['is_forget']))
                elif 'flag' in item:
                    is_forget_list.append(bool(item['flag']))
                else:
                    is_forget_list.append(False)
                
                if 'label' in item:
                    labels_list.append(item['label'])
                else:
                    labels_list.append(0)
            
            images = torch.stack(images_list)
            texts = torch.stack(texts_list)
            is_forget = torch.tensor(is_forget_list, dtype=torch.bool)
            labels = torch.tensor(labels_list, dtype=torch.long)
            
            return [images, texts, is_forget, labels]
        
        else:
            logging.warning(f"Unknown batch format: {type(first_elem)}, creating default data")
            images = torch.randn(batch_size, 3, 224, 224)
            texts = torch.randint(0, 1000, (batch_size, 77), dtype=torch.long)
            is_forget = torch.zeros(batch_size, dtype=torch.bool)
            labels = torch.zeros(batch_size, dtype=torch.long)
            
            return [images, texts, is_forget, labels]
    
    except Exception as e:
        logging.error(f"Error in cliperase_collate_fn: {e}, creating fallback batch")
        images = torch.randn(batch_size, 3, 224, 224)
        texts = torch.randint(0, 1000, (batch_size, 77), dtype=torch.long)
        is_forget = torch.zeros(batch_size, dtype=torch.bool)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        return [images, texts, is_forget, labels]


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None, forget_rule=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        
        self.forget_rule = forget_rule or (lambda x: False)
        self.forget_flags = []
        for caption in self.captions:
            self.forget_flags.append(self.forget_rule(str(caption)))
        
        logging.debug(f'Done loading data. Forget samples: {sum(self.forget_flags)}/{len(self)}')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            image = Image.open(str(self.images[idx])).convert('RGB')
            image = self.transforms(image)
        except Exception as e:
            logging.warning(f"Error loading image {self.images[idx]}: {e}")
            image = self.transforms(Image.new('RGB', (224, 224), color='white'))
        
        texts = self.tokenize([str(self.captions[idx])])[0]
        
        return DataSample(
            image=image,
            text=texts,
            is_forget=self.forget_flags[idx],
            metadata={"index": idx, "image_path": self.images[idx]}
        )


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


# ==================== Universal Dataset Support ====================

class CustomizedDataset(Dataset):
 
    def __init__(self):
        super().__init__()
        self.forget_rule = lambda x: False

    def is_in_forget_set(self, sample) -> bool:
        return self.forget_rule(sample)

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_rule = fn
        return

    def get_forget_indices(self):
        
        return []

    def get_retain_indices(self):

        return []


class Flickr30kDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, transform=None, tokenizer=None, *args, **kwargs):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file, delimiter=",")
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.forget_set_image_names = set()
        
        logging.info(f"Flickr30k CSV column names: {self.img_labels.columns.tolist()}")
        logging.info(f"Flickr30k CSV files shape: {self.img_labels.shape}")
        

        required_cols = ['image_name', 'comment']
        for col in required_cols:
            if col not in self.img_labels.columns:
                logging.warning(f"CSV file is missing column: {col}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_name = str(self.img_labels.iloc[idx]['image_name']).strip()
        caption = str(self.img_labels.iloc[idx]['comment']).strip()

        if isinstance(caption, str):

            caption = caption.lower()
            caption = re.sub(f'[{re.escape(string.punctuation)}]', '', caption)
            caption = re.sub(r'\s+', ' ', caption).strip()
            words = caption.split()[:60]
            caption = ' '.join(words)
        else:
            caption = ""

        img_path = os.path.join(self.img_dir, img_name)
        
        if idx < 5: 
            logging.debug(f"Loading image {idx}: {img_path}, exists: {os.path.exists(img_path)}")
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            raise ValueError("Failure loading image, check the path and file format.")

        if self.tokenizer:
            text_tokens = self.tokenizer([caption])[0]
        else:
            text_tokens = caption

        is_in_forget_set = img_name in self.forget_set_image_names

        return DataSample(
            image=image,
            text=text_tokens,
            is_forget=is_in_forget_set,
            metadata={
                "index": idx,
                "label": -1,
                "image_path": img_path,
                "caption": caption
            }
        )

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        retain_set_image_names = set()

        for idx in range(len(self)):
            img_name = str(self.img_labels.iloc[idx]['image_name']).strip()
            caption = str(self.img_labels.iloc[idx]['comment']).strip()

            sample = {"image": None, "text": caption, "img_name": img_name}
            
            if fn(sample):
                self.forget_set_image_names.add(img_name)
            else:
                retain_set_image_names.add(img_name)

        logging.info(
            f"Flickr30k Statistics: Forget set {len(self.forget_set_image_names)} images, "
            f"Retain set {len(retain_set_image_names)}."
        )
        
        return

    def get_forget_indices(self):

        forget_indices = []
        for idx in range(len(self)):
            img_name = str(self.img_labels.iloc[idx]['image_name']).strip()
            if img_name in self.forget_set_image_names:
                forget_indices.append(idx)
        return forget_indices

    def get_retain_indices(self):

        retain_indices = []
        for idx in range(len(self)):
            img_name = str(self.img_labels.iloc[idx]['image_name']).strip()
            if img_name not in self.forget_set_image_names:
                retain_indices.append(idx)
        return retain_indices


class CIFAR100Dataset(CustomizedDataset):
    def __init__(self, root="./data", train=True, transform=None, tokenizer=None, label_names=None, *args, **kwargs):
        super().__init__()
        self.dataset = datasets.CIFAR100(
            root=root, 
            train=train,
            download=True,
            transform=transform
        )
        self.transform = transform
        self.tokenizer = tokenizer
        

        self.label_names = label_names or [
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
        
        self.forget_indices_set = set()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.label_names[label]
        
        text = f"a photo of a {class_name}"
        
        if self.tokenizer:
            text_tokens = self.tokenizer([text])[0]
        else:
            text_tokens = text
        
        is_forget = idx in self.forget_indices_set
        
        return DataSample(
            image=image,
            text=text_tokens,
            is_forget=is_forget,
            metadata={
                "index": idx,
                "label": label,
                "class_name": class_name,
                "split": "train" if self.dataset.train else "test"
            }
        )

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:

        self.forget_indices_set.clear()
        
        for idx in range(len(self)):
            image, label = self.dataset[idx]
            class_name = self.label_names[label]
            text = f"a photo of a {class_name}"
            

            sample = {"image": image, "text": text, "label": label, "class_name": class_name}
            
            if fn(sample):
                self.forget_indices_set.add(idx)
        
        retain_count = len(self) - len(self.forget_indices_set)
        logging.info(
            f"CIFAR-100 Statistics: Forget set {len(self.forget_indices_set)} samples, "
            f"Retain set {retain_count} samples."
        )
        return

    def get_forget_indices(self):

        return list(self.forget_indices_set)

    def get_retain_indices(self):

        return [idx for idx in range(len(self)) if idx not in self.forget_indices_set]


class CC12MDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, transform=None, tokenizer=None, *args, **kwargs):
        super().__init__()
        with open(annotations_file, "r") as f:
            self.metadata = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.forget_indices_set = set()  


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
        else:
            caption = ""

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        
        if self.tokenizer:
            text_tokens = self.tokenizer([caption])[0]
        else:
            text_tokens = caption

        is_forget = idx in self.forget_indices_set

        return DataSample(
            image=image,
            text=text_tokens,
            is_forget=is_forget,
            metadata={
                "index": idx,
                "label": -1,
                "image_path": img_path,
                "caption": caption,
                "img_name": img_name
            }
        )

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
            else:
                caption = ""
             
            sample = {"image": None, "text": caption, "img_name": img_name}
            
            if fn(sample):
                self.forget_indices_set.add(idx)
        
        retain_count = len(self) - len(self.forget_indices_set)
        logging.info(
            f"CC12M Statistics: Forget set {len(self.forget_indices_set)} images, "
            f"Retain set {retain_count} images."
        )
        return

    def get_forget_indices(self):
        
        return list(self.forget_indices_set)

    def get_retain_indices(self):

        return [idx for idx in range(len(self)) if idx not in self.forget_indices_set]


class UniversalDataset(Dataset):
    def __init__(self, dataset_name: str, transform=None, tokenizer=None, *args: Any, **kwargs: Any):
        self.dataset_name = dataset_name
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.dataset_name_to_class = {
            "cifar": CIFAR100Dataset,
            "flickr": Flickr30kDataset,
            "cc12m": CC12MDataset,
        }
        
        DatasetCls = self.dataset_name_to_class.get(dataset_name, None)
        if DatasetCls:
            if dataset_name == "cifar":
                self.dataset = DatasetCls(
                    transform=transform, 
                    tokenizer=tokenizer,
                    *args, **kwargs
                )
            elif dataset_name == "flickr":
                self.dataset = DatasetCls(
                    *args, 
                    transform=transform,
                    tokenizer=tokenizer,
                    **kwargs
                )
            elif dataset_name == "cc12m":
                self.dataset = DatasetCls(
                    *args,
                    transform=transform,
                    tokenizer=tokenizer,
                    **kwargs
                )
            else:
                raise ValueError(f"unsupported dataset: {dataset_name}")

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



def UniversalDataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, 
                       use_balanced_loader=True, *args, **kwargs):
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cliperase_collate_fn,
        *args, **kwargs
    )



def get_universal_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, forget_rule=None):
   
    dataset_name = getattr(args, 'dataset_type', 'CIFAR100')
    
    dataset_args = {}
    
    if dataset_name == "cifar":
        dataset_args = {
            "root": getattr(args, 'data_dir', './data'),
            "train": is_train,
        }
    elif dataset_name == "flickr":
        if is_train:
            annotations_file = getattr(args, 'train_data', None)
        else:
            annotations_file = getattr(args, 'val_data', None)
        
        if not annotations_file:
            raise ValueError(f"Flickr30k requires annotations_file path")
        
        img_dir = getattr(args, 'img_dir', './data/flickr30k')
        dataset_args = {
            "annotations_file": annotations_file,
            "img_dir": img_dir,
        }
    elif dataset_name == "cc12m":
        if is_train:
            annotations_file = getattr(args, 'train_data', None)
        else:
            annotations_file = getattr(args, 'val_data', None)
        
        if not annotations_file:
            raise ValueError(f"CC12M requires annotations_file path")
        
        img_dir = getattr(args, 'img_dir', './data/cc12m')
        dataset_args = {
            "annotations_file": annotations_file,
            "img_dir": img_dir,
        }
    else:
        raise ValueError(f"Unsupported universal dataset: {dataset_name}")
    
    dataset = UniversalDataset(
        dataset_name=dataset_name,
        transform=preprocess_fn,
        tokenizer=tokenizer,
        **dataset_args
    )
    
    
    if forget_rule:
        dataset.set_forget_rule(forget_rule)
    
    num_samples = len(dataset)
    
    sampler = None
    if args.distributed and is_train:
        sampler = DistributedSampler(dataset)
    
    shuffle = is_train and sampler is None
    
    use_balanced = getattr(args, 'use_balanced_loader', True)
    
    dataloader = UniversalDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        use_balanced_loader=use_balanced,
        sampler=sampler
    )
    
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "cifar":
        return get_universal_dataset  
    elif dataset_type == "flickr":
        return get_universal_dataset 
    elif dataset_type == "cc12m":
        return get_universal_dataset  
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        elif ext in ['json']:
            return get_universal_dataset  
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None, forget_rule=None, create_forget_rule_fn=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    
    forget_fn = forget_rule
    if hasattr(args, 'forget_rule') and args.forget_rule:
        if create_forget_rule_fn is not None:
            forget_fn = create_forget_rule_fn(args)
        else:
            forget_fn = create_forget_rule_from_args(args)
    
    if args.train_data or getattr(args, 'dataset_type', None) in ["synthetic", "cc12m", "cifar", "flickr"]:
        dataset_type = getattr(args, 'dataset_type', 'csv')
        logging.info(f"create training dataset: {dataset_type}")
        
        data["train"] = get_dataset_fn(args.train_data, dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, 
            tokenizer=tokenizer, forget_rule=forget_fn
        )
    
    if args.val_data or getattr(args, 'dataset_type', None) in ["synthetic", "cc12m", "cifar", "flickr"]:
        dataset_type = getattr(args, 'dataset_type', 'csv')
        logging.info(f"create validation dataset: {dataset_type}")
        
        data["val"] = get_dataset_fn(args.val_data, dataset_type)(
            args, preprocess_val, is_train=False, 
            tokenizer=tokenizer, forget_rule=forget_fn
        )
    
    
    if getattr(args, 'imagenet_val', None) is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val", forget_rule=forget_fn)
    
    if getattr(args, 'imagenet_v2', None) is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2", forget_rule=forget_fn)
    
    logging.info(f"dataset loaded: {list(data.keys())}")
    return data


def create_forget_rule_from_args(args):
    
    if not getattr(args, 'forget_rule', None):
        return None
    
    forget_rule_type = args.forget_rule
    
    if forget_rule_type == "keyword_based":
        def forget_rule(sample):
            if not getattr(args, 'forget_keywords', None):
                return False
            keywords = [k.strip() for k in args.forget_keywords.split(',')]
            text = sample.get("text", "").lower()
            return any(keyword.lower() in text for keyword in keywords)
        return forget_rule
    
    elif forget_rule_type == "class_based":
        def forget_rule(sample):
            class_name = sample.get("class_name", "").lower()
            if "woman" in class_name or "women" in class_name:
                return True
            return False
        return forget_rule
    
    else:
        logging.warning(f"unsupported forget rule type: {forget_rule_type}")
        return None


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, forget_rule=None):  
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        forget_rule=forget_rule  
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=cliperase_collate_fn 
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split, forget_rule=None): 
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        collate_fn=cliperase_collate_fn  
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, forget_rule=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    def apply_forget_rule(sample):

        text = sample["text"]
        is_forget = forget_rule(text) if forget_rule else False
        return DataSample(
            image=sample["image"],
            text=text,
            is_forget=is_forget,
            metadata={"__key__": sample.get("__key__", "")}
        )

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
  
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        #wds.to_tuple("image", "text"),修改不返回元组，返回datasample
        wds.map(apply_forget_rule),
        wds.batched(args.batch_size, partial=not is_train)
    ])


    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, forget_rule=None):  
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        forget_rule=forget_rule  
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=cliperase_collate_fn 
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
            forget_rule=None
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size
        self.tokenize = tokenizer
        self.forget_rule = forget_rule or (lambda x: False)


        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        #return image, self.preprocess_txt(self.caption)
      
        is_forget = self.forget_rule(self.caption)
        return DataSample(
            image=image,
            text=self.preprocess_txt(self.caption),
            is_forget=is_forget,
            metadata={"index": idx, "synthetic": True}
        )



#def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None)
def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, forget_rule=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, 
        image_size=image_size, 
        dataset_size=args.train_num_samples, 
        tokenizer=tokenizer,
        forget_rule=forget_rule 
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=cliperase_collate_fn  
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)




        
