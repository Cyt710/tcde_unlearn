import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from copy import deepcopy
import os
import sys
import json
import clip
from typing import *
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import UniversalDataset, UniversalDataLoader
from model import CLIPModel, CLIPEraseLoss, CheckpointHelper, MemoryTrainedModelLoader
from zero_shot import train_zero_shot

def create_combined_dataset(train_forget_path, train_retain_path, val_forget_path, val_retain_path, forget_keyword):
    print("Loading training sets...")
    
    train_forget_dataset = UniversalDataset(
        "Flickr30k",
        annotations_file=os.path.join(train_forget_path, 'captions.csv'),
        img_dir=train_forget_path     
    )
    
    train_retain_dataset = UniversalDataset(
        "Flickr30k", 
        annotations_file=os.path.join(train_retain_path, 'captions.csv'),
        img_dir=train_retain_path
    )
    
    def forget_rule(caption):
        if isinstance(caption, str) and forget_keyword.lower() in caption.lower():
            return True
        return False
    
    train_forget_dataset.set_forget_rule(forget_rule)
    train_retain_dataset.set_forget_rule(forget_rule)
    
    print(f"Training forget set original size: {len(train_forget_dataset)}")
    print(f"Training retain set original size: {len(train_retain_dataset)}")
    
    train_dataset = ConcatDataset([train_forget_dataset, train_retain_dataset])
    
    print("Loading validation sets...")
    
    val_forget_dataset = UniversalDataset(
        "Flickr30k",
        annotations_file=os.path.join(val_forget_path, 'captions.csv'),
        img_dir=val_forget_path
    )
    
    val_retain_dataset = UniversalDataset(
        "Flickr30k", 
        annotations_file=os.path.join(val_retain_path, 'captions.csv'),
        img_dir=val_retain_path
    )
    
    val_forget_dataset.set_forget_rule(forget_rule)
    val_retain_dataset.set_forget_rule(forget_rule)
    
    print(f"Validation forget set original size: {len(val_forget_dataset)}")
    print(f"Validation retain set original size: {len(val_retain_dataset)}")
    
    val_dataset = ConcatDataset([val_forget_dataset, val_retain_dataset])
    
    train_forget_count = len(train_forget_dataset.get_forget_indices())
    train_retain_count = len(train_forget_dataset.get_retain_indices()) + len(train_retain_dataset.get_retain_indices())
    val_forget_count = len(val_forget_dataset.get_forget_indices())
    val_retain_count = len(val_forget_dataset.get_retain_indices()) + len(val_retain_dataset.get_retain_indices())
    
    print(f"\nDataset details:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"  - Forget set: {train_forget_count} samples")
    print(f"  - Retain set: {train_retain_count} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"  - Forget set: {val_forget_count} samples")
    print(f"  - Retain set: {val_retain_count} samples")
    
    return train_dataset, val_dataset

def trainflick():
    
    config = {
        "mode": "train",
        "task": "zero_shot",
        "dataset": {
            "name": "Flickr30k",
            "forget_keyword": "mountain",
            "base_dir": "/home/cyt/cliperase/dataset/CIFAR100/data",
            "train_forget": "train/forget_set",
            "train_retain": "train/retain_set", 
            "val_forget": "val/forget_set",
            "val_retain": "val/retain_set"
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1e-6,
            "weight_decay": 0.0,
            "epochs": 20
        },
        "dataloader": {
            "shuffle": True,
            "num_workers": 8
        },
        "misc": {
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "cliperase": {
            "lambda_rm": 0.0,
            "lambda_fm": 0.0,
            "lambda_cm": 0.0,
            "lambda_unif": 0.0,
            "lambda_dist": 0.0,
            "temperature": 0.07
        },
        "pretrained_model": {
            "checkpoint_path": "/home/cyt/cliperase/dataset/CIFAR100/checkpoints/flickr_normal_20260128_132055/checkpoint_epoch_50.pt"
        }
    }
    
    torch.manual_seed(config["misc"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["misc"]["seed"])
    
    print("Configuration:", json.dumps(config, indent=2))
    
    base_dir = config["dataset"]["base_dir"]
    train_forget_path = os.path.join(base_dir, config["dataset"]["train_forget"])
    train_retain_path = os.path.join(base_dir, config["dataset"]["train_retain"])
    val_forget_path = os.path.join(base_dir, config["dataset"]["val_forget"])
    val_retain_path = os.path.join(base_dir, config["dataset"]["val_retain"])
    
    print("\nVerifying data paths...")
    for path_name, path in [
        ("Training forget set", train_forget_path),
        ("Training retain set", train_retain_path),
        ("Validation forget set", val_forget_path),
        ("Validation retain set", val_retain_path)
    ]:
        csv_path = os.path.join(path, 'captions.csv')
        if os.path.exists(path) and os.path.exists(csv_path):
            image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            csv_data = pd.read_csv(csv_path, delimiter=",")
            print(f" {path_name}: {len(image_files)} images, {len(csv_data)} CSV rows")
        else:
            print(f"{path_name}: Path or CSV file does not exist - {path}")
            return None
    
    train_dataset, val_dataset = create_combined_dataset(
        train_forget_path, train_retain_path, val_forget_path, val_retain_path, "mountain"
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Dataset is empty, please check data paths")
        return None
    
    expected_train_size = len(pd.read_csv(os.path.join(train_forget_path, 'captions.csv'))) + \
                         len(pd.read_csv(os.path.join(train_retain_path, 'captions.csv')))
    expected_val_size = len(pd.read_csv(os.path.join(val_forget_path, 'captions.csv'))) + \
                       len(pd.read_csv(os.path.join(val_retain_path, 'captions.csv')))
    
    print(f"\nExpected dataset sizes:")
    print(f"Training set: {expected_train_size} samples")
    print(f"Validation set: {expected_val_size} samples")
    print(f"Actual dataset sizes:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    train_loader = UniversalDataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=config["dataloader"]["shuffle"],
        num_workers=config["dataloader"]["num_workers"]
    )
    
    val_loader = UniversalDataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"]
    )
    
    model = CLIPModel(device=config["misc"]["device"])
    print("CLIPModel initialized successfully")
    
    model = MemoryTrainedModelLoader.load_memory_trained_model(
        model, 
        config["pretrained_model"]["checkpoint_path"], 
        device=config["misc"]["device"]
    )
    original_model = deepcopy(model)
    
    for param in original_model.parameters():
        param.requires_grad = False
    
    criterion = CLIPEraseLoss(
        lambda_rm=0.0,
        lambda_fm=0.0,
        lambda_cm=0.0,
        lambda_unif=0.0,
        lambda_dist=0.0,
        lambda_ga=0.0,
        lambda_gd=0.0,
        lambda_kl=1.0,
        temperature=0.07
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.98),
        weight_decay=config["training"]["weight_decay"]
    )
    
    checkpoint_dir = "./flickr_checkpoints_KL"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_helper = CheckpointHelper(checkpoint_dir)
    
    print("\n" + "="*50)
    print("Starting training on Flickr30k dataset...")
    print("="*50)
    
    best_val_loss, best_forget_accuracy, best_retain_accuracy = train_zero_shot(
        model=model,
        original_model=original_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config["misc"]["device"],
        epochs=config["training"]["epochs"],
        checkpoint_helper=checkpoint_helper,
        mode="flickr",
    )
    
    print("\n" + "="*60)
    print("Flickr30k training completed!")
    print("="*60)
    print(f"Best validation results:")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    
    return {
        'config': config,
        'best_val_loss': best_val_loss,
        'model': model,
        'checkpoint_dir': checkpoint_dir,
        'device_info': {
            'device': str(config["misc"]["device"]),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

if __name__ == "__main__":
    results = trainflick()