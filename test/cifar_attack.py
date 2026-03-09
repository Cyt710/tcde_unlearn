# ---------- Simplified evaluation function (only evaluate forget set, minimum cosine similarity prediction) ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import clip
import numpy as np
from typing import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import UniversalDataset, UniversalDataLoader
from testcifar import CLIPModel

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

def evaluate_forget_set_only(model, data_loader, device, class_names, forget_classes):
    model.eval()

    results = {
        'total_forget_samples': 0,
        'successful_attacks': 0,
        'attack_success_rate': 0.0,
        'prediction_distribution': {class_name: 0 for class_name in class_names},
        'forget_class_predictions': {class_name: 0 for class_name in forget_classes}
    }

    forget_class_indices = [class_names.index(c) for c in forget_classes if c in class_names]
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)

    with torch.no_grad():
        for batch in data_loader:
            images, texts, flags, labels = batch
            image_tensor = torch.stack([model.preprocess(img) for img in images]).to(device)
            flags_tensor = torch.tensor(flags).to(device)

            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)

            curr_similarity = image_features @ text_features.T

            for i in range(len(images)):
                is_forget = flags_tensor[i].item()
                
                if not is_forget:
                    continue
                    
                results['total_forget_samples'] += 1
                
                sim_i = curr_similarity[i]
                pred_idx = sim_i.argmin().item()
                pred_class = class_names[pred_idx]
                
                results['prediction_distribution'][pred_class] += 1
                
                if pred_idx in forget_class_indices:
                    results['successful_attacks'] += 1
                    results['forget_class_predictions'][pred_class] += 1

    if results['total_forget_samples'] > 0:
        results['attack_success_rate'] = 100.0 * results['successful_attacks'] / results['total_forget_samples']
    
    return results

def print_forget_set_results(results, forget_classes, model_name="Model"):
    print("\n" + "="*70)
    print(f"{model_name} - Forget Set Attack Evaluation")
    print("="*70)
    print(f"Prediction method: Minimum Cosine Similarity")
    print(f"Success criteria: Prediction falls within forget classes {forget_classes}")
    print("-"*70)
    
    print(f"Total forget samples: {results['total_forget_samples']}")
    print(f"Successful attacks: {results['successful_attacks']}")
    print(f"Attack success rate: {results['attack_success_rate']:.2f}%")
    
    print("\nForget Set Prediction Distribution:")
    print(f"{'Predicted Class':<20} {'Count':<10} {'Percentage(%)':<10} {'Is Forget Class':<15}")
    print("-"*70)
    
    sorted_predictions = sorted(
        results['prediction_distribution'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for class_name, count in sorted_predictions:
        if count > 0:
            percentage = 100.0 * count / results['total_forget_samples'] if results['total_forget_samples'] > 0 else 0.0
            is_forget_class = "Yes" if class_name in forget_classes else "No"
            print(f"{class_name:<20} {count:<10} {percentage:.2f}% {is_forget_class:<15}")
    
    print("\nForget Class Prediction Statistics:")
    print(f"{'Forget Class':<20} {'Prediction Count':<10} {'Percentage(%)':<10}")
    print("-"*70)
    
    total_predictions_to_forget = sum(results['forget_class_predictions'].values())
    for class_name in forget_classes:
        count = results['forget_class_predictions'][class_name]
        if count > 0 or class_name in forget_classes:
            percentage_in_total = 100.0 * count / results['total_forget_samples'] if results['total_forget_samples'] > 0 else 0.0
            print(f"{class_name:<20} {count:<10} {percentage_in_total:.2f}%")
    
    print("\n" + "="*70)
    print("Attack Analysis:")
    print(f"Attack success rate: {results['attack_success_rate']:.2f}%")
  
    top_predictions = sorted_predictions[:5]
    print(f"\nTop 5 Predicted Classes:")
    for i, (class_name, count) in enumerate(top_predictions, 1):
        percentage = 100.0 * count / results['total_forget_samples'] if results['total_forget_samples'] > 0 else 0.0
        is_target = "← Target Class" if class_name in forget_classes else ""
        print(f"  {i}. {class_name:<15} {percentage:.1f}% {is_target}")

def load_model_from_checkpoint(model, checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Successfully loaded model weights (epoch {checkpoint.get('epoch', 'unknown')})")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Successfully loaded CLIP format model weights")
    else:
        state_dict = checkpoint
        print("Successfully loaded model state_dict")
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('clip_model.'):
            new_state_dict[key] = value
        else:
            new_key = f'clip_model.{key}'
            new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded and adapted model weights")
    
    return model

#Usage Example:
def main():
    config = {
        "dataset": {
            "name": "CIFAR100",
            "forget_classes": ["apple"],
            "data_root": "./data"
        },
        "training": {
            "batch_size": 64,
        },
        "dataloader": {
            "shuffle": False,
            "num_workers": 4
        },
        "misc": {
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    
    device = torch.device(config["misc"]["device"])
    print(f"Using device: {device}")
    
    forget_classes = config["dataset"]["forget_classes"]
    print(f"Forget classes: {forget_classes}")
    print(f"Evaluation target: Prediction to these classes counts as successful attack")
    print(f"Prediction method: Minimum cosine similarity")
    
    def forget_rule(sample):
        return sample["text"] in forget_classes
    
    print("\nLoading test dataset...")
    test_dataset = UniversalDataset(
        "CIFAR100",
        root=config["dataset"]["data_root"],
        train=False,
        download=True,
        label_names={i: name for i, name in enumerate(CIFAR100_LABEL_NAMES)}
    )
    test_dataset.set_forget_rule(forget_rule)
    
    forget_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i]["flag"])
    print(f"Forget set samples in test set: {forget_count}")
    
    test_loader = UniversalDataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["dataloader"]["shuffle"],
        num_workers=config["dataloader"]["num_workers"]
    )
    
    print("\nInitializing model...")
    model = CLIPModel().to(device)
    
    checkpoint_path = "your path to the trained model checkpoint"
    
    try:
        model = load_model_from_checkpoint(model, checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("\nEvaluating forget set attack performance...")
    results = evaluate_forget_set_only(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=CIFAR100_LABEL_NAMES,
        forget_classes=forget_classes
    )
    
    print_forget_set_results(results, forget_classes, model_name="Trained Model")
    
    output_file = "forget_set_attack_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'forget_classes': forget_classes,
            'total_forget_samples': results['total_forget_samples'],
            'successful_attacks': results['successful_attacks'],
            'attack_success_rate': results['attack_success_rate'],
            'prediction_distribution': results['prediction_distribution'],
            'forget_class_predictions': results['forget_class_predictions']
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()