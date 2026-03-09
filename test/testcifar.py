import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from copy import deepcopy
import os
import sys
import math
import json
import clip
import numpy as np
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import UniversalDataset, UniversalDataLoader

class CLIPModel(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", device=None):  
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.clip_model, self.preprocess = clip.load(
            model_name, 
            device=str(self.device)  
        )
        
        self.clip_model = self.clip_model.float()
        self.image_encoder = self.clip_model.visual
        self.text_encoder = self.clip_model

        for param in self.parameters():
            param.requires_grad = True

    def encode_image(self, images):
        if isinstance(images, torch.Tensor) and images.device != self.device:
            images = images.to(self.device)
        return self.image_encoder(images.float())

    def encode_text(self, texts):
        if isinstance(texts, torch.Tensor) and texts.device != self.device:
            texts = texts.to(self.device)
        return self.text_encoder.encode_text(texts)

    def forward(self, images, texts):
        img_f = self.encode_image(images)
        txt_f = self.encode_text(texts)
        return img_f, txt_f

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

def evaluate_zero_shot_per_class(model, data_loader, device, class_names):
    model.eval()
    
    class_stats = {}
    for class_name in class_names:
        class_stats[class_name] = {
            'total': 0,
            'correct': 0,
            'forget_total': 0,
            'forget_correct': 0,
            'retain_total': 0,
            'retain_correct': 0
        }
    
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
            
            image_features, text_features = ensure_same_dtype(image_features, text_features)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)
            
            true_labels = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            for i in range(len(images)):
                true_class = texts[i]
                is_forget = flags_tensor[i].item()
                is_correct = (predictions[i] == true_labels[i]).item()
                
                class_stats[true_class]['total'] += 1
                if is_correct:
                    class_stats[true_class]['correct'] += 1
                
                if is_forget:
                    class_stats[true_class]['forget_total'] += 1
                    if is_correct:
                        class_stats[true_class]['forget_correct'] += 1
                else:
                    class_stats[true_class]['retain_total'] += 1
                    if is_correct:
                        class_stats[true_class]['retain_correct'] += 1
    
    for class_name in class_names:
        stats = class_stats[class_name]
        stats['overall_accuracy'] = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['forget_accuracy'] = 100.0 * stats['forget_correct'] / stats['forget_total'] if stats['forget_total'] > 0 else 0.0
        stats['retain_accuracy'] = 100.0 * stats['retain_correct'] / stats['retain_total'] if stats['retain_total'] > 0 else 0.0
    
    return class_stats

def evaluate_retrieval_per_class(model, data_loader, device, class_names, k_list=[1]):
    model.eval()
    
    class_stats = {}
    for class_name in class_names:
        class_stats[class_name] = {
            'total': 0,
            'forget_total': 0,
            'retain_total': 0,
            'recall_at_1': 0,
            'forget_recall_at_1': 0,
            'retain_recall_at_1': 0
        }
    
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
           
            image_features, text_features = ensure_same_dtype(image_features, text_features)
            similarity = image_features @ text_features.T  
            
            true_labels = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            for i in range(len(images)):
                true_class = texts[i]
                is_forget = flags_tensor[i].item()
                true_label_idx = true_labels[i]
                
                sim_scores = similarity[i]
                sorted_indices = torch.argsort(sim_scores, descending=True)
                
                rank_1_correct = (sorted_indices[0] == true_label_idx).item()
                
                class_stats[true_class]['total'] += 1
                if rank_1_correct:
                    class_stats[true_class]['recall_at_1'] += 1
                
                if is_forget:
                    class_stats[true_class]['forget_total'] += 1
                    if rank_1_correct:
                        class_stats[true_class]['forget_recall_at_1'] += 1
                else:
                    class_stats[true_class]['retain_total'] += 1
                    if rank_1_correct:
                        class_stats[true_class]['retain_recall_at_1'] += 1
    
    for class_name in class_names:
        stats = class_stats[class_name]
        stats['recall_at_1'] = 100.0 * stats['recall_at_1'] / stats['total'] if stats['total'] > 0 else 0.0
        stats['forget_recall_at_1'] = 100.0 * stats['forget_recall_at_1'] / stats['forget_total'] if stats['forget_total'] > 0 else 0.0
        stats['retain_recall_at_1'] = 100.0 * stats['retain_recall_at_1'] / stats['retain_total'] if stats['retain_total'] > 0 else 0.0
    
    return class_stats

def comprehensive_evaluation_per_class(model, test_loader, device, class_names, forget_classes, model_name="Model"):
    prediction_stats = evaluate_zero_shot_per_class(
        model, test_loader, device, class_names
    )
    retrieval_stats = evaluate_retrieval_per_class(
        model, test_loader, device, class_names, k_list=[1]
    )
    
    print(f"\n{model_name} Per-Class Evaluation Results:")
    print("="*80)
    print(f"{'Class':<15} {'ZSRetrieval(%)':<20} {'ZSPrediction(%)':<20}")
    print(f"{'':<15} {'Df':<10} {'Dr':<10} {'Df':<10} {'Dr':<10}")
    print("-" * 80)
    
    forget_prediction_accuracies = []
    forget_retrieval_accuracies = []
    retain_prediction_accuracies = []
    retain_retrieval_accuracies = []
    
    for class_name in forget_classes:
        if class_name in prediction_stats and class_name in retrieval_stats:
            pred_stats = prediction_stats[class_name]
            ret_stats = retrieval_stats[class_name]
            
            df_pred = pred_stats['forget_accuracy']
            dr_pred = pred_stats['retain_accuracy']
            df_ret = ret_stats['forget_recall_at_1']
            dr_ret = ret_stats['retain_recall_at_1']
            
            forget_prediction_accuracies.append(df_pred)
            forget_retrieval_accuracies.append(df_ret)
            retain_prediction_accuracies.append(dr_pred)
            retain_retrieval_accuracies.append(dr_ret)
            
            print(f"{class_name:<15} {df_ret:<10.2f} {dr_ret:<10.2f} {df_pred:<10.2f} {dr_pred:<10.2f}")
    
    if forget_prediction_accuracies:
        mean_df_pred = np.mean(forget_prediction_accuracies)
        mean_dr_pred = np.mean(retain_prediction_accuracies)
        mean_df_ret = np.mean(forget_retrieval_accuracies)
        mean_dr_ret = np.mean(retain_retrieval_accuracies)
        
        var_df_pred = np.var(forget_prediction_accuracies)
        var_dr_pred = np.var(retain_prediction_accuracies)
        var_df_ret = np.var(forget_retrieval_accuracies)
        var_dr_ret = np.var(retain_retrieval_accuracies)
        
        print("-" * 80)
        print(f"{'Mean':<15} {mean_df_ret:<10.2f} {mean_dr_ret:<10.2f} {mean_df_pred:<10.2f} {mean_dr_pred:<10.2f}")
        print(f"{'Variance':<15} {var_df_ret:<10.2f} {var_dr_ret:<10.2f} {var_df_pred:<10.2f} {var_dr_pred:<10.2f}")
    
    return {
        'prediction': prediction_stats,
        'retrieval': retrieval_stats
    }

def evaluate_zero_shot(model, data_loader, device, class_names):
    model.eval()
    correct = 0
    total = 0
    forget_correct = 0
    forget_total = 0
    retain_correct = 0
    retain_total = 0
   
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
            
            image_features, text_features = ensure_same_dtype(image_features, text_features)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)
            
            labels = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            batch_correct = (predictions == labels)
            correct += batch_correct.sum().item()
            total += len(labels)
            
            if flags_tensor.sum() > 0:
                forget_correct += batch_correct[flags_tensor].sum().item()
                forget_total += flags_tensor.sum().item()
            
            retain_mask = ~flags_tensor
            if retain_mask.sum() > 0:
                retain_correct += batch_correct[retain_mask].sum().item()
                retain_total += retain_mask.sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    forget_accuracy = 100.0 * forget_correct / forget_total if forget_total > 0 else 0.0
    retain_accuracy = 100.0 * retain_correct / retain_total if retain_total > 0 else 0.0
    
    return accuracy, forget_accuracy, retain_accuracy

def evaluate_retrieval(model, data_loader, device, class_names, k_list=[1, 5, 10]):
    model.eval()
   
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
    
    results = {
        'overall': {f'R@{k}': 0.0 for k in k_list},
        'forget': {f'R@{k}': 0.0 for k in k_list},
        'retain': {f'R@{k}': 0.0 for k in k_list}
    }
    counts = {'overall': 0, 'forget': 0, 'retain': 0}
    
    with torch.no_grad():
        for batch in data_loader:
            images, texts, flags, labels = batch
            
            image_tensor = torch.stack([model.preprocess(img) for img in images]).to(device)
            flags_tensor = torch.tensor(flags).to(device)
            
            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            image_features, text_features = ensure_same_dtype(image_features, text_features)
            similarity = image_features @ text_features.T 
            labels = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            for i in range(len(images)):
                sim_scores = similarity[i]
                sorted_indices = torch.argsort(sim_scores, descending=True)
                
                true_label = labels[i]
                rank = (sorted_indices == true_label).nonzero(as_tuple=True)[0].item() + 1
                
                counts['overall'] += 1
                for k in k_list:
                    if rank <= k:
                        results['overall'][f'R@{k}'] += 1
                
                if flags_tensor[i]:
                    set_type = 'forget'
                else:
                    set_type = 'retain'
                
                counts[set_type] += 1
                for k in k_list:
                    if rank <= k:
                        results[set_type][f'R@{k}'] += 1
    
    for set_type in ['overall', 'forget', 'retain']:
        for k in k_list:
            if counts[set_type] > 0:
                results[set_type][f'R@{k}'] = 100.0 * results[set_type][f'R@{k}'] / counts[set_type]
            else:
                results[set_type][f'R@{k}'] = 0.0
    
    return results

def comprehensive_evaluation_overall(model, test_loader, device, class_names, model_name="Model"):
    print(f"\n{model_name} Overall Evaluation Results:")
    print("="*50)
    
    test_accuracy, test_forget_accuracy, test_retain_accuracy = evaluate_zero_shot(
        model, test_loader, device, class_names
    )
    
    retrieval_results = evaluate_retrieval(
        model, test_loader, device, class_names, k_list=[1, 5, 10]
    )
    
    print(f"Zero-shot Classification Accuracy:")
    print(f"  Overall Accuracy: {test_accuracy:.2f}%")
    print(f"  Forget Set Accuracy (Df↓): {test_forget_accuracy:.2f}%")
    print(f"  Retain Set Accuracy (Dr↑): {test_retain_accuracy:.2f}%")
    
    print(f"\nZero-shot Retrieval Performance:")
    print(f"  Overall - R@1: {retrieval_results['overall']['R@1']:.2f}%, R@5: {retrieval_results['overall']['R@5']:.2f}%, R@10: {retrieval_results['overall']['R@10']:.2f}%")
    print(f"  Forget Set - R@1: {retrieval_results['forget']['R@1']:.2f}%, R@5: {retrieval_results['forget']['R@5']:.2f}%, R@10: {retrieval_results['forget']['R@10']:.2f}%")
    print(f"  Retain Set - R@1: {retrieval_results['retain']['R@1']:.2f}%, R@5: {retrieval_results['retain']['R@5']:.2f}%, R@10: {retrieval_results['retain']['R@10']:.2f}%")
    
    return {
        'prediction': {
            'overall': test_accuracy,
            'forget': test_forget_accuracy,
            'retain': test_retain_accuracy
        },
        'retrieval': retrieval_results
    }

def analyze_cosine_similarity_distribution_with_flags(model, data_loader, device, class_names, save_path=None):
    model.eval()
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
    
    similarity_stats = {
        'forget_set': {
            'true_class_similarities': [],
            'other_class_similarities': [],
            'max_similarities': [],
            'min_similarities': [],
            'class_wise_stats': {cls: [] for cls in class_names},
            'similarity_matrix': []
        },
        'retain_set': {
            'true_class_similarities': [],
            'other_class_similarities': [],
            'max_similarities': [],
            'min_similarities': [],
            'class_wise_stats': {cls: [] for cls in class_names},
            'similarity_matrix': []
        }
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images, texts, flags, labels = batch
            image_tensor = torch.stack([model.preprocess(img) for img in images]).to(device)
            flags_tensor = torch.tensor(flags).to(device)
            
            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            image_features_curr, text_features_curr = ensure_same_dtype(image_features, text_features)
            cosine_similarities = image_features_curr @ text_features_curr.T
            
            true_label_indices = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            for i in range(len(images)):
                true_class = texts[i]
                true_class_idx = true_label_indices[i].item()
                is_forget = flags_tensor[i].item()
                
                sim_vector = cosine_similarities[i].cpu().numpy()
                
                set_key = 'forget_set' if is_forget else 'retain_set'
                stats = similarity_stats[set_key]
                
                true_sim = float(sim_vector[true_class_idx])
                stats['true_class_similarities'].append(true_sim)
                stats['class_wise_stats'][true_class].append(true_sim)
                
                other_sims = np.delete(sim_vector, true_class_idx)
                stats['other_class_similarities'].extend([float(sim) for sim in other_sims])
                
                stats['max_similarities'].append(np.max(sim_vector))
                stats['min_similarities'].append(np.min(sim_vector))
                
                if batch_idx % 10 == 0 and i == 0:
                    stats['similarity_matrix'].append(sim_vector)
            
            if batch_idx % 20 == 0:
                print(f"Processed {batch_idx * len(images)} samples...")
    
    if save_path:
        save_similarity_stats_simple(similarity_stats, save_path)
    
    return similarity_stats

def save_similarity_stats_simple(similarity_stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    json_path = save_path if save_path.endswith('.json') else save_path + '.json'
    
    save_data = {
        'forget_set': {
            'true_class_similarities': similarity_stats['forget_set']['true_class_similarities'],
            'other_class_similarities': similarity_stats['forget_set']['other_class_similarities']
        },
        'retain_set': {
            'true_class_similarities': similarity_stats['retain_set']['true_class_similarities'],
            'other_class_similarities': similarity_stats['retain_set']['other_class_similarities']
        }
    }
   
    print("Debug Info:")
    print(f"forget_set keys: {list(similarity_stats['forget_set'].keys())}")
    print(f"retain_set keys: {list(similarity_stats['retain_set'].keys())}")
    
    print(f"forget_set true_class_similarities length: {len(similarity_stats['forget_set']['true_class_similarities'])}")
    print(f"retain_set true_class_similarities length: {len(similarity_stats['retain_set']['true_class_similarities'])}")
   
    if len(similarity_stats['retain_set']['true_class_similarities']) == 0:
        print("Warning: retain_set data is empty!")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Similarity data saved to: {json_path}")
    
    npz_path = json_path.replace('.json', '.npz')
    np.savez(
        npz_path,
        forget_true=np.array(similarity_stats['forget_set']['true_class_similarities']),
        forget_other=np.array(similarity_stats['forget_set']['other_class_similarities']),
        retain_true=np.array(similarity_stats['retain_set']['true_class_similarities']),
        retain_other=np.array(similarity_stats['retain_set']['other_class_similarities'])
    )
    print(f"Similarity data saved as numpy format: {npz_path}")

def print_similarity_analysis_with_flags(similarity_stats, class_names):
    print(f"\nCosine Similarity Analysis for Forget and Retain Sets")
    print("="*100)
    
    for set_name in ['forget_set', 'retain_set']:
        stats = similarity_stats[set_name]
        set_display_name = "Forget Set" if set_name == 'forget_set' else "Retain Set"
        
        true_sims = np.array(stats['true_class_similarities'])
        other_sims = np.array(stats['other_class_similarities'])
        max_sims = np.array(stats['max_similarities'])
        min_sims = np.array(stats['min_similarities'])
        
        print(f"\n{set_display_name} Statistics:")
        print(f"  Total samples: {len(true_sims)}")
        print(f"  True class mean similarity: {np.mean(true_sims):.4f} ± {np.std(true_sims):.4f}")
        print(f"  Other classes mean similarity: {np.mean(other_sims):.4f} ± {np.std(other_sims):.4f}")
        print(f"  Max similarity mean: {np.mean(max_sims):.4f} ± {np.std(max_sims):.4f}")
        print(f"  Min similarity mean: {np.mean(min_sims):.4f} ± {np.std(min_sims):.4f}")
        
        discrimination_score = np.mean(true_sims) - np.mean(other_sims)
        discrimination_ratio = np.mean(true_sims) / np.mean(other_sims) if np.mean(other_sims) > 0 else float('inf')
        
        print(f"  Discrimination Analysis:")
        print(f"    True vs Other similarity difference: {discrimination_score:.4f}")
        print(f"    True vs Other similarity ratio: {discrimination_ratio:.2f}")
        
        print(f"  Per-Class Statistics (Top 5 classes):")
        print(f"    {'Class':<20} {'Samples':<8} {'Mean Sim':<12} {'Std Dev':<10}")
        print("    " + "-" * 60)
        
        class_stats = []
        for class_name in class_names[:5]:  
            sims = stats['class_wise_stats'][class_name]
            if sims:
                class_stats.append((class_name, len(sims), np.mean(sims), np.std(sims)))
                print(f"    {class_name:<20} {len(sims):<8} {np.mean(sims):<12.4f} {np.std(sims):<10.4f}")
        
        if class_stats:
            highest_class = max(class_stats, key=lambda x: x[2])
            lowest_class = min(class_stats, key=lambda x: x[2])
            print(f"    Highest similarity class: {highest_class[0]} ({highest_class[2]:.4f})")
            print(f"    Lowest similarity class: {lowest_class[0]} ({lowest_class[2]:.4f})")

def evaluate_zero_shot_accuracy_with_flags(model, data_loader, device, class_names):
    model.eval()
    
    results = {
        'overall': {'correct': 0, 'total': 0},
        'forget_set': {'correct': 0, 'total': 0},
        'retain_set': {'correct': 0, 'total': 0},
        'forget_prediction_distribution': {cls: 0 for cls in class_names}
    }
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images, texts, flags, labels = batch
            image_tensor = torch.stack([model.preprocess(img) for img in images]).to(device)
            flags_tensor = torch.tensor(flags).to(device)
            
            image_features = model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            image_features_curr, text_features_curr = ensure_same_dtype(image_features, text_features)
            curr_similarity = image_features_curr @ text_features_curr.T
            
            labels_idx = torch.tensor([class_names.index(text) for text in texts]).to(device)
            
            for i in range(len(images)):
                is_forget = flags_tensor[i].item()
                set_key = 'forget_set' if is_forget else 'retain_set'
                
                results['overall']['total'] += 1
                results[set_key]['total'] += 1
                
                pred_idx = curr_similarity[i].argmax().item()
                true_idx = labels_idx[i].item()
                
                if pred_idx == true_idx:
                    results['overall']['correct'] += 1
                    results[set_key]['correct'] += 1
                
                if is_forget:
                    pred_class = class_names[pred_idx]
                    results['forget_prediction_distribution'][pred_class] += 1
            
            if batch_idx % 20 == 0:
                print(f"Evaluated {batch_idx * len(images)} samples...")
    
    for key in ['overall', 'forget_set', 'retain_set']:
        total = results[key]['total']
        correct = results[key]['correct']
        results[key]['accuracy'] = 100.0 * correct / total if total > 0 else 0.0
    
    return results

def comprehensive_evaluation_with_flags(model, test_loader, device, class_names, model_name="CLIP Model", save_path=None):
    print(f"\n{model_name} Performance Evaluation with Flag-based Distinction:")
    print("="*80)

    similarity_stats = analyze_cosine_similarity_distribution_with_flags(
        model, test_loader, device, class_names, save_path=save_path
    )
    
    print_similarity_analysis_with_flags(similarity_stats, class_names)
    
    accuracy_results = evaluate_zero_shot_accuracy_with_flags(
        model, test_loader, device, class_names
    )
    
    print(f"\nGenerating similarity distribution plots...")
    
    print(f"\nPerformance Summary:")
    print(f"  Overall Zero-shot Classification Accuracy: {accuracy_results['overall']['accuracy']:.2f}%")
    print(f"  Forget Set Zero-shot Classification Accuracy: {accuracy_results['forget_set']['accuracy']:.2f}%")
    print(f"  Retain Set Zero-shot Classification Accuracy: {accuracy_results['retain_set']['accuracy']:.2f}%")
    
    forget_stats = similarity_stats['forget_set']
    retain_stats = similarity_stats['retain_set']
    
    forget_true_sims = np.array(forget_stats['true_class_similarities'])
    retain_true_sims = np.array(retain_stats['true_class_similarities'])
    
    print(f"  Forget Set Mean True Class Similarity: {np.mean(forget_true_sims):.4f}")
    print(f"  Retain Set Mean True Class Similarity: {np.mean(retain_true_sims):.4f}")
    
    print(f"\nForget Set Prediction Distribution:")
    print(f"{'Predicted Class':<20} {'Count':<10} {'Percentage(%)':<10}")
    print("-" * 50)
    
    total_forget_samples = accuracy_results['forget_set']['total']
    forget_dist = accuracy_results['forget_prediction_distribution']
    
    valid_predictions = {cls: count for cls, count in forget_dist.items() if count > 0}
    
    if valid_predictions:
        for class_name, count in sorted(valid_predictions.items(), key=lambda x: x[1], reverse=True):
            percentage = 100.0 * count / total_forget_samples if total_forget_samples > 0 else 0.0
            print(f"{class_name:<20} {count:<10} {percentage:.2f}%")
    else:
        print("  No forget set prediction data")
    
    return {
        'similarity_stats': similarity_stats,
        'accuracy_results': accuracy_results,
        'forget_prediction_distribution': forget_dist
    }

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

def main():
    config = {
        "dataset": {
            "name": "CIFAR100",
            "forget_classes": ['unlearned concept'], 
            "data_root": "your data root"
        },
        "training": {
            "batch_size": 128,
        },
        "dataloader": {
            "shuffle": False,
            "num_workers": 8
        },
        "misc": {
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    
    torch.manual_seed(config["misc"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["misc"]["seed"])
    device = torch.device(config["misc"]["device"])
    
    print(f"Using device: {device}")
    
    forget_classes = config["dataset"]["forget_classes"]
    def forget_rule(sample):
        return sample["text"] in forget_classes
    
    test_dataset = UniversalDataset(
        "CIFAR100",
        root=config["dataset"]["data_root"],
        train=False,
        download=True,
        label_names={i: name for i, name in enumerate(CIFAR100_LABEL_NAMES)}
    )
    test_dataset.set_forget_rule(forget_rule)
    
    test_loader = UniversalDataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["dataloader"]["shuffle"],
        num_workers=config["dataloader"]["num_workers"]
    )
    
    forget_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i]["flag"])
    retain_count = len(test_dataset) - forget_count
    print(f"Test set statistics: Total {len(test_dataset)} samples")
    print(f"Forget set: {forget_count} samples ({forget_classes})")
    print(f"Retain set: {retain_count} samples")
    
    model = CLIPModel().to(device)
    
    checkpoint_path = " your checkpoint path "
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    print(f"\nLoading specified model: {checkpoint_path}")
    try:
        model = load_model_from_checkpoint(model, checkpoint_path, device)
        print("Successfully loaded specified model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Testing with randomly initialized model")
    
    print("\n" + "="*80)
    print("Part 1: Traditional Evaluation Metrics")
    print("="*80)
    
    trained_per_class_results = comprehensive_evaluation_per_class(
        model, test_loader, device, CIFAR100_LABEL_NAMES, forget_classes, "Trained CLIPErase Model"
    )
    
    print("\n" + "="*80)
    trained_results = comprehensive_evaluation_overall(
        model, test_loader, device, CIFAR100_LABEL_NAMES, "Trained CLIPErase Model"
    )
    
    print("\n" + "="*80)
    print("Part 2: Cosine Similarity Distribution Analysis")
    print("="*80)

    save_dir = " your cosine similarity save path "
    os.makedirs(save_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    similarity_results = comprehensive_evaluation_with_flags(
        model, test_loader, device, CIFAR100_LABEL_NAMES, 
        model_name="CLIPErase Model", save_path=os.path.join(save_dir, f"similarity_stats_{timestamp}.json")
    )
    
    print("\n" + "="*80)
    print("Part 3: Comprehensive Performance Comparison")
    print("="*80)
    
    print("\nZero-shot Classification Accuracy Comparison:")
    print(f"{'Metric':<15} {'Trained Model':<10}")
    print("-" * 50)
   
    pred_trained = trained_results['prediction']
    
    metrics = [
        ('Overall Accuracy', pred_trained['overall']),
        ('Forget Set Accuracy', pred_trained['forget']),
        ('Retain Set Accuracy', pred_trained['retain'])
    ]
    
    for metric_name, trained_value in metrics:
        print(f"{metric_name:<15} {trained_value:<10.2f}%")
    
    print("\nZero-shot Retrieval Performance Comparison (R@1):")
    print(f"{'Metric':<15} {'Trained Model':<10}")
    print("-" * 50)
    
    ret_trained = trained_results['retrieval']
    
    retrieval_metrics = [
        ('Overall R@1', ret_trained['overall']['R@1']),
        ('Forget Set R@1', ret_trained['forget']['R@1']),
        ('Retain Set R@1', ret_trained['retain']['R@1'])
    ]
    
    for metric_name, trained_value in retrieval_metrics:
        print(f"{metric_name:<15} {trained_value:<10.2f}%")
    
    print("\n" + "="*80)
    print("Part 4: Results Saving")
    print("="*80)
    
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {
        'model_info': {
            'checkpoint_path': checkpoint_path,
            'forget_classes': forget_classes,
            'device': str(device)
        },
        'traditional_evaluation': {
            'prediction': {
                'overall': float(pred_trained['overall']),
                'forget': float(pred_trained['forget']),
                'retain': float(pred_trained['retain'])
            },
            'retrieval': {
                'overall': {k: float(v) for k, v in ret_trained['overall'].items()},
                'forget': {k: float(v) for k, v in ret_trained['forget'].items()},
                'retain': {k: float(v) for k, v in ret_trained['retain'].items()}
            }
        },
        'similarity_analysis': {
            'forget_set': {
                'mean_true_similarity': float(np.mean(similarity_results['similarity_stats']['forget_set']['true_class_similarities'])),
                'std_true_similarity': float(np.std(similarity_results['similarity_stats']['forget_set']['true_class_similarities'])),
                'mean_other_similarity': float(np.mean(similarity_results['similarity_stats']['forget_set']['other_class_similarities']))
            },
            'retain_set': {
                'mean_true_similarity': float(np.mean(similarity_results['similarity_stats']['retain_set']['true_class_similarities'])),
                'std_true_similarity': float(np.std(similarity_results['similarity_stats']['retain_set']['true_class_similarities'])),
                'mean_other_similarity': float(np.mean(similarity_results['similarity_stats']['retain_set']['other_class_similarities']))
            }
        }
    }
    
    with open(os.path.join(output_dir, "comprehensive_evaluation_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Evaluation results saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("Part 5: Final Evaluation Summary")
    print("="*80)
    
    print(f"Model path: {checkpoint_path}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Forget classes: {forget_classes}")
    print(f"Forget set samples: {forget_count}")
    print(f"Retain set samples: {retain_count}")
    print(f"\nKey Metrics:")
    print(f"  1. Forget Set Classification Accuracy: {pred_trained['forget']:.2f}% (Target: Lower)")
    print(f"  2. Retain Set Classification Accuracy: {pred_trained['retain']:.2f}% (Target: Maintain)")
    print(f"  3. Forget Set R@1 Retrieval Accuracy: {ret_trained['forget']['R@1']:.2f}%")
    print(f"  4. Retain Set R@1 Retrieval Accuracy: {ret_trained['retain']['R@1']:.2f}%")
    print(f"  5. Forget Set Mean True Class Similarity: {results_summary['similarity_analysis']['forget_set']['mean_true_similarity']:.4f}")
    print(f"  6. Retain Set Mean True Class Similarity: {results_summary['similarity_analysis']['retain_set']['mean_true_similarity']:.4f}")

if __name__ == "__main__":
    main()