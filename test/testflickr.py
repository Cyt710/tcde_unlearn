
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import clip
from typing import List, Dict, Tuple
import os
import pandas as pd
from PIL import Image
import logging
import gc
from testcifar import CLIPModel

logging.basicConfig(level=logging.WARNING)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class FlickrTester:
    def __init__(self, model, device, test_forget_dir, test_retain_dir, batch_size=16):
        self.model = model
        self.device = device
        self.test_forget_dir = test_forget_dir
        self.test_retain_dir = test_retain_dir
        self.batch_size = batch_size
        
        self._check_gpu_status()
        
        self.test_data, self.image_to_texts = self.load_test_data()
        
        self.forget_data = [item for item in self.test_data if item['is_forget']]
        self.retain_data = [item for item in self.test_data if not item['is_forget']]
       
        self.forget_text_queries = self._construct_forget_queries()
      
        self.all_image_features, self.all_image_info = self._extract_all_image_features()
           
        self.text_pool_data = self._build_text_pool()
       
        self.all_text_features, self.all_text_info = self._extract_all_text_features()
        
        self._clear_memory()
    
    def _construct_forget_queries(self):
       
        mountain_queries = [
        
        "a picture of mountains",
        "a photo of mountain peaks", 
        "a view of hills and mountains",
        "mountain landscape",
        "snowy mountains",
        "rocky mountains",
        "mountain scenery",
        "mountains and sky",
        "mountain range",
        "alpine mountains",
        "high mountains",
        "green mountains",
        "mountain view",
        "mountainous terrain",
        "mountain slope",
        
        "a majestic mountain with snow capped peaks",
        "rugged mountain terrain with cliffs",
        "volcanic mountain with smoke",
        "mountain pass between two peaks",
        "mountain ridge at sunset",
        "steep mountain face with rock formations",
        "rolling mountains in the distance",
        "mountain valley with river",
        "mountain summit covered in clouds",
        "mountain foothills with trees",
        "mountain plateau with grass",
        "mountain canyon with steep walls",
        "mountain glacier and ice fields",
        "mountain lake surrounded by peaks",
        "mountain meadow with wildflowers",
        "mountain desert landscape",
        "mountain tundra with sparse vegetation",
        "mountain forest on the slopes",
        "mountain waterfall cascading down rocks",
        "mountain hot springs",
        
        "sunny day in the mountains",
        "cloudy mountains with mist",
        "rainy mountain landscape",
        "foggy mountains in the morning",
        "snowstorm in the mountains",
        "mountains during golden hour",
        "mountain sunset with orange sky",
        "mountain sunrise over peaks",
        "moonlit mountain landscape",
        "mountains in autumn with colorful trees",
        "spring mountains with melting snow",
        "summer mountains with green vegetation",
        "winter mountains completely covered in snow",
        "mountains after rain with clear air",
        "mountains with rainbow after storm",
        
        "the Himalayan mountains",
        "the Rocky Mountains",
        "the Alps mountain range",
        "the Andes mountains",
        "the Appalachian Mountains",
        "the Sierra Nevada mountains",
        "the Ural Mountains",
        "the Caucasus Mountains",
        "the Carpathian Mountains",
        "the Pyrenees mountains",
        "the Atlas Mountains",
        "the Tian Shan mountains",
        "the Kunlun Mountains",
        "the Drakensberg mountains",
        "the Scottish Highlands",
        
        "aerial view of mountains",
        "close up of mountain rocks",
        "panoramic mountain vista",
        "mountains reflected in a lake",
        "mountains from a valley below",
        "mountains seen through trees",
        "mountains with a small cabin",
        "mountains with hiking trail",
        "mountains with cable car",
        "mountains with winding road",
        "mountains with ancient ruins",
        "mountains with monastery on cliff",
        "mountains with village at base",
        "mountains with wildlife",
        "mountains with climbers",
        
        "painting of mountains",
        "black and white mountain photography",
        "abstract mountain shapes",
        "impressionist mountain scene",
        "surreal mountain landscape",
        "minimalist mountain composition",
        "mountain silhouette against sky",
        "textured mountain surface",
        "geometric mountain patterns",
        "dreamlike mountain imagery",
       
        "peaceful mountain retreat",
        "majestic and grand mountains",
        "lonely mountain in wilderness",
        "forbidding mountain landscape",
        "serene mountain scenery",
        "powerful mountain presence",
        "ancient and timeless mountains",
        "mysterious mountain fog",
        "tranquil mountain morning",
        "dramatic mountain storm",
        
        "mountain climbing expedition",
        "skiing down mountain slopes",
        "hiking in mountain trails",
        "camping in mountain wilderness",
        "mountain biking on dirt trails",
        "paragliding over mountains",
        "mountain railway journey",
        "fishing in mountain stream",
        "photographing mountain wildlife",
        "mountain rescue operation",
        
        "extreme high altitude mountains",
        "frozen mountain landscape",
        "barren mountain desert",
        "tropical mountain rainforest",
        "coastal mountains meeting ocean",
       
        "sacred mountain in local culture",
        "legendary mountain from myths",
        "historical mountain fortress",
        "mountain pilgrimage site",
        "mountain with spiritual significance"
        ]
    
        return mountain_queries
    
    def _check_gpu_status(self):
        if torch.cuda.is_available():
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Initial Memory Usage: {allocated:.2f} GB / {reserved:.2f} GB")
    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def load_test_data(self):
       
        test_data = []
        image_to_texts = {}  
        
        forget_csv = os.path.join(self.test_forget_dir, 'captions.csv')
        if os.path.exists(forget_csv):
            forget_df = pd.read_csv(forget_csv)
            for _, row in forget_df.iterrows():
                img_name = row['image_name'].strip()
                img_path = os.path.join(self.test_forget_dir, img_name)
                
                if os.path.exists(img_path):
                    text = str(row['comment']).strip()
                    
                    test_data.append({
                        'image_path': img_path,
                        'image_name': img_name,
                        'text': text,
                        'is_forget': True
                    })
                    
                    if img_name not in image_to_texts:
                        image_to_texts[img_name] = []
                    image_to_texts[img_name].append(text)
       
        retain_csv = os.path.join(self.test_retain_dir, 'captions.csv')
        if os.path.exists(retain_csv):
            retain_df = pd.read_csv(retain_csv)
            for _, row in retain_df.iterrows():
                img_name = row['image_name'].strip()
                img_path = os.path.join(self.test_retain_dir, img_name)
                
                if os.path.exists(img_path):
                    text = str(row['comment']).strip()
                   
                    test_data.append({
                        'image_path': img_path,
                        'image_name': img_name,
                        'text': text,
                        'is_forget': False
                    })
                    
                    if img_name not in image_to_texts:
                        image_to_texts[img_name] = []
                    image_to_texts[img_name].append(text)
        
        for img_name, texts in list(image_to_texts.items())[:5]:  
            print(f"Image '{img_name}' has {len(texts)} captions")
        
        return test_data, image_to_texts
    
    def _extract_all_image_features(self):
        
        image_features_list = []
        image_info_list = []
        
        for i in tqdm(range(0, len(self.test_data), self.batch_size), desc="Extracting image features"):
            batch_items = self.test_data[i:i+self.batch_size]
            batch_features = []
            
            for item in batch_items:
                try:
                    image = Image.open(item['image_path']).convert('RGB')
                    image_tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.model.encode_image(image_tensor)
                        image_features = F.normalize(image_features, p=2, dim=-1)
                    
                    batch_features.append(image_features.cpu())
                    image_info_list.append(item)
                    
                    del image_tensor
                    
                except Exception as e:
                    logging.warning(f"Unable to load image {item['image_path']}: {e}")
                    dummy_feature = torch.zeros(1, 512)
                    batch_features.append(dummy_feature)
                    image_info_list.append(item)
            
            if batch_features:
                batch_tensor = torch.cat(batch_features, dim=0)
                image_features_list.append(batch_tensor)
            
            if i % (self.batch_size * 10) == 0:
                self._clear_memory()
        
        if image_features_list:
            all_features = torch.cat(image_features_list, dim=0)
        else:
            all_features = torch.zeros(len(self.test_data), 512)
        
        return all_features.to(self.device), image_info_list
    
    def _build_text_pool(self):
        text_pool = []
        
        for item in self.test_data:
            text_pool.append({
                'text': item['text'],
                'image_name': item['image_name'],
                'is_forget': item['is_forget'],
                'is_specific_query': False
            })
        
        for query in self.forget_text_queries:
            text_pool.append({
                'text': query,
                'image_name': 'mountain_query',
                'is_forget': True,
                'is_specific_query': True
            })
        
        print(f"Text pool constructed: {len(text_pool)} texts")
        print(f"  - Original texts: {len(self.test_data)}")
        print(f"  - Specific query texts: {len(self.forget_text_queries)}")
        
        return text_pool
    
    def _extract_all_text_features(self):
        print("Extracting text features...")
        
        all_texts = []
        text_info_list = []
        
        for text_item in self.text_pool_data:
            all_texts.append(text_item['text'])
            text_info_list.append(text_item)
        
        batch_size = min(64, len(all_texts))
        all_features_list = []
        
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding texts"):
            batch_texts = all_texts[i:i+batch_size]
            
            try:
                batch_tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
                
                with torch.no_grad():
                    batch_features = self.model.encode_text(batch_tokens)
                    batch_features = F.normalize(batch_features, p=2, dim=-1)
                
                all_features_list.append(batch_features.cpu())
                
                del batch_tokens
                
            except Exception as e:
                logging.warning(f"Failed to process text batch: {e}")
                dummy_features = torch.zeros(len(batch_texts), 512)
                all_features_list.append(dummy_features)
        
        all_features = torch.cat(all_features_list, dim=0).to(self.device)
        
        return all_features, text_info_list
    
    def attack_success_rate_evaluation(self, top_k=10):
        """
        Attack success rate evaluation:
        Search for forget set samples by finding minimum cosine similarity (maximum negative correlation)
        Calculate the proportion of forget set that can be discovered by attacker
        """
        print("\n" + "="*60)
        print("Task 3: Attack Success Rate Evaluation (Adversarial Search)")
        print("="*60)
        
        self.model.eval()
        
        with torch.no_grad():
            print(f"\n1. Image Retrieval Attack Success Rate")
            print("   Attack method: Use mountain-related text queries to find images with minimum cosine similarity")
            
            forget_image_attack_success = {k: 0 for k in [1, 5, 10, top_k]}
            
            forget_image_indices = []
            for idx, item in enumerate(self.all_image_info):
                if item['is_forget']:
                    forget_image_indices.append(idx)
            
            print(f"   Total forget set images: {len(forget_image_indices)}")
            
            for query_text in tqdm(self.forget_text_queries, desc="IR attack testing"):
                try:
                    text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
                    query_features = self.model.encode_text(text_tokens)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(self.all_image_features, query_features.transpose(0, 1)).squeeze()
                    
                    min_k = min(top_k, len(similarities))
                    worst_indices = torch.topk(similarities, k=min_k, largest=False).indices.cpu().numpy()
                    
                    for k in [1, 5, 10, top_k]:
                        if k <= len(worst_indices):
                            worst_k = worst_indices[:k]
                            found_forget_images = 0
                            
                            for idx in worst_k:
                                if idx in forget_image_indices:
                                    found_forget_images += 1
                            
                            if found_forget_images > 0:
                                forget_image_attack_success[k] += 1
                    
                    del text_tokens, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process query: {query_text}... {e}")
                    continue
            
            print(f"\n2. Text Retrieval Attack Success Rate")
            print("   Attack method: Use forget set images to find texts with minimum cosine similarity")
            
            forget_text_attack_success = {k: 0 for k in [1, 5, 10, top_k]}
            
            forget_text_indices = []
            for idx, text_item in enumerate(self.all_text_info):
                if text_item['is_forget'] and not text_item['is_specific_query']:
                    forget_text_indices.append(idx)
            
            print(f"   Total forget set texts: {len(forget_text_indices)}")
            
            attack_test_images = self.forget_data[:min(50, len(self.forget_data))]
            
            for query_item in tqdm(attack_test_images, desc="TR attack testing"):
                try:
                    image = Image.open(query_item['image_path']).convert('RGB')
                    image_tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
                    query_features = self.model.encode_image(image_tensor)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(query_features, self.all_text_features.transpose(0, 1)).squeeze()
                    
                    min_k = min(top_k, len(similarities))
                    worst_indices = torch.topk(similarities, k=min_k, largest=False).indices.cpu().numpy()
                    
                    for k in [1, 5, 10, top_k]:
                        if k <= len(worst_indices):
                            worst_k = worst_indices[:k]
                            found_forget_texts = 0
                            
                            for idx in worst_k:
                                if idx in forget_text_indices:
                                    found_forget_texts += 1
                            
                            if found_forget_texts > 0:
                                forget_text_attack_success[k] += 1
                    
                    del image_tensor, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process image: {query_item['image_path']} {e}")
                    continue
            
            image_attack_rate = {k: forget_image_attack_success[k] / len(self.forget_text_queries) 
                                 for k in [1, 5, 10, top_k]}
            text_attack_rate = {k: forget_text_attack_success[k] / len(attack_test_images) 
                                for k in [1, 5, 10, top_k]}
            
            print(f"\nAttack Success Rate Results:")
            print(f"\nImage Retrieval Attack Success Rate (Negative Correlation Search):")
            for k in [1, 5, 10, top_k]:
                print(f"  Top-{k} smallest similarities containing at least one forget image: {image_attack_rate[k]:.3f} "
                      f"({forget_image_attack_success[k]}/{len(self.forget_text_queries)})")
            
            print(f"\nText Retrieval Attack Success Rate (Negative Correlation Search):")
            for k in [1, 5, 10, top_k]:
                print(f"  Top-{k} smallest similarities containing at least one forget text: {text_attack_rate[k]:.3f} "
                      f"({forget_text_attack_success[k]}/{len(attack_test_images)})")
            
            print(f"\nComparative Analysis:")
            print(f"  Image Retrieval:")
            print(f"    Normal Search (Max Similarity) Target: Find forget set images")
            print(f"    Attack Search (Min Similarity) Target: Find forget set images")
            print(f"  Text Retrieval:")
            print(f"    Normal Search (Max Similarity) Target: Find correct image captions")
            print(f"    Attack Search (Min Similarity) Target: Find forget set texts (regardless of matching)")
            
            return {
                'image_attack_success': {
                    'rates': image_attack_rate,
                    'counts': forget_image_attack_success,
                    'total_queries': len(self.forget_text_queries)
                },
                'text_attack_success': {
                    'rates': text_attack_rate,
                    'counts': forget_text_attack_success,
                    'total_queries': len(attack_test_images)
                },
                'forget_image_count': len(forget_image_indices),
                'forget_text_count': len(forget_text_indices)
            }
    
    def image_retrieval_evaluation(self, k_list=[1, 5, 10]):
        """
        Image Retrieval evaluation:
        - Forget set queries: Use specific mountain text queries
        - Retain set queries: Use texts from retain set
        """
        print("\n" + "="*60)
        print("Task 1: Image Retrieval (Find Images by Text)")
        print("="*60)
        
        self.model.eval()
        
        with torch.no_grad():
            image_features_all = self.all_image_features
            image_info_all = self.all_image_info
            
            max_k = max(k_list)
            
            print(f"\n1. Forget Set Queries ({len(self.forget_text_queries)} mountain-related texts)")
            forget_recall = {k: 0 for k in k_list}
            
            for query_text in tqdm(self.forget_text_queries, desc="Forget queries"):
                try:
                    text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
                    query_features = self.model.encode_text(text_tokens)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(image_features_all, query_features.transpose(0, 1)).squeeze()
                    
                    topk_indices = torch.topk(similarities, k=max_k).indices.cpu().numpy()
                    
                    for k in k_list:
                        topk_subset = topk_indices[:k]
                        found_forget_image = False
                        
                        for idx in topk_subset:
                            retrieved_item = image_info_all[idx]
                            if retrieved_item['is_forget']:
                                found_forget_image = True
                                break
                        
                        if found_forget_image:
                            forget_recall[k] += 1
                    
                    del text_tokens, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process query: {query_text}... {e}")
                    continue
            
            print(f"\n2. Retain Set Queries (Randomly select {min(5000, len(self.retain_data))} texts from retain set)")
            retain_query_items = []
            if len(self.retain_data) > 5000:
                import random
                retain_query_items = random.sample(self.retain_data, 5000)
            else:
                retain_query_items = self.retain_data
            
            retain_recall = {k: 0 for k in k_list}
            
            for query_item in tqdm(retain_query_items, desc="Retain queries"):
                query_text = query_item['text']
                query_image_name = query_item['image_name']
                
                try:
                    text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
                    query_features = self.model.encode_text(text_tokens)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(image_features_all, query_features.transpose(0, 1)).squeeze()
                    
                    topk_indices = torch.topk(similarities, k=max_k).indices.cpu().numpy()
                    
                    for k in k_list:
                        topk_subset = topk_indices[:k]
                        found_correct_image = False
                        
                        for idx in topk_subset:
                            retrieved_item = image_info_all[idx]
                            retrieved_image_name = retrieved_item['image_name']
                            
                            if retrieved_image_name in self.image_to_texts:
                                image_texts = self.image_to_texts[retrieved_image_name]
                                if query_text in image_texts:
                                    found_correct_image = True
                                    break
                        
                        if found_correct_image:
                            retain_recall[k] += 1
                    
                    del text_tokens, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process query: {query_text[:50]}... {e}")
                    continue
            
            avg_forget_recall = {k: forget_recall[k] / len(self.forget_text_queries) for k in k_list}
            avg_retain_recall = {k: retain_recall[k] / len(retain_query_items) for k in k_list}
            
            print(f"\nImage Retrieval Results:")
            for k in k_list:
                print(f"  k={k}:")
                print(f"    Forget query recall: {avg_forget_recall[k]:.3f} ({forget_recall[k]}/{len(self.forget_text_queries)})")
                print(f"    Retain query recall: {avg_retain_recall[k]:.3f} ({retain_recall[k]}/{len(retain_query_items)})")
            
            return {
                'forget_recall': avg_forget_recall,
                'retain_recall': avg_retain_recall,
                'forget_counts': forget_recall,
                'retain_counts': retain_recall
            }
    
    def text_retrieval_evaluation(self, k_list=[1, 5, 10]):
        """
        Text Retrieval evaluation:
        - Forget set images: Retrieved texts should be the correct captions for that image
        - Retain set images: Retrieved texts should be the correct captions for that image
        """
        print("\n" + "="*60)
        print("Task 2: Text Retrieval (Find Texts by Image)")
        print("="*60)
        
        self.model.eval()
        
        forget_queries = self.forget_data[:500] if len(self.forget_data) > 500 else self.forget_data
        retain_queries = self.retain_data[:5000] if len(self.retain_data) > 5000 else self.retain_data
        
        print(f"Image queries: forget set {len(forget_queries)}, retain set {len(retain_queries)}")
        
        with torch.no_grad():
            text_features_all = self.all_text_features
            text_info_all = self.all_text_info
            
            max_k = max(k_list)
            forget_recall = {k: 0 for k in k_list}
            retain_recall = {k: 0 for k in k_list}
            
            print(f"\n1. Forget Set Image Queries ({len(forget_queries)} images)")
            for query_item in tqdm(forget_queries, desc="Forget images"):
                try:
                    image = Image.open(query_item['image_path']).convert('RGB')
                    image_tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
                    query_features = self.model.encode_image(image_tensor)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(query_features, text_features_all.transpose(0, 1)).squeeze()
                    
                    topk_indices = torch.topk(similarities, k=max_k).indices.cpu().numpy()

                    query_image_name = query_item['image_name']
                    correct_texts = set()
                    if query_image_name in self.image_to_texts:
                        correct_texts = set(self.image_to_texts[query_image_name])
                    
                    for k in k_list:
                        topk_subset = topk_indices[:k]
                        found_correct_text = False

                        for idx in topk_subset:
                            retrieved_text_info = text_info_all[idx]
                            retrieved_text = retrieved_text_info['text']
                            
                            if retrieved_text in correct_texts:
                                found_correct_text = True
                                break
                        
                        if found_correct_text:
                            forget_recall[k] += 1
                    
                    del image_tensor, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process image: {query_item['image_path']} {e}")
                    continue
            
            print(f"\n2. Retain Set Image Queries ({len(retain_queries)} images)")
            for query_item in tqdm(retain_queries, desc="Retain images"):
                try:
                    image = Image.open(query_item['image_path']).convert('RGB')
                    image_tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
                    query_features = self.model.encode_image(image_tensor)
                    query_features = F.normalize(query_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(query_features, text_features_all.transpose(0, 1)).squeeze()
                    
                    topk_indices = torch.topk(similarities, k=max_k).indices.cpu().numpy()
                    
                    query_image_name = query_item['image_name']
                    correct_texts = set()
                    if query_image_name in self.image_to_texts:
                        correct_texts = set(self.image_to_texts[query_image_name])
                    
                    for k in k_list:
                        topk_subset = topk_indices[:k]
                        found_correct_text = False
                        
                        for idx in topk_subset:
                            retrieved_text_info = text_info_all[idx]
                            retrieved_text = retrieved_text_info['text']
                            
                            if retrieved_text in correct_texts:
                                found_correct_text = True
                                break
                        
                        if found_correct_text:
                            retain_recall[k] += 1
                    
                    del image_tensor, query_features, similarities
                    
                except Exception as e:
                    logging.warning(f"Unable to process image: {query_item['image_path']} {e}")
                    continue
            
            avg_forget_recall = {k: forget_recall[k] / len(forget_queries) for k in k_list}
            avg_retain_recall = {k: retain_recall[k] / len(retain_queries) for k in k_list}
            
            print(f"\nText Retrieval Results:")
            for k in k_list:
                print(f"  k={k}:")
                print(f"    Forget images retrieving correct texts: {avg_forget_recall[k]:.3f} ({forget_recall[k]}/{len(forget_queries)})")
                print(f"    Retain images retrieving correct texts: {avg_retain_recall[k]:.3f} ({retain_recall[k]}/{len(retain_queries)})")
            
            return {
                'forget_recall': avg_forget_recall,
                'retain_recall': avg_retain_recall,
                'forget_counts': forget_recall,
                'retain_counts': retain_recall
            }
    
    def run_comprehensive_evaluation(self, model_name="Model"):
        print(f"\n{'='*70}")
        print(f"{model_name} - Comprehensive Evaluation")
        print(f"{'='*70}")
        
        self._clear_memory()
        
        results = {}
        
        print("\n>>> Starting Image Retrieval Evaluation...")
        results['image_retrieval'] = self.image_retrieval_evaluation(k_list=[1, 5, 10])
        
        self._clear_memory()
        
        print("\n>>> Starting Text Retrieval Evaluation...")
        results['text_retrieval'] = self.text_retrieval_evaluation(k_list=[1, 5, 10])
        
        self._clear_memory()
        
        print("\n>>> Starting Attack Success Rate Evaluation...")
        results['attack_success'] = self.attack_success_rate_evaluation(top_k=20)
        
        self._clear_memory()
        
        self.print_summary(results, model_name)
        
        return results
    
    def print_summary(self, results, model_name):
        print(f"\n{'='*70}")
        print(f"{model_name} Evaluation Summary")
        print(f"{'='*70}")
        
        if 'image_retrieval' in results:
            ir = results['image_retrieval']
            print(f"\n1. Image Retrieval (Find Images by Text):")
            print(f"  Forget set queries: {len(self.forget_text_queries)} mountain-related texts")
            print(f"  Retain set queries: Randomly selected texts")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}:")
                print(f"    • Forget queries retrieving forget images: {ir['forget_recall'][k]:.3f}")
                print(f"    • Retain queries retrieving retain images: {ir['retain_recall'][k]:.3f}")
        
        if 'text_retrieval' in results:
            tr = results['text_retrieval']
            print(f"\n2. Text Retrieval (Find Texts by Image):")
            print(f"  Forget set images: Retrieving correct captions for that image is correct")
            print(f"  Retain set images: Retrieving correct captions for that image is correct")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}:")
                print(f"    • Forget images retrieving correct texts: {tr['forget_recall'][k]:.3f}")
                print(f"    • Retain images retrieving correct texts: {tr['retain_recall'][k]:.3f}")
        
        if 'attack_success' in results:
            attack = results['attack_success']
            print(f"\n3. Attack Success Rate (Adversarial Search):")
            print(f"  Total forget set images: {attack['forget_image_count']}")
            print(f"  Total forget set texts: {attack['forget_text_count']}")
            print(f"\n  Image Retrieval Attack Success Rate:")
            for k in [1, 5, 10, 20]:
                if k in attack['image_attack_success']['rates']:
                    rate = attack['image_attack_success']['rates'][k]
                    count = attack['image_attack_success']['counts'][k]
                    total = attack['image_attack_success']['total_queries']
                    print(f"    Top-{k} smallest similarities containing at least one forget image: {rate:.3f} ({count}/{total})")
            
            print(f"\n  Text Retrieval Attack Success Rate:")
            for k in [1, 5, 10, 20]:
                if k in attack['text_attack_success']['rates']:
                    rate = attack['text_attack_success']['rates'][k]
                    count = attack['text_attack_success']['counts'][k]
                    total = attack['text_attack_success']['total_queries']
                    print(f"    Top-{k} smallest similarities containing at least one forget text: {rate:.3f} ({count}/{total})")


def test_flickr_model(model_path, test_forget_dir, test_retain_dir):
    torch.cuda.empty_cache()
    gc.collect()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory initial state: {allocated:.2f} GB / {reserved:.2f} GB")
    
    
    model = CLIPModel().to(device)
    model.eval()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    tester = FlickrTester(model, device, test_forget_dir, test_retain_dir, batch_size=8)
    
    results = tester.run_comprehensive_evaluation("Trained Model")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

#Usage example
if __name__ == "__main__":
    model_path = "your path to the trained model checkpoint"
    test_forget_dir = "your path to the forget dataset"
    test_retain_dir = "your path to the retain dataset"
    
    results = test_flickr_model(model_path, test_forget_dir, test_retain_dir)
    
    if results:
        print("\n✅ Test completed!")
        print("\nFinal evaluation results:")
        print("-" * 50)
        
        if 'image_retrieval' in results:
            ir = results['image_retrieval']
            print(f"Image Retrieval R@1: forget={ir['forget_recall'][1]:.3f}, retain={ir['retain_recall'][1]:.3f}")
            print(f"Image Retrieval R@5: forget={ir['forget_recall'][5]:.3f}, retain={ir['retain_recall'][5]:.3f}")
        
        if 'text_retrieval' in results:
            tr = results['text_retrieval']
            print(f"Text Retrieval R@1: forget={tr['forget_recall'][1]:.3f}, retain={tr['retain_recall'][1]:.3f}")
            print(f"Text Retrieval R@5: forget={tr['forget_recall'][5]:.3f}, retain={tr['retain_recall'][5]:.3f}")
        
        if 'attack_success' in results:
            attack = results['attack_success']
            print(f"\nAttack Success Rate:")
            print(f"Image Retrieval Attack Success Rate@5: {attack['image_attack_success']['rates'][5]:.3f}")
            print(f"Text Retrieval Attack Success Rate@5: {attack['text_attack_success']['rates'][5]:.3f}")
            
    else:
        print("\n❌ Test failed!")
