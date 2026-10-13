import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import open_clip
from dataset import UniversalDataset, UniversalDataLoader
import os
import sys
from tqdm import tqdm
import random

# 导入CIFAR100类别名称
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

def load_openclip_model(checkpoint_path=None, model_name="ViT-B/32", device=None):
    """加载OpenCLIP模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    if checkpoint_path is None:
        print(f"Loading pretrained OpenCLIP model: {model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="laion2b_s32b_b82k"
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        return model, preprocess, tokenizer
    
    print(f"Loading checkpoint from {checkpoint_path}")
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
    
    print("✓ Model loaded successfully")
    return model, preprocess, tokenizer

def generate_gaussian_noise_embeddings(model, tokenizer, num_samples=100, device="cuda"):
    """生成高斯噪声图像的embedding（通过随机像素生成图像）"""
    print(f"\nGenerating {num_samples} Gaussian noise image embeddings...")
    
    noise_embeddings = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating noise embeddings"):
            # 生成随机噪声图像 (3, 224, 224) 符合CLIP输入尺寸
            noise_image = torch.rand(3, 224, 224).to(device)
            noise_image = noise_image.unsqueeze(0)  # [1, 3, 224, 224]
            
            # 编码图像
            image_features = model.encode_image(noise_image)
            image_features = F.normalize(image_features, p=2, dim=-1)
            noise_embeddings.append(image_features.cpu())
    
    return torch.cat(noise_embeddings, dim=0)

def extract_embeddings(model, preprocess, data_loader, device, class_names, forget_classes, num_samples_per_class=30):
    """
    提取遗忘集的图像和文本embeddings
    """
    model.eval()
    
    forget_class_indices = [class_names.index(c) for c in forget_classes if c in class_names]
    
    # 获取文本embeddings
    print("\nPreparing text embeddings for forget classes...")
    tokenizer = open_clip.get_tokenizer("ViT-B/32")
    text_inputs = tokenizer([f"a photo of a {c}" for c in forget_classes]).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_inputs)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        text_embeddings = text_embeddings.cpu()
    
    # 提取遗忘集图像embeddings
    print("\nExtracting forget set image embeddings...")
    forget_image_embeddings = []
    forget_labels = []
    forget_class_counts = {c: 0 for c in forget_classes}
    forget_classes_set = set(forget_classes) 
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            images, texts, flags, labels = batch
            
            for idx, (img, text, flag) in enumerate(zip(images, texts, flags)):
                if flag and text in forget_classes_set:
                    # 检查是否已达到每个类别的样本数
                    if forget_class_counts[text] >= num_samples_per_class:
                        continue
                    
                    # 预处理图像并编码
                    image_tensor = preprocess(img).unsqueeze(0).to(device)
                    image_features = model.encode_image(image_tensor)
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    
                    forget_image_embeddings.append(image_features.cpu())
                    forget_labels.append(text)
                    forget_class_counts[text] += 1
    
    if not forget_image_embeddings:
        print("Warning: No forget images found in dataset!")
        return None, None, None
    
    forget_image_embeddings = torch.cat(forget_image_embeddings, dim=0)
    print(f"Extracted {len(forget_image_embeddings)} forget images")
    
    return forget_image_embeddings, text_embeddings, forget_labels

def plot_tsne(embeddings_dict, labels_dict, colors_dict, save_path=None):
    """
    绘制t-SNE可视化图
    embeddings_dict: {'name': tensor} 多个embedding集合
    labels_dict: {'name': list} 对应的标签列表
    colors_dict: {'name': color} 对应的颜色
    """
    # 准备数据
    all_embeddings = []
    all_labels = []
    all_types = []
    
    for name, embeddings in embeddings_dict.items():
        if embeddings is None or len(embeddings) == 0:
            print(f"Warning: {name} has no data, skipping...")
            continue
        
        # 确保是numpy数组
        if torch.is_tensor(embeddings):
            embeddings = embeddings.numpy()
        
        n_samples = len(embeddings)
        all_embeddings.append(embeddings)
        
        if name in labels_dict and labels_dict[name] is not None:
            labels = labels_dict[name]
            if len(labels) != n_samples:
                # 为噪声生成标签
                if name == 'Noise':
                    labels = [f'Noise_{i}' for i in range(n_samples)]
                else:
                    # 使用默认标签
                    labels = [f'{name}_{i}' for i in range(n_samples)]
            all_labels.extend(labels)
        else:
            # 如果没有提供标签，使用类型名
            all_labels.extend([f'{name}_{i}' for i in range(n_samples)])
        
        all_types.extend([name] * n_samples)
    
    if not all_embeddings:
        print("No embeddings to visualize!")
        return None, None
    
    # 合并所有embeddings
    combined_embeddings = np.vstack(all_embeddings)
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    # t-SNE降维
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(combined_embeddings)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 为每个类型绘制点
    unique_types = list(set(all_types))
    type_colors = {}
    
    # 分配颜色
    for i, type_name in enumerate(unique_types):
        if type_name in colors_dict:
            type_colors[type_name] = colors_dict[type_name]
        else:
            # 使用循环颜色
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            type_colors[type_name] = colors[i % len(colors)]
    
    # 绘制散点
    for type_name in unique_types:
        mask = np.array([t == type_name for t in all_types])
        x = tsne_results[mask, 0]
        y = tsne_results[mask, 1]
        
        if type_name == 'Noise':
            alpha = 0.6
            size = 15
            marker = 'o'
        elif type_name == 'Text':
            alpha = 0.9
            size = 200
            marker = 's'
            # 对文本点添加标签
            for i, (xi, yi, label) in enumerate(zip(x, y, np.array(all_labels)[mask])):
                # 提取类别名称
                class_name = label.replace('a photo of a ', '')
                ax.annotate(class_name, (xi, yi), fontsize=14, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor='black', linewidth=0.5))
        else:  # Image
            alpha = 0.7
            size = 30
            marker = 'o'
        
        ax.scatter(x, y, c=[type_colors[type_name]], label=type_name, 
                  alpha=alpha, s=size, marker=marker, edgecolors='black', linewidth=0.5)
    
    # 去掉标题
    # ax.set_title(title, fontsize=18, fontweight='bold')
    
    # 放大图例
    ax.legend(loc='best', fontsize=22, framealpha=0.9)
    

    
    # 放大刻度字体
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to: {save_path}")
    
    plt.show()
    
    return tsne_results, fig

def main():
    # 配置
    config = {
        "dataset": {
            "name": "CIFAR100",
            "forget_classes": ['apple'],
            "data_root": "./data"
        },
        "training": {
            "batch_size": 128,
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
    
    torch.manual_seed(config["misc"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["misc"]["seed"])
    device = torch.device(config["misc"]["device"])
    
    print(f"Using device: {device}")
    
    forget_classes = config["dataset"]["forget_classes"]
    forget_classes_set = set(forget_classes)
    
    # 定义forget规则
    def forget_rule(sample):
        return sample["text"] in forget_classes
    
    # 加载数据集
    print("Loading CIFAR100 dataset...")
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
    
    # 统计
    forget_count = sum(1 for i in range(len(test_dataset)) if test_dataset[i]["flag"])
    retain_count = len(test_dataset) - forget_count
    print(f"Forget set samples: {forget_count}")
    print(f"Retain set samples: {retain_count}")
    
    # 加载模型
    print("\n" + "="*80)
    print("Loading OpenCLIP Model")
    print("="*80)
    
    checkpoint_path = "/home/cyt/open_clip/src/logs/cliperase_vit_L14_val_CLiperase20260518_043722/cliperase_vit_L14_val_CLiperase20260518_043722/checkpoints/epoch_20.pt"
    model, preprocess, tokenizer = load_openclip_model(
        checkpoint_path=checkpoint_path,
        model_name="ViT-L-14",
        device=device
    )
    model.eval()
    
    # 提取embeddings
    forget_image_embeddings, text_embeddings, forget_labels = extract_embeddings(
        model=model,
        preprocess=preprocess,
        data_loader=test_loader,
        device=device,
        class_names=CIFAR100_LABEL_NAMES,
        forget_classes=forget_classes,
        num_samples_per_class=100
    )
    
    if forget_image_embeddings is None:
        print("No embeddings extracted. Exiting...")
        return
    
    # 生成高斯噪声embeddings
    noise_embeddings = generate_gaussian_noise_embeddings(
        model=model,
        tokenizer=tokenizer,
        num_samples=100,
        device=device
    )
    
    print(f"Forget image embeddings shape: {forget_image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Noise embeddings shape: {noise_embeddings.shape}")
    
    # 准备t-SNE数据
    embeddings_dict = {
        'Forget Images': forget_image_embeddings,
        'Text': text_embeddings,
        'Noise': noise_embeddings
    }
    
    labels_dict = {
        'Forget Images': forget_labels,
        'Text': [f"a photo of a {c}" for c in forget_classes],
        'Noise': None
    }
    
    # 使用亮色系：亮蓝色 (#2979ff), 亮红色 (#ff5252), 亮绿色 (#00e676)
    colors_dict = {
        'Forget Images': '#2979ff',  # 亮蓝色
        'Text': '#ff5252',           # 亮红色
        'Noise': '#00e676'           # 亮绿色
    }

    # 绘制t-SNE
    output_dir = "./tsne_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tsne_forget_images_text_noise.png")
    
    tsne_results, fig = plot_tsne(
        embeddings_dict=embeddings_dict,
        labels_dict=labels_dict,
        colors_dict=colors_dict,
        save_path=save_path
    )
    
    print("\n" + "="*80)
    print("t-SNE visualization completed!")
    print(f"Results saved to: {save_path}")
    print("="*80)

if __name__ == "__main__":
    main()