# extract_and_convert_weights.py
import torch
import os

def convert_cliperase_to_standard(state_dict):
    """将CLIPErase模型权重转换为标准CLIP格式"""
    new_state_dict = {}
    
    # 映射规则
    mapping = {
        # 图像编码器部分
        'image_encoder.class_embedding': 'visual.class_embedding',
        'image_encoder.positional_embedding': 'visual.positional_embedding',
        'image_encoder.proj': 'visual.proj',
        'image_encoder.conv1.weight': 'visual.conv1.weight',
        'image_encoder.ln_pre.weight': 'visual.ln_pre.weight',
        'image_encoder.ln_pre.bias': 'visual.ln_pre.bias',
        'image_encoder.ln_post.weight': 'visual.ln_post.weight',
        'image_encoder.ln_post.bias': 'visual.ln_post.bias',
        
        # 文本编码器部分
        'text_encoder.positional_embedding': 'positional_embedding',
        'text_encoder.text_projection': 'text_projection',
        'text_encoder.logit_scale': 'logit_scale',
        'text_encoder.token_embedding.weight': 'token_embedding.weight',
        'text_encoder.ln_final.weight': 'ln_final.weight',
        'text_encoder.ln_final.bias': 'ln_final.bias',
    }
    
    # 处理所有键
    for old_key, value in state_dict.items():
        new_key = old_key
        
        # 特殊处理：移除 text_encoder.visual. 前缀
        if old_key.startswith('text_encoder.visual.'):
            new_key = old_key.replace('text_encoder.visual.', 'visual.')
        
        # 应用映射规则
        for old_prefix, new_prefix in mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                break
        
        # 处理transformer层（有特殊模式）
        if 'image_encoder.transformer.resblocks.' in old_key:
            # 图像编码器的transformer层
            new_key = old_key.replace('image_encoder.transformer.resblocks.', 'visual.transformer.resblocks.')
        
        elif 'text_encoder.transformer.resblocks.' in old_key:
            # 文本编码器的transformer层
            new_key = old_key.replace('text_encoder.transformer.resblocks.', 'transformer.resblocks.')
        
        # 处理text_encoder.visual.transformer的特殊情况
        elif 'text_encoder.visual.transformer.resblocks.' in old_key:
            new_key = old_key.replace('text_encoder.visual.transformer.resblocks.', 'visual.transformer.resblocks.')
        
        new_state_dict[new_key] = value
        # print(f"{old_key} -> {new_key}")  # 调试用
    
    return new_state_dict

def extract_and_convert(checkpoint_path, output_path):
    """提取并转换权重"""
    print(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"原始权重键数量: {len(state_dict)}")
    print(f"前5个键: {list(state_dict.keys())[:5]}")
    
    # 转换权重格式
    print("转换权重格式...")
    converted_state_dict = convert_cliperase_to_standard(state_dict)
    
    print(f"转换后权重键数量: {len(converted_state_dict)}")
    print(f"前5个转换后的键: {list(converted_state_dict.keys())[:5]}")
    
    # 保存
    torch.save(converted_state_dict, output_path)
    print(f"✓ 转换后的权重已保存到: {output_path}")
    
    return converted_state_dict

if __name__ == "__main__":
    checkpoint_path = "/home/cyt/cliperase/dataset/CIFAR100/checkpoints/flickr_normal_20260128_132055/checkpoint_epoch_50.pt"
    output_path = "./standard_clip_weights.pt"
    
    extract_and_convert(checkpoint_path, output_path)
    
    # 验证
    print("\n验证转换后的权重...")
    test_state_dict = torch.load(output_path, map_location='cpu')
    
    # 检查关键键是否存在
    required_keys = [
        'visual.class_embedding',
        'visual.positional_embedding',
        'visual.proj',
        'positional_embedding',
        'text_projection',
        'logit_scale'
    ]
    
    print("关键键检查:")
    for key in required_keys:
        exists = key in test_state_dict
        print(f"  {key}: {'✓' if exists else '✗'}")