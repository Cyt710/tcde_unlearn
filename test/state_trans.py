import torch
import os
import shutil
from transformers import CLIPTextModel, CLIPTokenizer


def convert_openclip_to_sd_fixed(
    openclip_checkpoint_path: str,
    output_dir: str,
    template_model_path: str
):
    """
    将OpenCLIP训练的ViT-L/14模型转换为Stable Diffusion可用的text encoder格式
    自动处理 'module.' 前缀
    """
    
    print("="*60)
    print("OpenCLIP to Stable Diffusion Converter (FIXED v2)")
    print("="*60)
    
    print(f"\n📥 Loading checkpoint from: {openclip_checkpoint_path}")
    checkpoint = torch.load(openclip_checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"✅ Loaded checkpoint with {len(state_dict)} parameters")
    
    # 🔧 检测是否有 'module.' 前缀
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    print(f"🔍 Detected 'module.' prefix: {has_module_prefix}")
    
    # 🔧 如果有 'module.' 前缀，先去掉
    if has_module_prefix:
        print("🔄 Removing 'module.' prefix from all keys...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # 去掉 'module.'
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        print(f"✅ Removed 'module.' prefix. Now {len(state_dict)} parameters")
    
    # 打印文本相关的key数量（用于调试）
    text_keys = [k for k in state_dict.keys() if 'transformer' in k or 
                 'positional_embedding' in k or 'token_embedding' in k or 
                 'ln_final' in k or 'text_projection' in k]
    print(f"📋 Found {len(text_keys)} text-related keys in checkpoint")
    
    if len(text_keys) == 0:
        print("❌ No text-related keys found! Please check checkpoint format.")
        # 打印前20个key帮助调试
        print("\n📋 First 20 keys in checkpoint:")
        for i, k in enumerate(list(state_dict.keys())[:20]):
            print(f"   {i}: {k}")
        return None
    
    # 创建输出目录并复制配置文件
    print(f"\n📁 Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("📋 Copying tokenizer and config from template...")
    for file in os.listdir(template_model_path):
        if file in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
            src = os.path.join(template_model_path, file)
            dst = os.path.join(output_dir, file)
            shutil.copy2(src, dst)
            print(f"   Copied: {file}")
    
    # 加载模板模型
    print(f"\n📁 Loading template model from: {template_model_path}")
    sd_text_encoder = CLIPTextModel.from_pretrained(template_model_path)
    template_state_dict = sd_text_encoder.state_dict()
    
    print("\n🔄 Mapping text encoder weights...")
    
    sd_state_dict = {}
    mapped_count = 0
    
    # ============================================================
    # 1. 顶层映射（现在key已经没有'module.'前缀了）
    # ============================================================
    mapping_rules = {
        'positional_embedding': 'text_model.embeddings.position_embedding.weight',
        'token_embedding.weight': 'text_model.embeddings.token_embedding.weight',
        'ln_final.weight': 'text_model.final_layer_norm.weight',
        'ln_final.bias': 'text_model.final_layer_norm.bias',
        'text_projection': 'text_model.projection.weight',
    }
    
    print("\n📌 Mapping top-level weights:")
    for openclip_key, sd_key in mapping_rules.items():
        if openclip_key in state_dict:
            value = state_dict[openclip_key]
            if sd_key in template_state_dict:
                target_shape = template_state_dict[sd_key].shape
                if value.shape == target_shape:
                    sd_state_dict[sd_key] = value
                    mapped_count += 1
                    print(f"   ✅ Mapped: {openclip_key} -> {sd_key} {value.shape}")
                else:
                    print(f"   ⚠️ Shape mismatch: {openclip_key} {value.shape} vs {target_shape}")
            else:
                print(f"   ⚠️ Target key not found: {sd_key}")
        else:
            print(f"   ⚠️ Source key not found: {openclip_key}")
    
    # ============================================================
    # 2. Transformer层映射（12层）
    # ============================================================
    print("\n📌 Mapping transformer layers:")
    
    # 🔧 检测transformer层数（支持12层ViT-L/14）
    num_layers = 0
    for k in state_dict.keys():
        if k.startswith('transformer.resblocks.'):
            layer_idx = int(k.split('.')[2])
            if layer_idx >= num_layers:
                num_layers = layer_idx + 1
    print(f"   Detected {num_layers} transformer layers")
    
    for i in range(num_layers):
        prefix = f'transformer.resblocks.{i}'
        sd_prefix = f'text_model.encoder.layers.{i}'
        
        # LayerNorm1
        for suffix in ['weight', 'bias']:
            key = f'{prefix}.ln_1.{suffix}'
            if key in state_dict:
                sd_key = f'{sd_prefix}.layer_norm1.{suffix}'
                if sd_key in template_state_dict:
                    sd_state_dict[sd_key] = state_dict[key]
                    mapped_count += 1
        
        # Attention: in_proj_weight (拆分为 q, k, v)
        key = f'{prefix}.attn.in_proj_weight'
        if key in state_dict:
            value = state_dict[key]
            if value.shape[0] % 3 == 0:
                split_size = value.shape[0] // 3
                sd_state_dict[f'{sd_prefix}.self_attn.q_proj.weight'] = value[:split_size]
                sd_state_dict[f'{sd_prefix}.self_attn.k_proj.weight'] = value[split_size:2*split_size]
                sd_state_dict[f'{sd_prefix}.self_attn.v_proj.weight'] = value[2*split_size:]
                mapped_count += 3
        
        # Attention: in_proj_bias (拆分为 q, k, v)
        key = f'{prefix}.attn.in_proj_bias'
        if key in state_dict:
            value = state_dict[key]
            if value.shape[0] % 3 == 0:
                split_size = value.shape[0] // 3
                sd_state_dict[f'{sd_prefix}.self_attn.q_proj.bias'] = value[:split_size]
                sd_state_dict[f'{sd_prefix}.self_attn.k_proj.bias'] = value[split_size:2*split_size]
                sd_state_dict[f'{sd_prefix}.self_attn.v_proj.bias'] = value[2*split_size:]
                mapped_count += 3
        
        # Attention: out_proj
        for suffix in ['weight', 'bias']:
            key = f'{prefix}.attn.out_proj.{suffix}'
            if key in state_dict:
                sd_key = f'{sd_prefix}.self_attn.out_proj.{suffix}'
                if sd_key in template_state_dict:
                    sd_state_dict[sd_key] = state_dict[key]
                    mapped_count += 1
        
        # LayerNorm2
        for suffix in ['weight', 'bias']:
            key = f'{prefix}.ln_2.{suffix}'
            if key in state_dict:
                sd_key = f'{sd_prefix}.layer_norm2.{suffix}'
                if sd_key in template_state_dict:
                    sd_state_dict[sd_key] = state_dict[key]
                    mapped_count += 1
        
        # MLP: fc1 (c_fc)
        for suffix in ['weight', 'bias']:
            key = f'{prefix}.mlp.c_fc.{suffix}'
            if key in state_dict:
                sd_key = f'{sd_prefix}.mlp.fc1.{suffix}'
                if sd_key in template_state_dict:
                    sd_state_dict[sd_key] = state_dict[key]
                    mapped_count += 1
        
        # MLP: fc2 (c_proj)
        for suffix in ['weight', 'bias']:
            key = f'{prefix}.mlp.c_proj.{suffix}'
            if key in state_dict:
                sd_key = f'{sd_prefix}.mlp.fc2.{suffix}'
                if sd_key in template_state_dict:
                    sd_state_dict[sd_key] = state_dict[key]
                    mapped_count += 1
        
        # 每层打印进度
        if (i + 1) % 4 == 0 or i == num_layers - 1:
            print(f"   ✅ Layer {i+1}/{num_layers} done (mapped {mapped_count} total)")
    
    print(f"\n✅ Mapped {mapped_count} weights total")
    
    # 加载权重到SD模型
    print("\n📥 Loading weights to SD model...")
    loaded_count = 0
    
    for key, value in sd_state_dict.items():
        if key in template_state_dict:
            if template_state_dict[key].shape == value.shape:
                template_state_dict[key].copy_(value)
                loaded_count += 1
            else:
                print(f"   ⚠️ Shape mismatch for {key}: {value.shape} vs {template_state_dict[key].shape}")
        else:
            print(f"   ⚠️ Key not in target: {key}")
    
    print(f"✅ Loaded {loaded_count} weights to SD model")
    
    # 保存模型
    print(f"\n💾 Saving converted model to: {output_dir}")
    sd_text_encoder.save_pretrained(output_dir)
    
    print(f"✅ Conversion completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    
    # 验证
    print("\n🔍 Verifying saved model...")
    verify_model = CLIPTextModel.from_pretrained(output_dir)
    verify_tokenizer = CLIPTokenizer.from_pretrained(output_dir)
    
    test_text = ["a photo of an apple"]
    inputs = verify_tokenizer(test_text, return_tensors="pt", padding=True)
    outputs = verify_model(**inputs)
    print(f"✅ Verification successful! Output shape: {outputs.last_hidden_state.shape}")
    
    return output_dir


def main():
    config = {
        "openclip_checkpoint": "/home/cyt/open_clip/src/logs/cliperase_vit_B32_val_CLiperase20260508_141004/cliperase_vit_B32_val_CLiperase20260508_141004/checkpoints/epoch_20.pt",
        "template_model": "/home/cyt/open_clip/tests/converted_l14_text",
        "output_dir": "./converted_GD_clip_to_sd_vit_l14_fixed"
    }
    
    print(f"📂 OpenCLIP checkpoint: {config['openclip_checkpoint']}")
    print(f"📂 Template model: {config['template_model']}")
    print(f"📂 Output directory: {config['output_dir']}")
    
    if not os.path.exists(config['openclip_checkpoint']):
        print(f"\n❌ Checkpoint file not found: {config['openclip_checkpoint']}")
        return
    
    if not os.path.exists(config['template_model']):
        print(f"\n❌ Template model not found: {config['template_model']}")
        return
    
    try:
        convert_openclip_to_sd_fixed(
            openclip_checkpoint_path=config['openclip_checkpoint'],
            output_dir=config['output_dir'],
            template_model_path=config['template_model']
        )
        print("\n" + "="*60)
        print("🎉 Conversion completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()