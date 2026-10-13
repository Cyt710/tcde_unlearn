import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import sys
import logging
from typing import List, Dict, Any, Tuple
import clip

def setup_logging(log_dir: str = "./logs", log_name: str = "training.log"):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def safe_tokenize(texts, device):
    tokenized_texts = []
    for text in texts:
        text_str = str(text)
        try:
            tokens = clip.tokenize(text_str, truncate=True)
            tokenized_texts.append(tokens)
        except RuntimeError as e:
            logging.warning(f"Tokenization failed, further truncating text: {e}")
            short_text = text_str[:80]
            tokens = clip.tokenize(short_text, truncate=True)
            tokenized_texts.append(tokens)
    
    return torch.cat(tokenized_texts).to(device)

def move_to_device(batch, device, mode="normal"):
    if mode == "normal":
        images, texts, labels = batch
        
        if isinstance(labels, (list, tuple)):
            labels_tensor = torch.tensor(labels).to(device)
        else:
            labels_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
            
        return images, texts, labels_tensor
    else:
        images, texts, flags, labels = batch
        
        flags_tensor = torch.tensor(flags).to(device)
        
        if isinstance(labels, (list, tuple)):
            labels_tensor = torch.tensor(labels).to(device)
        else:
            labels_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
            
        return images, texts, flags_tensor, labels_tensor

def train_zero_shot(model, original_model, train_loader, val_loader, optimizer, criterion, device, epochs, checkpoint_helper, mode="normal"):
    logging.info(f"Starting training - Mode: {mode}")
    logging.info(f"Training set size: {len(train_loader.dataset)}, Validation set size: {len(val_loader.dataset)}")
    logging.info(f"Device: {device}, Training epochs: {epochs}")
    
    logging.info("Verifying model device:")
    model_params_device = next(model.parameters()).device
    logging.info(f"Model parameters device: {model_params_device}")
    logging.info(f"Target device: {device}")
    
    if model_params_device != device:
        logging.warning(f"Model not on target device! Moving model...")
        model = model.to(device)
        original_model = original_model.to(device)
        logging.info(f"Model moved to {device}")
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        batch_count = 0
        
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, batch in enumerate(train_iterator):
            if mode == "normal":
                images, texts, labels = move_to_device(batch, device, mode)
                flags_tensor = torch.zeros(len(images), dtype=torch.bool).to(device)
            else:
                images, texts, flags, labels = move_to_device(batch, device, mode)
                flags_tensor = torch.tensor(flags).to(device)
            
            try:
                preprocessed_images = []
                for img in images:
                    img_tensor = model.preprocess(img)
                    if hasattr(img_tensor, 'to'):
                        img_tensor = img_tensor.to(device)
                    preprocessed_images.append(img_tensor)
                image_tensor = torch.stack(preprocessed_images).to(device)
            except Exception as e:
                logging.error(f"Image preprocessing failed: {e}")
                image_tensor = torch.stack([img.to(device) if hasattr(img, 'to') else torch.tensor(img).to(device) 
                                          for img in images])
            
            text_tokens = safe_tokenize(texts, device)
            
            if batch_idx == 0:
                logging.info(f"Image tensor device: {image_tensor.device}")
                logging.info(f"Text token device: {text_tokens.device}")
            
            try:
                batch_image_features = model.encode_image(image_tensor)
                batch_text_features = model.encode_text(text_tokens)
                
                if batch_idx == 0:
                    logging.info(f"Image features device: {batch_image_features.device}")
                    logging.info(f"Text features device: {batch_text_features.device}")
            except Exception as e:
                logging.error(f"Feature extraction failed: {e}")
                continue
            
            with torch.no_grad():
                try:
                    original_batch_image_features = original_model.encode_image(image_tensor)
                    original_batch_text_features = original_model.encode_text(text_tokens)
                except Exception as e:
                    logging.error(f"Original model feature extraction failed: {e}")
                    original_batch_image_features = batch_image_features.detach()
                    original_batch_text_features = batch_text_features.detach()
            
            if mode == "normal":
                loss, loss_dict = criterion(
                    batch_image_features, batch_text_features,
                    original_batch_image_features, original_batch_text_features, 
                    flags_tensor
                )
            else:
                stage_flag = 1 if epoch < 30 else 0
                loss, loss_dict = criterion(
                    batch_image_features, batch_text_features,
                    original_batch_image_features, original_batch_text_features, 
                    flags_tensor
                )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
            else:
                loss_value = loss
            
            total_train_loss += loss_value
            batch_count += 1

            if mode == "normal":
                train_iterator.set_postfix({
                    'Avg Loss': f'{total_train_loss / batch_count:.4f}'
                })
            else:
                retention_val = loss_dict.get("retention_loss", 0)
                forgetting_val = loss_dict.get("forgetting_loss", 0)
                uniform_val = loss_dict.get("uniform_loss", 0)
                dist_val = loss_dict.get("dist_loss", 0)
                consistency_val = loss_dict.get("consistency_loss", 0)
                
                if isinstance(retention_val, torch.Tensor):
                    retention_val = retention_val.item()
                if isinstance(forgetting_val, torch.Tensor):
                    forgetting_val = forgetting_val.item()
                if isinstance(uniform_val, torch.Tensor):
                    uniform_val = uniform_val.item()
                if isinstance(dist_val, torch.Tensor):
                    dist_val = dist_val.item()
                if isinstance(consistency_val, torch.Tensor):
                    consistency_val = consistency_val.item()
                
                train_iterator.set_postfix({
                    'Loss': f'{loss_value:.4f}',
                    'Retention': f'{retention_val:.4f}',
                    'Forgetting': f'{forgetting_val:.4f}',
                    'uniform_loss': f'{uniform_val:.4f}',
                    'dist_loss': f'{dist_val:.4f}',
                    'Consistency': f'{consistency_val:.4f}'
                })
        
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0.0
        
        logging.info(f"Starting validation phase...")
        avg_val_loss = validate_model_loss_only(
            model, original_model, val_loader, criterion, device, mode=mode
        )
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['epochs'].append(epoch + 1)
        
        logging.info(f"Epoch {epoch+1}/{epochs} completed")
        logging.info(f"Training loss: {avg_train_loss:.6f}, Validation loss: {avg_val_loss:.6f}")
        
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint_helper.save_checkpoint(
            model, optimizer, epoch+1, avg_val_loss, checkpoint_filename
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            logging.info(f"New best model! Validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
        else:
            logging.info(f"Current validation loss {avg_val_loss:.6f} not better than best {best_val_loss:.6f}")
    
    logging.info("="*50)
    logging.info("Training completed!")
    logging.info("="*50)
    logging.info(f"Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    logging.info(f"Training history:")
    logging.info(f"  Final training loss: {train_history['train_loss'][-1]:.6f}")
    logging.info(f"  Final validation loss: {train_history['val_loss'][-1]:.6f}")
    
    history_path = os.path.join(checkpoint_helper.checkpoint_dir, "train_history.json")
    import json
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    logging.info(f"Training history saved to: {history_path}")
    
    return best_val_loss, train_history
    
def validate_model_loss_only(model, original_model, val_loader, criterion, device, mode="normal"):
    model.eval()
    total_val_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc='Validating')
        for batch_idx, batch in enumerate(val_iterator):
            if mode == "normal":
                images, texts, labels = move_to_device(batch, device, mode)
                flags_tensor = torch.zeros(len(images), dtype=torch.bool).to(device)
            else:
                images, texts, flags, labels = move_to_device(batch, device, mode)
                flags_tensor = torch.tensor(flags).to(device)
            
            try:
                preprocessed_images = []
                for img in images:
                    img_tensor = model.preprocess(img)
                    if hasattr(img_tensor, 'to'):
                        img_tensor = img_tensor.to(device)
                    preprocessed_images.append(img_tensor)
                image_tensor = torch.stack(preprocessed_images).to(device)
            except Exception as e:
                logging.error(f"Validation image preprocessing failed: {e}")
                image_tensor = torch.stack([img.to(device) if hasattr(img, 'to') else torch.tensor(img).to(device) 
                                          for img in images])
            
            text_tokens = safe_tokenize(texts, device)
            
            batch_image_features = model.encode_image(image_tensor)
            batch_text_features = model.encode_text(text_tokens)
            
            original_batch_image_features = original_model.encode_image(image_tensor)
            original_batch_text_features = original_model.encode_text(text_tokens)
            
            val_loss, loss_dict = criterion(
                batch_image_features, batch_text_features,
                original_batch_image_features, original_batch_text_features, 
                flags_tensor
            )
            
            if isinstance(val_loss, torch.Tensor):
                val_loss_value = val_loss.item()
            else:
                val_loss_value = val_loss
            
            total_val_loss += val_loss_value
            batch_count += 1
            
            val_iterator.set_postfix({
                'Val Loss': f'{val_loss_value:.4f}',
                'Avg Loss': f'{total_val_loss / batch_count:.4f}'
            })
    
    avg_val_loss = total_val_loss / batch_count if batch_count > 0 else 0.0
    return avg_val_loss


if __name__ == "__main__":
    logger = setup_logging(log_dir="./training_logs", log_name="train_flickr.log")
    
    logging.info("GPU training script loaded")