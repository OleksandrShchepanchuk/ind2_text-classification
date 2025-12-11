import torch
import mlflow
import os
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

# Optimized hardware settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_llrd_params(model, peak_lr, weight_decay=0.01, layer_decay=0.95):

    opt_parameters = []
    
    # 1. Head (Classifier) - Highest LR
    head_params = list(model.backbone.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    opt_parameters.append({
        "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
        "lr": peak_lr, "weight_decay": weight_decay
    })
    opt_parameters.append({
        "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
        "lr": peak_lr, "weight_decay": 0.0
    })

    # 2. Body (Transformer Layers) - Decaying LR
    # Automatically finds layers for BERT (12) 
    encoder_layers = model.backbone.base_model.encoder.layer
    
    for i, layer in enumerate(reversed(encoder_layers)):
        layer_lr = peak_lr * (layer_decay ** (i + 1)) # Decay as we go down
        
        opt_parameters.append({
            "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": layer_lr, "weight_decay": weight_decay
        })
        opt_parameters.append({
            "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": layer_lr, "weight_decay": 0.0
        })

    # 3. Embeddings - Lowest LR
    embed_lr = peak_lr * (layer_decay ** (len(encoder_layers) + 1))
    embed_params = list(model.backbone.base_model.embeddings.named_parameters())
    
    opt_parameters.append({
        "params": [p for n, p in embed_params if not any(nd in n for nd in no_decay)],
        "lr": embed_lr, "weight_decay": weight_decay
    })
    opt_parameters.append({
        "params": [p for n, p in embed_params if any(nd in n for nd in no_decay)],
        "lr": embed_lr, "weight_decay": 0.0
    })
    
    return opt_parameters

def train_one_epoch(model, loader, optimizer, device, scheduler, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    correct = 0
    count = 0
    
    pbar = tqdm(loader, desc="Training")
    
    # Auto-detect precision (BFloat16 for Ampere+, Float16 for T4/V100)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # --- Mixed Precision Forward ---
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            outputs = model(ids, mask, labels=labels)
            # CRITICAL: Normalize loss for gradient accumulation
            loss = outputs["loss"] / accumulation_steps
        
        # --- Backward with Scaling ---
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Only update weights every N steps
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            # Fallback for BF16 (no scaler needed usually, but logic stays same)
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        
        # Track metrics (multiply loss back to report correct value)
        total_loss += loss.item() * accumulation_steps
        
        # Detach predictions to free graph memory immediately
        preds = torch.argmax(outputs["logits"], dim=1).detach()
        correct += (preds == labels).sum().item()
        count += labels.size(0)
        
        pbar.set_postfix({'loss': f"{total_loss/(step+1):.4f}", 'acc': f"{correct/count:.4f}"})
        
    return total_loss / len(loader), correct / count

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    count = 0
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad(): 
        for batch in loader:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(ids, mask, labels=labels)
            
            total_loss += outputs["loss"].item()
            preds = torch.argmax(outputs["logits"], dim=1)
            correct += (preds == labels).sum().item()
            count += labels.size(0)
            
    return total_loss / len(loader), correct / count

def run_training(model, train_loader, val_loader, device_str, epochs, 
                 tracking_uri, experiment_name, save_dir, 
                 optimizer=None, 
                 accumulation_steps=1,learning_rate = 2e-5, layer_decay=0.95, run_name=None): 
    
    device = torch.device(device_str)
    model.to(device)
    
    use_scaler = not torch.cuda.is_bf16_supported()
    scaler = GradScaler() if use_scaler else None
    
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default") 
        except Exception as e:
            print(f"Compilation failed, falling back to eager mode: {e}")
            pass 

    if optimizer is None:
        optimizer_grouped_parameters = get_llrd_params(
            model, 
            peak_lr=learning_rate,      
            layer_decay=layer_decay   
        )

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    total_steps = (len(train_loader) // accumulation_steps) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    best_acc = 0.0
    
    with mlflow.start_run(run_name=run_name):
        # Log key hyperparameters and run context
        logged_batch_size = getattr(train_loader, "batch_size", None)
        logged_model_name = getattr(getattr(model, "backbone", None), "name_or_path", None)
        if logged_model_name is None:
            logged_model_name = getattr(getattr(model, "config", None), "name_or_path", model.__class__.__name__)

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": logged_batch_size,
            "learning_rate": learning_rate,
            "layer_decay": layer_decay,
            "accumulation_steps": accumulation_steps,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "optimizer": optimizer.__class__.__name__,
            "device": device_str,
            "model_name": logged_model_name,
            "precision": "mixed_bfloat16" if torch.cuda.is_bf16_supported() else "mixed_float16"
        })
        
        for epoch in range(epochs):
            # Pass correct args to train_one_epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, scheduler, scaler, accumulation_steps
            )
            
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Handle compiled model saving
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(save_model.state_dict(), os.path.join(save_dir, f"best_model_{run_name}_{best_acc:.4f}.pth"))
                
    return best_acc


def generate_submission(model, test_loader, test_ids, device_str, id_to_label, output_path="submission.csv"):
    device = torch.device(device_str)
    model.to(device)
    model.eval()
    
    predictions = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Submission"):
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(ids, mask)
            
            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            predictions.extend(preds)
            
    predicted_labels = [id_to_label[p.item()] for p in predictions]
    
    import pandas as pd
    submission = pd.DataFrame({'id': test_ids, 'label': predicted_labels})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission