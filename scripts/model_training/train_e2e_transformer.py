import os
import sys
import json
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Dataset Preparation
class MathTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_split_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    q_df_list = []
    
    for row in data:
        q_id = row['id']
        subject = row.get('subject', 'unknown')
        q_df_list.append({'id': q_id, 'subject': subject})
        
        if row.get('human'): records.append({'id': q_id, 'text': row['human'], 'label': 'Human'})
        if row.get('deepseek'): records.append({'id': q_id, 'text': row['deepseek'], 'label': 'Deepseek'})
        if row.get('kimi'): records.append({'id': q_id, 'text': row['kimi'], 'label': 'Kimi'})
        if row.get('glm'): records.append({'id': q_id, 'text': row['glm'], 'label': 'GLM'})
        if row.get('qwen'): records.append({'id': q_id, 'text': row['qwen'], 'label': 'Qwen'})
        
    df = pd.DataFrame(records)
    q_df = pd.DataFrame(q_df_list).drop_duplicates('id')
    
    # Stratified Split (1800 / 200)
    train_size = min(1800, len(q_df) - 1)
    train_q, test_q = train_test_split(q_df, train_size=train_size, stratify=q_df['subject'], random_state=42)
    
    train_df = df[df['id'].isin(set(train_q['id']))].copy()
    test_df = df[df['id'].isin(set(test_q['id']))].copy()
    
    return train_df, test_df

# 2. Main Training Function
def train_e2e_model():
    print("="*50)
    print("🤖 STARTING END-TO-END LLM CLASSIFICATION EXPERIMENT")
    print("="*50)
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    json_path = os.path.join(base_dir, 'dataset', 'full_dataset_pro.json')
    
    train_df, test_df = load_and_split_data(json_path)
    
    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['label'].values)
    test_labels = le.transform(test_df['label'].values)
    
    print(f"Data Split: {len(train_df)} Train records, {len(test_df)} Test records.")
    
    # 3. Model & Tokenizer Selection
    # Using a fast, highly capable embedding/encoder model (distilbert is robust and light)
    model_name = "distilbert-base-uncased"
    print(f"Loading Pretrained Model: {model_name} (This will extract semantic embeddings E2E)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")
    
    model.to(device)
    
    # 4. DataLoaders
    # Batch size 16 to avoid OOM on Mac MPS, Max Length 512 for full context
    batch_size = 16 
    train_dataset = MathTextDataset(train_df['text'].values, train_labels, tokenizer, max_length=512)
    test_dataset = MathTextDataset(test_df['text'].values, test_labels, tokenizer, max_length=512)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 5. Optimizer & Scheduler
    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    # 6. Training Loop with Checkpointing and Early Stopping
    best_acc = 0.0
    patience = 2
    epochs_no_improve = 0
    start_epoch = 0
    
    checkpoint_dir = os.path.join(base_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'e2e_transformer_latest.pt')
    best_model_path = os.path.join(checkpoint_dir, 'e2e_transformer_best.pt')
    
    # Check for existing checkpoint to resume
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Found checkpoint! Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resumed from epoch {start_epoch} with Best Acc: {best_acc:.4f}\n")
    
    t0_total = time.time()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        t0_epoch = time.time()
        
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
            optimizer.step()
            scheduler.step()
            
            # Print every 100 steps (~1-2 minutes)
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_attn_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                
                outputs = model(b_input_ids, attention_mask=b_attn_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(b_labels.cpu().numpy())
                
        val_acc = accuracy_score(val_true, val_preds)
        epoch_time = time.time() - t0_epoch
        
        print(f"✅ Epoch {epoch+1}/{epochs} Summary: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    🌟 New Best Accuracy! Model saved to: {best_model_path}")
        else:
            epochs_no_improve += 1
            
        # Save latest checkpoint (overwrites old one)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'epochs_no_improve': epochs_no_improve
        }, checkpoint_path)
        print(f"    💾 Checkpoint for epoch {epoch+1} saved to: {checkpoint_path}")
            
        if epochs_no_improve >= patience:
            print(f"⚠️ Early stopping triggered at epoch {epoch+1}.")
            break
            
    total_time = time.time() - t0_total
    print("\n" + "="*50)
    print(f"🏆 E2E TRAINING COMPLETE 🏆")
    print(f"Total Time: {total_time:.2f}s | Best Validation Accuracy: {best_acc:.4f}")
    
    # Load best model and generate full report
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    final_preds = []
    final_true = []
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            
            final_preds.extend(preds.cpu().numpy())
            final_true.extend(b_labels.cpu().numpy())
            
    print("\nDetailed Classification Report:")
    print(classification_report(final_true, final_preds, target_names=le.classes_))
    
    # Save Results & Model (Optional)
    output_dir = os.path.join(base_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(output_dir, 'e2e_transformer_best.pt'))
    
    # Save report
    results = {
        "Model": "End-to-End DistilBERT",
        "Type": "E2E Deep Learning (Transformers)",
        "Accuracy": best_acc,
        "Time(s)": total_time,
        "Epochs": epoch + 1
    }
    
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    pd.DataFrame([results]).to_csv(os.path.join(docs_dir, 'e2e_dl_results.csv'), index=False)

if __name__ == '__main__':
    train_e2e_model()
