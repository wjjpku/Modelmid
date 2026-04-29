import os
import sys
import json
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# Import the custom feature extractor from train_classifier
sys.path.append(os.path.join(base_dir, 'scripts', 'model_training'))
from train_classifier import TextFeatureExtractor, DenseTransformer

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

def load_chinese_data():
    json_path = os.path.join(base_dir, 'dataset', 'generalization', 'test_100_chinese_archive_questions.json')
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return None
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        q_id = row['id']
        if row.get('human'): records.append({'id': q_id, 'text': row['human'], 'label': 'Human'})
        if row.get('deepseek'): records.append({'id': q_id, 'text': row['deepseek'], 'label': 'Deepseek'})
        if row.get('kimi'): records.append({'id': q_id, 'text': row['kimi'], 'label': 'Kimi'})
        if row.get('glm'): records.append({'id': q_id, 'text': row['glm'], 'label': 'GLM'})
        if row.get('qwen'): records.append({'id': q_id, 'text': row['qwen'], 'label': 'Qwen'})
        
    return pd.DataFrame(records)

def evaluate_ml_model(df):
    print("\n" + "="*50)
    print("🤖 EVALUATING BEST ML MODEL (HistGradientBoosting) ON CHINESE DATA")
    print("="*50)
    
    model_path = os.path.join(base_dir, 'models', 'best_classifier_model.pkl')
    if not os.path.exists(model_path):
        print("ML Model not found.")
        return
        
    with open(model_path, 'rb') as f:
        best_pipeline = pickle.load(f)
        
    X_test = pd.DataFrame({'text': df['text']})
    y_test = df['label']
    
    print("Running inference...")
    y_pred = best_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal ML Model Test Accuracy on Chinese Data (500 records): {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
def evaluate_dl_model(df):
    print("\n" + "="*50)
    print("🤖 EVALUATING BEST DL MODEL (E2E Transformer) ON CHINESE DATA")
    print("="*50)
    
    le = LabelEncoder()
    le.fit(['Deepseek', 'GLM', 'Human', 'Kimi', 'Qwen'])
    test_labels = le.transform(df['label'].values)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_path = os.path.join(base_dir, 'models', 'e2e_transformer_best.pt')
    if not os.path.exists(model_path):
        print(f"DL Model not found at {model_path}")
        return
        
    print(f"Loading DL model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    test_dataset = MathTextDataset(df['text'].values, test_labels, tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    final_preds = []
    final_true = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            
            final_preds.extend(preds.cpu().numpy())
            final_true.extend(b_labels.cpu().numpy())
            
    acc = accuracy_score(final_true, final_preds)
    print(f"\nFinal DL Model Test Accuracy on Chinese Data (500 records): {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(final_true, final_preds, target_names=le.classes_))

if __name__ == '__main__':
    df = load_chinese_data()
    if df is not None and len(df) > 0:
        evaluate_ml_model(df)
        evaluate_dl_model(df)
