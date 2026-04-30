import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False

SOURCE_FIELDS = [
    ('human', 'Human'),
    ('deepseek', 'Deepseek'),
    ('kimi', 'Kimi'),
    ('glm', 'GLM'),
    ('qwen', 'Qwen'),
    ('gpt_4_1_mini', 'GPT-4.1-mini'),
]

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(os.path.join(base_dir, 'scripts'))
sys.path.append(os.path.join(base_dir, 'scripts', 'model_training'))
from plotting_utils import configure_matplotlib_fonts
from train_classifier import TextFeatureExtractor, DenseTransformer

configure_matplotlib_fonts()

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

def load_clean_data():
    json_path = os.path.join(base_dir, 'dataset', 'generalization', 'test_100_new_questions.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        q_id = row['id']
        for field_name, label in SOURCE_FIELDS:
            if row.get(field_name):
                records.append({'id': q_id, 'text': row[field_name], 'label': label})
        
    return pd.DataFrame(records)

def plot_cm(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    
    plt.title(title, fontsize=16, pad=15)
    plt.ylabel('True Label (真实标签)', fontsize=14)
    plt.xlabel('Predicted Label (预测标签)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def evaluate_dl_model(df, classes, y_true_text, y_true_encoded, label_encoder):
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping DL evaluation: transformers is not installed.")
        return None, None

    print("\n--- Evaluating DL Model (E2E DistilBERT) ---")
    model_name = "distilbert-base-uncased"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))
    except Exception as exc:
        print(f"Skipping DL evaluation: failed to load base model/tokenizer ({exc}).")
        return None, None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_path_dl = os.path.join(base_dir, 'models', 'e2e_transformer_best.pt')
    if not os.path.exists(model_path_dl):
        print(f"Skipping DL evaluation: checkpoint not found at {model_path_dl}.")
        return None, None

    try:
        state_dict = torch.load(model_path_dl, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        print(f"Skipping DL evaluation: checkpoint is incompatible with {len(classes)} labels ({exc}).")
        return None, None

    model.to(device)
    model.eval()

    test_dataset = MathTextDataset(df['text'].values, y_true_encoded, tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=16)

    final_preds_dl_encoded = []
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)

            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            final_preds_dl_encoded.extend(preds.cpu().numpy())

    y_pred_dl_text = label_encoder.inverse_transform(final_preds_dl_encoded)
    acc_dl = accuracy_score(y_true_text, y_pred_dl_text)
    print(f"DL Model Accuracy: {acc_dl:.4f}")
    print(classification_report(y_true_text, y_pred_dl_text, labels=classes))

    plot_cm(
        y_true_text,
        y_pred_dl_text,
        classes,
        f"Confusion Matrix: E2E DistilBERT (Acc: {acc_dl:.1%})",
        os.path.join(base_dir, 'docs', 'figures', 'gpt_augmented', 'confusion_matrix_dl.png'),
    )
    return y_pred_dl_text, acc_dl

def main():
    df = load_clean_data()
    print(f"Loaded clean dataset: {len(df)} records.")
    
    le = LabelEncoder()
    le.fit(sorted(df['label'].unique()))
    classes = le.classes_
    
    y_true_text = df['label'].values
    y_true_encoded = le.transform(y_true_text)
    
    # ---------------------------------------------------------
    # 1. Evaluate ML Model (HistGB)
    # ---------------------------------------------------------
    print("\n--- Evaluating ML Model (HistGradientBoosting) ---")
    model_path_ml = os.path.join(base_dir, 'models', 'best_classifier_model.pkl')
    with open(model_path_ml, 'rb') as f:
        ml_pipeline = pickle.load(f)
        
    X_test_ml = pd.DataFrame({'text': df['text']})
    y_pred_ml_text = ml_pipeline.predict(X_test_ml)
    
    acc_ml = accuracy_score(y_true_text, y_pred_ml_text)
    print(f"ML Model Accuracy: {acc_ml:.4f}")
    print(classification_report(y_true_text, y_pred_ml_text, labels=classes))
    
    plot_cm(y_true_text, y_pred_ml_text, classes, 
            f"Confusion Matrix: ML Baseline (Acc: {acc_ml:.1%})", 
            os.path.join(base_dir, 'docs', 'figures', 'gpt_augmented', 'confusion_matrix_ml.png'))

    # ---------------------------------------------------------
    # 2. Evaluate DL Model (DistilBERT), if the environment supports it
    # ---------------------------------------------------------
    y_pred_dl_text, _ = evaluate_dl_model(df, classes, y_true_text, y_true_encoded, le)
            
    # ---------------------------------------------------------
    # 3. Save Results
    # ---------------------------------------------------------
    results_df = df[['id', 'label']].copy()
    results_df.rename(columns={'label': 'True_Label'}, inplace=True)
    results_df['ML_Prediction'] = y_pred_ml_text
    if y_pred_dl_text is not None:
        results_df['DL_Prediction'] = y_pred_dl_text
    
    results_path = os.path.join(base_dir, 'results', 'generalization', 'clean_test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nAll predictions successfully saved to: {results_path}")

if __name__ == '__main__':
    main()
