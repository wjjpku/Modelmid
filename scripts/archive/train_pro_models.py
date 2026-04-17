import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_classifier import TextFeatureExtractor, DenseTransformer

def load_pro_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    # Each item represents a question
    for row in data:
        q_id = row['id']
        subject = row.get('subject', 'unknown')
        
        # Add human
        if row.get('human') and str(row['human']).strip():
            records.append({'id': q_id, 'subject': subject, 'text': str(row['human']).strip(), 'label': 'Human'})
        # Add deepseek
        if row.get('deepseek') and str(row['deepseek']).strip():
            records.append({'id': q_id, 'subject': subject, 'text': str(row['deepseek']).strip(), 'label': 'Deepseek'})
        # Add kimi 
        if row.get('kimi') and str(row['kimi']).strip():
            records.append({'id': q_id, 'subject': subject, 'text': str(row['kimi']).strip(), 'label': 'Kimi'})
        # Add GLM
        if row.get('glm') and str(row['glm']).strip():
            records.append({'id': q_id, 'subject': subject, 'text': str(row['glm']).strip(), 'label': 'GLM'})
        # Add Qwen
        if row.get('qwen') and str(row['qwen']).strip():
            records.append({'id': q_id, 'subject': subject, 'text': str(row['qwen']).strip(), 'label': 'Qwen'})
            
    return pd.DataFrame(records), data

# Define Deep Learning Model (PyTorch)
class TextDLModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextDLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_pytorch_model(X_train, y_train, X_test, y_test, label_encoder, input_dim, epochs=30, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training Deep Learning model on device: {device}")
    
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    if hasattr(X_train, "toarray"): X_train = X_train.toarray()
    if hasattr(X_test, "toarray"): X_test = X_test.toarray()
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train_enc).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test_enc).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TextDLModel(input_dim, len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0
    best_y_pred = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            acc = correct / len(y_test_tensor)
            
        scheduler.step(acc)
        
        if (epoch+1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")
            
        if acc > best_acc:
            best_acc = acc
            best_y_pred = predicted.cpu().numpy()
            
    y_pred_labels = label_encoder.inverse_transform(best_y_pred)
    return y_pred_labels, best_acc

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    json_path = os.path.join(base_dir, 'dataset', 'full_dataset_pro.json')
    
    if not os.path.exists(json_path):
        print(f"Dataset {json_path} not found. Please run generation script first.")
        return
        
    print(f"Loading extended dataset from {json_path}...")
    df, raw_data = load_pro_data(json_path)
    
    print(f"Total questions in dataset: {len(raw_data)}")
    print(f"Total records (question-source pairs): {len(df)}")
    
    # 1. Stratified split at QUESTION level
    # Create a dataframe of unique questions and their subjects
    q_df = pd.DataFrame([{'id': q['id'], 'subject': q.get('subject', 'unknown')} for q in raw_data])
    
    # We want exactly 1800 for training, and the rest (should be ~200) for testing
    train_size = min(1800, len(q_df) - 1)
    if len(q_df) < 1800:
        print(f"Warning: Dataset only has {len(q_df)} questions. Using 90% for training.")
        train_q, test_q = train_test_split(q_df, test_size=0.1, stratify=q_df['subject'], random_state=42)
    else:
        train_q, test_q = train_test_split(q_df, train_size=1800, stratify=q_df['subject'], random_state=42)
        
    train_ids = set(train_q['id'])
    test_ids = set(test_q['id'])
    
    train_df = df[df['id'].isin(train_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()
    
    print(f"\nQuestion Split: {len(train_q)} Train / {len(test_q)} Test")
    print(f"Record Split:   {len(train_df)} Train / {len(test_df)} Test")
    
    X_train_text = train_df['text']
    y_train = train_df['label']
    X_test_text = test_df['text']
    y_test = test_df['label']
    
    # 2. Feature Extraction Pipeline
    print("\nExtracting features (TF-IDF + Custom)...")
    feature_extractor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
        ('custom', Pipeline([
            ('extractor', TextFeatureExtractor()),
            ('scaler', StandardScaler())
        ]), 'text')
    ])
    
    X_train_df = pd.DataFrame({'text': X_train_text})
    X_test_df = pd.DataFrame({'text': X_test_text})
    
    t0 = time.time()
    X_train_feat = feature_extractor.fit_transform(X_train_df)
    X_test_feat = feature_extractor.transform(X_test_df)
    
    if hasattr(X_train_feat, "toarray"): X_train_feat = X_train_feat.toarray()
    if hasattr(X_test_feat, "toarray"): X_test_feat = X_test_feat.toarray()
    
    print(f"Feature extraction done in {time.time()-t0:.2f}s. Feature dimension: {X_train_feat.shape[1]}")
    
    # 3. Train Machine Learning Model (HistGradientBoosting)
    print("\n" + "="*50)
    print("🚀 TRAINING MACHINE LEARNING MODEL (HistGradientBoosting)")
    print("="*50)
    
    ml_clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    
    t0 = time.time()
    ml_clf.fit(X_train_feat, y_train)
    ml_time = time.time() - t0
    
    y_pred_ml = ml_clf.predict(X_test_feat)
    ml_acc = accuracy_score(y_test, y_pred_ml)
    print(f"ML Model trained in {ml_time:.2f}s")
    print(f"ML Model Accuracy: {ml_acc:.4f}")
    print("\nML Classification Report:")
    print(classification_report(y_test, y_pred_ml))
    
    # 4. Train Deep Learning Model (PyTorch DNN)
    print("\n" + "="*50)
    print("🧠 TRAINING DEEP LEARNING MODEL (PyTorch DNN)")
    print("="*50)
    
    le = LabelEncoder()
    le.fit(y_train)
    
    t0 = time.time()
    y_pred_dl, dl_acc = train_pytorch_model(X_train_feat, y_train, X_test_feat, y_test, le, input_dim=X_train_feat.shape[1])
    dl_time = time.time() - t0
    
    print(f"\nDL Model trained in {dl_time:.2f}s")
    print(f"DL Model Accuracy: {dl_acc:.4f}")
    print("\nDL Classification Report:")
    print(classification_report(y_test, y_pred_dl))
    
    # 5. Summary
    print("\n" + "="*50)
    print("📊 COMPARISON SUMMARY 📊")
    print("="*50)
    print(f"Machine Learning (HistGradientBoosting): {ml_acc:.4f} (Time: {ml_time:.2f}s)")
    print(f"Deep Learning    (PyTorch DNN):          {dl_acc:.4f} (Time: {dl_time:.2f}s)")
    
    # Check general vs specific subject performance
    test_df['ml_pred'] = y_pred_ml
    test_df['dl_pred'] = y_pred_dl
    
    print("\n--- Accuracy by Subject ---")
    for subject in test_df['subject'].unique():
        sub_df = test_df[test_df['subject'] == subject]
        sub_ml_acc = accuracy_score(sub_df['label'], sub_df['ml_pred'])
        sub_dl_acc = accuracy_score(sub_df['label'], sub_df['dl_pred'])
        print(f"[{subject:25s}] ML: {sub_ml_acc:.4f} | DL: {sub_dl_acc:.4f} (N={len(sub_df)})")
        
    # Save the models
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    import joblib
    joblib.dump(ml_clf, os.path.join(base_dir, 'models', 'pro_ml_model.pkl'))
    joblib.dump(feature_extractor, os.path.join(base_dir, 'models', 'pro_feature_extractor.pkl'))
    print("\nModels saved to models/ directory.")

if __name__ == '__main__':
    main()
