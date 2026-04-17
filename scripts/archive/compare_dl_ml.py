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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import copy

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_classifier import TextFeatureExtractor, DenseTransformer
from train_pro_models import load_pro_data

# ==========================================
# DEEP LEARNING ARCHITECTURES
# ==========================================

# 1. Simple MLP (Multi-Layer Perceptron)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Deep ResNet-like DNN (with skip connections and BatchNorm)
class ResNetDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 512)
        
        # Block 1
        self.b1_fc1 = nn.Linear(512, 512)
        self.b1_bn1 = nn.BatchNorm1d(512)
        self.b1_fc2 = nn.Linear(512, 512)
        self.b1_bn2 = nn.BatchNorm1d(512)
        
        # Downsample
        self.fc_mid = nn.Linear(512, 256)
        
        # Block 2
        self.b2_fc1 = nn.Linear(256, 256)
        self.b2_bn1 = nn.BatchNorm1d(256)
        self.b2_fc2 = nn.Linear(256, 256)
        self.b2_bn2 = nn.BatchNorm1d(256)
        
        self.fc_out = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc_in(x))
        
        # Block 1
        identity = x
        out = self.relu(self.b1_bn1(self.b1_fc1(x)))
        out = self.dropout(out)
        out = self.b1_bn2(self.b1_fc2(out))
        x = self.relu(out + identity)
        
        x = self.relu(self.fc_mid(x))
        
        # Block 2
        identity = x
        out = self.relu(self.b2_bn1(self.b2_fc1(x)))
        out = self.dropout(out)
        out = self.b2_bn2(self.b2_fc2(out))
        x = self.relu(out + identity)
        
        x = self.dropout(x)
        return self.fc_out(x)

# 3. 1D-CNN (Treating features as a sequence)
class Conv1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
        # Calculate output size after 2 poolings: input_dim // 4
        fc_in_dim = 64 * (input_dim // 4)
        
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Reshape to (batch_size, channels, seq_len)
        x = x.unsqueeze(1) 
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

# ==========================================
# TRAINING LOOP WITH EARLY STOPPING
# ==========================================
def train_dl_model(model_name, model_class, X_train, y_train, X_test, y_test, label_encoder, input_dim, max_epochs=150, batch_size=128, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Training {model_name} on {device} ---")
    
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train_enc).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test_enc).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = model_class(input_dim, len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    best_acc = 0
    best_model_weights = None
    epochs_no_improve = 0
    best_epoch = 0
    
    t0 = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            acc = correct / len(y_test_tensor)
            
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with Acc: {best_acc:.4f}")
            break
            
    train_time = time.time() - t0
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = label_encoder.inverse_transform(predicted.cpu().numpy())
        
    return y_pred, best_acc, train_time, best_epoch + 1

# ==========================================
# MAIN EXPERIMENT
# ==========================================
def run_comprehensive_comparison():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    json_path = os.path.join(base_dir, 'dataset', 'full_dataset_pro.json')
    
    print("Loading Pro Dataset...")
    df, raw_data = load_pro_data(json_path)
    
    # Stratified Split (1800/200)
    q_df = pd.DataFrame([{'id': q['id'], 'subject': q.get('subject', 'unknown')} for q in raw_data])
    train_size = min(1800, len(q_df) - 1)
    train_q, test_q = train_test_split(q_df, train_size=train_size, stratify=q_df['subject'], random_state=42)
    
    train_df = df[df['id'].isin(set(train_q['id']))].copy()
    test_df = df[df['id'].isin(set(test_q['id']))].copy()
    
    X_train_text = train_df['text']
    y_train = train_df['label']
    X_test_text = test_df['text']
    y_test = test_df['label']
    
    # Extract Features
    print("Extracting Features (TF-IDF + Custom)...")
    feature_extractor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
        ('custom', Pipeline([('extractor', TextFeatureExtractor()), ('scaler', StandardScaler())]), 'text')
    ])
    
    X_train_feat = feature_extractor.fit_transform(pd.DataFrame({'text': X_train_text}))
    X_test_feat = feature_extractor.transform(pd.DataFrame({'text': X_test_text}))
    
    if hasattr(X_train_feat, "toarray"): X_train_feat = X_train_feat.toarray()
    if hasattr(X_test_feat, "toarray"): X_test_feat = X_test_feat.toarray()
    
    input_dim = X_train_feat.shape[1]
    le = LabelEncoder()
    le.fit(y_train)
    
    results = []
    
    # --- 1. Traditional ML Models ---
    print("\n" + "="*50)
    print("🚀 EVALUATING MACHINE LEARNING MODELS")
    print("="*50)
    
    ml_models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)
    }
    
    for name, clf in ml_models.items():
        print(f"Training {name}...")
        t0 = time.time()
        clf.fit(X_train_feat, y_train)
        t_time = time.time() - t0
        
        y_pred = clf.predict(X_test_feat)
        acc = accuracy_score(y_test, y_pred)
        print(f"-> Acc: {acc:.4f} | Time: {t_time:.2f}s")
        
        results.append({
            "Model": name,
            "Type": "ML",
            "Accuracy": acc,
            "Time(s)": t_time,
            "Epochs_to_Converge": "N/A"
        })
        
    # --- 2. Deep Learning Models ---
    print("\n" + "="*50)
    print("🧠 EVALUATING DEEP LEARNING MODELS (with Early Stopping)")
    print("="*50)
    
    dl_models = {
        "Simple_MLP": SimpleMLP,
        "ResNet_DNN": ResNetDNN,
        "Conv1D_Net": Conv1DNet
    }
    
    for name, m_class in dl_models.items():
        y_pred, acc, t_time, epochs = train_dl_model(
            name, m_class, X_train_feat, y_train, X_test_feat, y_test, 
            le, input_dim, max_epochs=200, patience=20
        )
        print(f"-> Acc: {acc:.4f} | Time: {t_time:.2f}s | Converged at epoch: {epochs}")
        
        results.append({
            "Model": name,
            "Type": "DL",
            "Accuracy": acc,
            "Time(s)": t_time,
            "Epochs_to_Converge": epochs
        })
        
    # --- Summary ---
    print("\n" + "="*50)
    print("🏆 FINAL COMPARISON REPORT 🏆")
    print("="*50)
    
    res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    
    # Save results
    output_dir = os.path.join(base_dir, 'docs')
    res_df.to_csv(os.path.join(output_dir, 'ml_vs_dl_comparison.csv'), index=False)

if __name__ == '__main__':
    run_comprehensive_comparison()