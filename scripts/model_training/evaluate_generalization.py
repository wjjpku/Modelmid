import os
import sys
import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import TextFeatureExtractor, DenseTransformer

def evaluate_generalization():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')'
    model_path = os.path.join(base_dir, 'models/best_classifier_model.pkl')
    data_path = os.path.join(base_dir, 'dataset/generalization_dataset.json')
    
    print("Loading pre-trained best classifier (Hist Gradient Boosting)...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Loading generalization dataset (MATH dataset)...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        subject = row['subject']
        if row.get('human') and str(row['human']).strip():
            records.append({'id': row['id'], 'subject': subject, 'text': row['human'], 'true_label': 'Human'})
        if row.get('deepseek') and str(row['deepseek']).strip():
            records.append({'id': row['id'], 'subject': subject, 'text': row['deepseek'], 'true_label': 'Deepseek'})
        if row.get('kimi') and str(row['kimi']).strip():
            records.append({'id': row['id'], 'subject': subject, 'text': row['kimi'], 'true_label': 'Kimi'})
        if row.get('glm') and str(row['glm']).strip():
            records.append({'id': row['id'], 'subject': subject, 'text': row['glm'], 'true_label': 'GLM'})
        if row.get('qwen') and str(row['qwen']).strip():
            records.append({'id': row['id'], 'subject': subject, 'text': row['qwen'], 'true_label': 'Qwen'})
            
    df = pd.DataFrame(records)
    print(f"Total out-of-domain records to evaluate: {len(df)}")
    
    X = pd.DataFrame({'text': df['text']})
    y_true = df['true_label']
    
    print("\nPredicting on Out-Of-Domain Data...")
    y_pred = model.predict(X)
    df['predicted_label'] = y_pred
    
    overall_acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print("🌍 GENERALIZATION EXPERIMENT RESULTS 🌍")
    print("="*50)
    print(f"Overall Cross-Domain Accuracy: {overall_acc * 100:.2f}%")
    
    print("\n--- Accuracy Breakdown by True Source ---")
    grouped_source = df.groupby('true_label')
    for name, group in grouped_source:
        acc = accuracy_score(group['true_label'], group['predicted_label'])
        print(f"{name:10s}: {acc * 100:.2f}% ({len(group)} samples)")
        
    print("\n--- Accuracy Breakdown by Subject ---")
    grouped_subject = df.groupby('subject')
    for name, group in grouped_subject:
        acc = accuracy_score(group['true_label'], group['predicted_label'])
        print(f"{name:15s}: {acc * 100:.2f}% ({len(group)} samples)")
        
    print("\n--- Detailed Classification Matrix (Overall) ---")
    labels = ['Human', 'Deepseek', 'Kimi', 'GLM', 'Qwen']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=['Pred_' + l for l in labels])
    print(cm_df)
    
    # Save the results for reporting
    df.to_csv(os.path.join(base_dir, 'dataset/generalization_predictions.csv'), index=False)

if __name__ == '__main__':
    evaluate_generalization()