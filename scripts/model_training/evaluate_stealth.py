import os
import sys
import json
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import TextFeatureExtractor, DenseTransformer

def evaluate_stealth_data():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    model_path = os.path.join(base_dir, 'models/best_classifier_model.pkl')
    stealth_data_path = os.path.join(base_dir, 'dataset/stealth_dataset.json')
    
    print("Loading pre-trained classifier...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Loading stealth data...")
    with open(stealth_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        if row.get('deepseek_stealth'):
            records.append({'id': row['id'], 'text': row['deepseek_stealth'], 'true_label': 'Deepseek'})
        if row.get('kimi_stealth'):
            records.append({'id': row['id'], 'text': row['kimi_stealth'], 'true_label': 'Kimi'})
        if row.get('glm_stealth'):
            records.append({'id': row['id'], 'text': row['glm_stealth'], 'true_label': 'GLM'})
        if row.get('qwen_stealth'):
            records.append({'id': row['id'], 'text': row['qwen_stealth'], 'true_label': 'Qwen'})
            
    df = pd.DataFrame(records)
    print(f"Total stealth AI records to evaluate: {len(df)}")
    
    X = pd.DataFrame({'text': df['text']})
    y_true = df['true_label']
    
    print("\nPredicting with our Anti-AI Classifier...")
    y_pred = model.predict(X)
    
    # Analyze how many were misclassified as 'Human'
    df['predicted_label'] = y_pred
    
    human_predictions = df[df['predicted_label'] == 'Human']
    total_ai = len(df)
    fooled_count = len(human_predictions)
    
    print("\n" + "="*50)
    print("🚨 COUNTER-INTERVENTION EXPERIMENT RESULTS 🚨")
    print("="*50)
    print(f"Total AI texts generated with stealth prompts: {total_ai}")
    print(f"Number of texts successfully classified as AI (any model): {total_ai - fooled_count}")
    print(f"Number of AI texts that FOOLED the classifier into thinking they are HUMAN: {fooled_count}")
    print(f"Stealth Success Rate (Misclassified as Human): {fooled_count / total_ai * 100:.2f}%")
    
    print("\n--- Breakdown by True Model ---")
    grouped = df.groupby('true_label')
    for name, group in grouped:
        model_total = len(group)
        model_fooled = len(group[group['predicted_label'] == 'Human'])
        print(f"{name:10s}: {model_fooled}/{model_total} fooled the classifier ({model_fooled/model_total*100:.2f}%)")
        
    print("\n--- Detailed Classification Matrix ---")
    print("Rows: True Source | Columns: Predicted Source")
    cm = confusion_matrix(y_true, y_pred, labels=['Deepseek', 'Kimi', 'GLM', 'Qwen', 'Human'])
    cm_df = pd.DataFrame(cm, index=['Deepseek', 'Kimi', 'GLM', 'Qwen', 'Human'], columns=['Pred_Deepseek', 'Pred_Kimi', 'Pred_GLM', 'Pred_Qwen', 'Pred_Human'])
    # Drop the Human row since we didn't input true Human text
    cm_df = cm_df.drop('Human')
    print(cm_df)
    
    # Save the predictions to analyze feature changes
    df.to_csv(os.path.join(base_dir, 'dataset/stealth_predictions.csv'), index=False)
    
    # Let's extract features of the stealth text to see what changed
    print("\n--- Analyzing Feature Shifts (Why were they fooled?) ---")
    extractor = TextFeatureExtractor()
    features_df = extractor.transform(X['text'])
    features_df['true_label'] = y_true.values
    features_df['predicted_label'] = y_pred
    
    fooled_features = features_df[features_df['predicted_label'] == 'Human']
    detected_features = features_df[features_df['predicted_label'] != 'Human']
    
    if len(fooled_features) > 0:
        print("Average Features of AI texts that FOOLED the classifier:")
        print(f"  Paragraphs: {fooled_features['num_paragraphs'].mean():.2f}")
        print(f"  Declarative Density: {fooled_features['declarative_density'].mean():.2f}")
        print(f"  Inline Math: {fooled_features['inline_math_count'].mean():.2f}")
        
    if len(detected_features) > 0:
        print("\nAverage Features of AI texts that were DETECTED as AI:")
        print(f"  Paragraphs: {detected_features['num_paragraphs'].mean():.2f}")
        print(f"  Declarative Density: {detected_features['declarative_density'].mean():.2f}")
        print(f"  Inline Math: {detected_features['inline_math_count'].mean():.2f}")

if __name__ == '__main__':
    evaluate_stealth_data()