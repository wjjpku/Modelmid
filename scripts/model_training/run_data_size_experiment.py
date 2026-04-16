import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier

# Import custom feature extractor
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import load_data, TextFeatureExtractor

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, "toarray") else X

def get_pipeline(clf, use_dense_transformer=False):
    combined_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=800, ngram_range=(1, 2), # Slightly reduced max_features to save memory
                                  token_pattern=r'(?u)\b\w+\b|\\[a-zA-Z]+',
                                  stop_words=None), 'text'),
        ('custom', Pipeline([
            ('extractor', TextFeatureExtractor()),
            ('scaler', StandardScaler())
        ]), 'text')
    ])
    
    if use_dense_transformer:
        return Pipeline([
            ('features', combined_features),
            ('to_dense', DenseTransformer()),
            ('clf', clf)
        ])
    else:
        return Pipeline([
            ('features', combined_features),
            ('clf', clf)
        ])

def run_data_size_experiment():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    output_dir = os.path.join(base_dir, 'latex_report/figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading and splitting dataset...")
    df = load_data(dataset_path)
    
    # User requested: Training sizes: 100, 400, 800 *questions*. 
    # Since our dataset expands 1 question to 5 records (Human, Deepseek, Kimi, GLM, Qwen),
    # "200 questions as test set" means 1000 records.
    # "800 questions as max training set" means 4000 records.
    # Total dataset is exactly 1000 questions (5000 records).
    
    # We need to split by 'id' to ensure no data leakage across questions
    unique_ids = df['id'].unique()
    
    # Reserve 200 questions for testing
    train_ids, test_ids = train_test_split(unique_ids, test_size=200, random_state=42)
    
    test_df = df[df['id'].isin(test_ids)]
    X_test = pd.DataFrame({'text': test_df['text']})
    y_test = test_df['label']
    
    print(f"Test Set: 200 questions ({len(test_df)} records)")
    
    train_sizes_questions = [20, 50, 100, 400, 800]
    
    classifiers = {
        'Linear SVM': (SVC(kernel='linear', probability=False, random_state=42), False),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
        'Hist Gradient Boosting': (HistGradientBoostingClassifier(max_iter=100, random_state=42), True),
        'XGBoost (sklearn GB)': (GradientBoostingClassifier(n_estimators=100, random_state=42), False)
    }
    
    results = []
    
    for size_q in train_sizes_questions:
        print(f"\n--- Running Experiment for Training Size: {size_q} questions ({size_q * 5} records) ---")
        
        # Subsample training IDs
        current_train_ids = train_ids[:size_q]
        current_train_df = df[df['id'].isin(current_train_ids)]
        
        X_train = pd.DataFrame({'text': current_train_df['text']})
        y_train = current_train_df['label']
        
        for clf_name, (clf, needs_dense) in classifiers.items():
            print(f"  Training {clf_name}...")
            pipeline = get_pipeline(clf, needs_dense)
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"    -> Accuracy: {acc:.4f}")
            results.append({
                'Model': clf_name,
                'Training Questions': size_q,
                'Accuracy': acc
            })
            
            # Explicit garbage collection to manage memory
            del pipeline
            del y_pred
            gc.collect()
            
    # Save results to DataFrame
    res_df = pd.DataFrame(results)
    
    # 1. Pivot table for LaTeX
    pivot_table = res_df.pivot(index='Model', columns='Training Questions', values='Accuracy')
    pivot_table.to_csv(os.path.join(base_dir, 'latex_report/results_table.csv'))
    print("\nFinal Results Table:")
    print(pivot_table)
    
    # 2. Plot Learning Curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x='Training Questions', y='Accuracy', hue='Model', marker='o', linewidth=2, markersize=8)
    plt.title('Model Accuracy vs Training Set Size (200 Test Questions)', fontsize=16)
    plt.xlabel('Number of Training Questions', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.xticks(train_sizes_questions)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)
    plt.close()
    print(f"Saved learning curves plot to {output_dir}/learning_curves.png")

if __name__ == '__main__':
    run_data_size_experiment()