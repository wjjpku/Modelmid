import os
import sys
import pandas as pd
import numpy as np
import time

# Append model_training to path to reuse existing load_data and feature extractor
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import load_data, TextFeatureExtractor

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Import different Classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Using HistGradientBoostingClassifier as a fast alternative to LightGBM/XGBoost built into sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

def run_classifier_comparison():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    
    print("Loading data...")
    df = load_data(dataset_path)
    X = df['text']
    y = df['label']
    X_df = pd.DataFrame({'text': X})
    print(f"Dataset loaded: {len(df)} samples.")
    
    # 1. Define the identical feature extraction pipeline
    print("Building Feature Pipeline (TF-IDF + 28 Custom Structural Features)...")
    combined_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                  token_pattern=r'(?u)\b\w+\b|\\[a-zA-Z]+',
                                  stop_words=None), 'text'),
        ('custom', Pipeline([
            ('extractor', TextFeatureExtractor()),
            ('scaler', StandardScaler())
        ]), 'text')
    ])
    
    # Create a wrapper for classifiers that require dense input
    class DenseTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            return X.toarray() if hasattr(X, "toarray") else X

    # 2. Define the classifiers to test
    classifiers = {
        'Linear SVM (Baseline)': Pipeline([
            ('features', combined_features),
            ('clf', SVC(kernel='linear', probability=True, random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('features', combined_features),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
        ]),
        'Random Forest': Pipeline([
            ('features', combined_features),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        'Hist Gradient Boosting': Pipeline([
            ('features', combined_features),
            ('to_dense', DenseTransformer()),
            ('clf', HistGradientBoostingClassifier(max_iter=100, random_state=42))
        ]),
        'XGBoost (via sklearn GB)': Pipeline([
            ('features', combined_features),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nStarting 5-Fold Cross Validation for each classifier...")
    print("-" * 60)
    
    results = {}
    
    for name, pipeline in classifiers.items():
        print(f"Evaluating {name}...")
        start_time = time.time()
        
        # Run CV
        try:
            scores = cross_val_score(pipeline, X_df, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_acc = scores.mean()
            std_acc = scores.std()
            elapsed_time = time.time() - start_time
            
            results[name] = {
                'Accuracy': mean_acc,
                'Std': std_acc,
                'Time (s)': elapsed_time
            }
            
            print(f"  => Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}) | Time: {elapsed_time:.1f}s")
        except Exception as e:
            print(f"  => Failed to evaluate {name}. Error: {e}")
            
    print("-" * 60)
    print("🏆 FINAL RANKING 🏆")
    # Sort results by Accuracy descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:25s} | Acc: {metrics['Accuracy']:.4f} | Time: {metrics['Time (s)']:.1f}s")

if __name__ == '__main__':
    run_classifier_comparison()