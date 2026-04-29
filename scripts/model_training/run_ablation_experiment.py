import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# Add parent directory to path to import from train_classifier
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_classifier import TextFeatureExtractor, DenseTransformer, load_data

def run_ablation_study():
    """
    Runs an ablation study to understand the contribution of different feature groups 
    to the classification performance.
    """
    print("="*50)
    print("🧪 RUNNING FEATURE ABLATION EXPERIMENT 🧪")
    print("="*50)
    
    # 1. Load Data
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    data_path = os.path.join(base_dir, 'dataset', 'training', 'full_dataset.json')
    
    print("Loading dataset...")
    df = load_data(data_path)
    X = pd.DataFrame({'text': df['text']})
    y = df['label']
    
    # We will use 5-fold cross validation for stable evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    
    results = []

    # ---------------------------------------------------------
    # Group 1: Full Model (TF-IDF + All Custom Features)
    # ---------------------------------------------------------
    print("\n[1/5] Evaluating Full Model (All Features)...")
    full_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
        ('custom', Pipeline([('extractor', TextFeatureExtractor()), ('scaler', StandardScaler())]), 'text')
    ])
    full_pipeline = Pipeline([('features', full_features), ('to_dense', DenseTransformer()), ('clf', base_clf)])
    full_scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Full Model Accuracy: {full_scores.mean():.4f} (±{full_scores.std():.4f})")
    results.append({"Configuration": "Full Model (TF-IDF + Custom)", "Accuracy": full_scores.mean(), "Std": full_scores.std()})

    # ---------------------------------------------------------
    # Group 2: Only TF-IDF (No Handcrafted Features)
    # ---------------------------------------------------------
    print("\n[2/5] Evaluating TF-IDF Only (Ablating Custom Features)...")
    tfidf_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text')
    ])
    tfidf_pipeline = Pipeline([('features', tfidf_features), ('to_dense', DenseTransformer()), ('clf', base_clf)])
    tfidf_scores = cross_val_score(tfidf_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"TF-IDF Only Accuracy: {tfidf_scores.mean():.4f} (±{tfidf_scores.std():.4f})")
    results.append({"Configuration": "TF-IDF Only", "Accuracy": tfidf_scores.mean(), "Std": tfidf_scores.std()})

    # ---------------------------------------------------------
    # Group 3: Only Custom Features (No TF-IDF)
    # ---------------------------------------------------------
    print("\n[3/5] Evaluating Custom Features Only (Ablating TF-IDF)...")
    custom_only_features = ColumnTransformer([
        ('custom', Pipeline([('extractor', TextFeatureExtractor()), ('scaler', StandardScaler())]), 'text')
    ])
    custom_pipeline = Pipeline([('features', custom_only_features), ('to_dense', DenseTransformer()), ('clf', base_clf)])
    custom_scores = cross_val_score(custom_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Custom Features Only Accuracy: {custom_scores.mean():.4f} (±{custom_scores.std():.4f})")
    results.append({"Configuration": "Custom Features Only", "Accuracy": custom_scores.mean(), "Std": custom_scores.std()})

    # ---------------------------------------------------------
    # Now let's do fine-grained ablation on the custom features.
    # We will create modified extractors.
    # ---------------------------------------------------------
    
    class MacroStructureExtractor(TextFeatureExtractor):
        def transform(self, X, y=None):
            df = super().transform(X, y)
            # Keep only macro structure features
            cols = ['length', 'num_lines', 'avg_line_length', 'num_paragraphs', 'avg_paragraph_length']
            return df[cols]

    class MathLatexExtractor(TextFeatureExtractor):
        def transform(self, X, y=None):
            df = super().transform(X, y)
            # Keep only math and latex features
            cols = ['inline_math_count', 'display_math_count', 'math_density', 'num_frac', 'num_textbf', 
                    'latex_env_count', 'left_right_brackets', 'math_cal_bb', 'implication_arrows', 'qed_symbols']
            return df[cols]

    class SemanticLexicalExtractor(TextFeatureExtractor):
        def transform(self, X, y=None):
            df = super().transform(X, y)
            # Keep only semantic/logical word features
            cols = ['logical_words_count', 'logical_words_density', 'declarative_density', 
                    'transition_words_density', 'conclusion_words_density', 'num_list_items',
                    'vocab_richness', 'avg_word_length']
            return df[cols]

    # ---------------------------------------------------------
    # Group 4: Ablating Specific Custom Feature Sub-groups
    # ---------------------------------------------------------
    
    # 4a. TF-IDF + Math/LaTeX + Semantic (Ablating Macro Structure)
    print("\n[4/5] Evaluating TF-IDF + Math + Semantic (Ablating Macro Structure)...")
    no_macro_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
        ('math', Pipeline([('extractor', MathLatexExtractor()), ('scaler', StandardScaler())]), 'text'),
        ('semantic', Pipeline([('extractor', SemanticLexicalExtractor()), ('scaler', StandardScaler())]), 'text')
    ])
    no_macro_pipeline = Pipeline([('features', no_macro_features), ('to_dense', DenseTransformer()), ('clf', base_clf)])
    no_macro_scores = cross_val_score(no_macro_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results.append({"Configuration": "Full w/o Macro Structure", "Accuracy": no_macro_scores.mean(), "Std": no_macro_scores.std()})

    # 4b. TF-IDF + Macro + Semantic (Ablating Math/LaTeX)
    print("\n[5/5] Evaluating TF-IDF + Macro + Semantic (Ablating Math/LaTeX)...")
    no_math_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
        ('macro', Pipeline([('extractor', MacroStructureExtractor()), ('scaler', StandardScaler())]), 'text'),
        ('semantic', Pipeline([('extractor', SemanticLexicalExtractor()), ('scaler', StandardScaler())]), 'text')
    ])
    no_math_pipeline = Pipeline([('features', no_math_features), ('to_dense', DenseTransformer()), ('clf', base_clf)])
    no_math_scores = cross_val_score(no_math_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results.append({"Configuration": "Full w/o Math/LaTeX", "Accuracy": no_math_scores.mean(), "Std": no_math_scores.std()})

    # ---------------------------------------------------------
    # Save and Print Summary
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 ABLATION EXPERIMENT RESULTS SUMMARY 📊")
    print("="*50)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False)
    
    print(results_df.to_string(index=False))
    
    # Calculate drops
    full_acc = results_df[results_df["Configuration"] == "Full Model (TF-IDF + Custom)"]["Accuracy"].values[0]
    
    print("\n📉 Performance Drops (Importance of Feature Groups):")
    for _, row in results_df.iterrows():
        if row["Configuration"] != "Full Model (TF-IDF + Custom)":
            drop = full_acc - row["Accuracy"]
            print(f"- Removing [{row['Configuration']}]: Drop of {drop*100:.2f}%")
            
    # Save to CSV
    output_dir = os.path.join(base_dir, 'results', 'classification')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    run_ablation_study()
