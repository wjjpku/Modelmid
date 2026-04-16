import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import TextFeatureExtractor

def analyze_glm_collapse():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    predictions_path = os.path.join(base_dir, 'dataset/generalization_predictions.csv')
    
    print("Loading Generalization Predictions...")
    df = pd.read_csv(predictions_path)
    
    # Isolate specific subsets
    glm_true = df[df['true_label'] == 'GLM']
    glm_pred_kimi = glm_true[glm_true['predicted_label'] == 'Kimi']
    glm_pred_glm = glm_true[glm_true['predicted_label'] == 'GLM']
    kimi_true_kimi_pred = df[(df['true_label'] == 'Kimi') & (df['predicted_label'] == 'Kimi')]
    
    print(f"Total GLM samples: {len(glm_true)}")
    print(f"GLM incorrectly predicted as Kimi: {len(glm_pred_kimi)}")
    print(f"GLM correctly predicted as GLM: {len(glm_pred_glm)}")
    print(f"Kimi correctly predicted as Kimi: {len(kimi_true_kimi_pred)}")
    
    print("\nExtracting structural features for analysis...")
    extractor = TextFeatureExtractor()
    
    features_glm_pred_kimi = extractor.transform(glm_pred_kimi['text'])
    features_glm_pred_glm = extractor.transform(glm_pred_glm['text'])
    features_kimi_true = extractor.transform(kimi_true_kimi_pred['text'])
    
    # Calculate means
    mean_glm_pred_kimi = features_glm_pred_kimi.mean()
    mean_glm_pred_glm = features_glm_pred_glm.mean()
    mean_kimi_true = features_kimi_true.mean()
    
    print("\n--- Feature Comparison (Mean Values) ---")
    features_to_compare = [
        'num_paragraphs', 'avg_paragraph_length', 'num_lines', 
        'inline_math_count', 'display_math_count', 'math_density',
        'declarative_density', 'transition_words_density', 'num_list_items'
    ]
    
    comparison_df = pd.DataFrame({
        'GLM (Misclassified as Kimi)': mean_glm_pred_kimi[features_to_compare],
        'GLM (Correctly Classified)': mean_glm_pred_glm[features_to_compare],
        'Kimi (True Baseline)': mean_kimi_true[features_to_compare]
    }).round(2)
    
    print(comparison_df)
    
    # Export case study texts
    print("\n--- Exporting Case Study Text Sample ---")
    if len(glm_pred_kimi) > 0:
        sample_row = glm_pred_kimi.iloc[0]
        print(f"\nSubject: {sample_row['subject']}")
        print(f"Problem ID: {sample_row['id']}")
        print(f"\n--- GLM Text that looked like Kimi ---\n{sample_row['text'][:1000]}...\n")
        
    if len(kimi_true_kimi_pred) > 0:
        # Find a kimi sample for the same subject if possible
        kimi_samples = kimi_true_kimi_pred[kimi_true_kimi_pred['subject'] == sample_row['subject']]
        if len(kimi_samples) > 0:
            kimi_sample_row = kimi_samples.iloc[0]
            print(f"\n--- True Kimi Text for Comparison ---\n{kimi_sample_row['text'][:1000]}...\n")

if __name__ == '__main__':
    analyze_glm_collapse()