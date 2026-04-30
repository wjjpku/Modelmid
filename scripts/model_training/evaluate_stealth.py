import os
import sys
import json
import pickle
import argparse
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
from train_classifier import TextFeatureExtractor, DenseTransformer

def evaluate_stealth_data(
    dataset_path: str | None = None,
    output_csv: str | None = None,
    include_gpt_field: str | None = None,
    gpt_true_label: str = "GPT-4.1-mini",
    include_default_fields: bool = True,
):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    model_path = os.path.join(base_dir, 'models/best_classifier_model.pkl')
    stealth_data_path = dataset_path or os.path.join(base_dir, 'dataset', 'adversarial', 'stealth_dataset.json')
    
    print("Loading pre-trained classifier...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Loading stealth data...")
    with open(stealth_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        if include_default_fields:
            if row.get('deepseek_stealth'):
                records.append({'id': row['id'], 'text': row['deepseek_stealth'], 'true_label': 'Deepseek'})
            if row.get('kimi_stealth'):
                records.append({'id': row['id'], 'text': row['kimi_stealth'], 'true_label': 'Kimi'})
            if row.get('glm_stealth'):
                records.append({'id': row['id'], 'text': row['glm_stealth'], 'true_label': 'GLM'})
            if row.get('qwen_stealth'):
                records.append({'id': row['id'], 'text': row['qwen_stealth'], 'true_label': 'Qwen'})
        if include_gpt_field and row.get(include_gpt_field):
            records.append({'id': row['id'], 'text': row[include_gpt_field], 'true_label': gpt_true_label})
            
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
    print("COUNTER-INTERVENTION EXPERIMENT RESULTS")
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
    label_order = ['Deepseek', 'Kimi', 'GLM', 'Qwen', gpt_true_label, 'Human']
    present_labels = [label for label in label_order if label in set(y_true) or label in set(y_pred)]
    if 'Human' not in present_labels:
        present_labels.append('Human')
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(
        cm,
        index=present_labels,
        columns=[f'Pred_{label}' for label in present_labels],
    )
    if 'Human' in cm_df.index:
        cm_df = cm_df.drop('Human')
    print(cm_df)

    # Save the predictions to analyze feature changes
    predictions_path = output_csv or os.path.join(base_dir, 'results', 'adversarial', 'stealth_predictions.csv')
    df.to_csv(predictions_path, index=False)
    
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

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate stealth answers against the trained classifier.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to stealth dataset JSON. Defaults to dataset/adversarial/stealth_dataset.json.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for per-sample predictions CSV.",
    )
    parser.add_argument(
        "--include-gpt-field",
        default=None,
        help="Optional dataset field to evaluate as GPT stealth text.",
    )
    parser.add_argument(
        "--gpt-true-label",
        default="GPT-4.1-mini",
        help="True label name to assign to the optional GPT field in the report.",
    )
    parser.add_argument(
        "--skip-default-models",
        action="store_true",
        help="Only evaluate the optional GPT field instead of the original four stealth fields.",
    )
    args = parser.parse_args()
    evaluate_stealth_data(
        dataset_path=args.dataset,
        output_csv=args.output_csv,
        include_gpt_field=args.include_gpt_field,
        gpt_true_label=args.gpt_true_label,
        include_default_fields=not args.skip_default_models,
    )

if __name__ == '__main__':
    main()
