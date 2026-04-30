import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plotting_utils import configure_matplotlib_fonts

configure_matplotlib_fonts()

def plot_stealth_results():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    output_path = os.path.join(base_dir, 'docs', 'figures', 'gpt_augmented', 'stealth_success_rate.png')
    predictions_path = os.path.join(base_dir, 'results', 'adversarial', 'stealth_predictions.csv')

    df = pd.read_csv(predictions_path)
    grouped = (
        df.assign(fooled=df['predicted_label'].eq('Human'))
        .groupby('true_label', sort=False)['fooled']
        .mean()
        .mul(100)
    )

    model_order = ['Deepseek', 'Kimi', 'GLM', 'Qwen', 'GPT-4.1-mini']
    models = [model for model in model_order if model in grouped.index]
    success_rates = [grouped.loc[model] for model in models]
    
    plt.figure(figsize=(10, 6))
    
    color_map = {
        'Deepseek': '#1f77b4',
        'Kimi': '#ff7f0e',
        'GLM': '#d62728',
        'Qwen': '#9467bd',
        'GPT-4.1-mini': '#2ca02c',
    }
    colors = [color_map[model] for model in models]
    
    bars = plt.bar(models, success_rates, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.title('Counter-Intervention: Stealth Success Rate by Model\n(Percentage of AI texts misclassified as Human)', fontsize=16, pad=20)
    plt.xlabel('AI Model', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 110)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved stealth visualization to {output_path}")

if __name__ == '__main__':
    plot_stealth_results()
