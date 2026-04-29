import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_stealth_results():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    output_path = os.path.join(base_dir, 'docs', 'figures', 'stealth_success_rate.png')
    
    models = ['Deepseek', 'Kimi', 'GLM', 'Qwen']
    success_rates = [68.18, 18.00, 70.00, 98.00]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    
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
