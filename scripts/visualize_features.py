import os
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        if row.get('human') and str(row['human']).strip():
            records.append({'id': row['id'], 'text': str(row['human']).strip(), 'label': 'Human'})
        if row.get('deepseek') and str(row['deepseek']).strip():
            records.append({'id': row['id'], 'text': str(row['deepseek']).strip(), 'label': 'Deepseek'})
        if row.get('kimi') and str(row['kimi']).strip():
            records.append({'id': row['id'], 'text': str(row['kimi']).strip(), 'label': 'Kimi'})
        if row.get('glm') and str(row['glm']).strip():
            records.append({'id': row['id'], 'text': str(row['glm']).strip(), 'label': 'GLM'})
    return pd.DataFrame(records)

# 2. 与 train_classifier.py 保持一致的特征提取
def extract_features(df):
    features = []
    for _, row in df.iterrows():
        text = row['text']
        f = {'label': row['label']}
        
        f['length'] = len(text)
        f['num_lines'] = len(text.split('\n'))
        f['avg_line_length'] = f['length'] / max(1, f['num_lines'])
        
        words = re.findall(r'\b\w+\b', text.lower())
        f['word_count'] = len(words)
        f['vocab_richness'] = len(set(words)) / max(1, len(words))
        f['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
        
        f['comma_density'] = text.count(',') / max(1, f['length']) * 1000
        f['period_density'] = text.count('.') / max(1, f['length']) * 1000
        
        inline_math = len(re.findall(r'(?<!\$)\$(?!\$)(.*?)\$', text))
        display_math = len(re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{align\}(.*?)\\end\{align\}', text, re.DOTALL))
        f['inline_math_count'] = inline_math
        f['display_math_count'] = display_math
        f['math_density'] = (inline_math + display_math) / max(1, f['length']) * 1000
        
        f['num_frac'] = text.count('\\frac')
        f['num_textbf'] = text.count('\\textbf')
        
        logical_words = ['because', 'therefore', 'obviously', 'similarly', 'assume', 'thus', 'hence', 'clearly', 'by definition', 'since', 'then', 'so', 'it follows that']
        f['logical_words_count'] = sum(text.lower().count(w) for w in logical_words)
        f['logical_words_density'] = f['logical_words_count'] / max(1, f['length']) * 1000
        
        declarative_words = ['we', 'let', 'suppose', 'consider', 'now', 'note']
        f['declarative_density'] = sum(words.count(w) for w in declarative_words) / max(1, f['length']) * 1000
        
        f['num_list_items'] = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|\(\d+\)|first|second|finally|step\s*\d+)', text, re.IGNORECASE))
        
        features.append(f)
        
    return pd.DataFrame(features)

def generate_visualizations(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    palette = {'Human': '#2ca02c', 'Deepseek': '#1f77b4', 'Kimi': '#ff7f0e', 'GLM': '#d62728'}
    
    # 1. Violin Plots (更好的分布视角，替代之前的 Boxplot)
    features_to_plot = [
        ('vocab_richness', '词汇丰富度 (Vocabulary Richness)', 'Unique Words / Total Words'),
        ('inline_math_count', '行内公式频次 (Inline Math Count)', 'Count'),
        ('display_math_count', '块级公式频次 (Display Math Count)', 'Count'),
        ('avg_line_length', '平均行长分布 (Avg Line Length)', 'Chars / Line'),
        ('declarative_density', '祈使代词密度 (Declarative Words Density)', 'Density per 1000 chars'),
        ('logical_words_density', '逻辑连接词密度 (Logical Words Density)', 'Density per 1000 chars')
    ]
    
    for feature, title, ylabel in features_to_plot:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='label', y=feature, data=df, palette=palette, inner='quartile')
        plt.title(title, fontsize=16)
        plt.xlabel('Source', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violin_{feature}.png'), dpi=300)
        plt.close()
        
    # 2. 特征相关性热力图 (Feature Correlation Heatmap)
    numeric_df = df.drop(columns=['label'])
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title('特征相关性热力图 (Feature Correlation Heatmap)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. PCA 2D 聚类降维散点图 (可解释性：看看四类样本在特征空间是否线性可分)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca_1', y='pca_2', hue='label', style='label', 
                    data=df, palette=palette, s=60, alpha=0.7)
    plt.title(f'PCA 降维特征空间聚类 (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
    plt.legend(title='Source', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters_2d.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

if __name__ == '__main__':
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    output_dir = os.path.join(base_dir, 'docs/figures')
    
    if os.path.exists(dataset_path):
        print("Loading data and extracting features for visualization...")
        raw_df = load_data(dataset_path)
        feat_df = extract_features(raw_df)
        print("Generating advanced plots (Violin, PCA, Heatmap)...")
        generate_visualizations(feat_df, output_dir)
    else:
        print(f"Error: Could not find dataset at {dataset_path}")