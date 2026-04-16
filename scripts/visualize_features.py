import os
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
# 解决中文显示问题（针对 macOS）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载并整理数据
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
    return pd.DataFrame(records)

# 2. 特征提取函数
def extract_features(df):
    features = []
    for _, row in df.iterrows():
        text = row['text']
        f = {'label': row['label']}
        
        # 基础排版特征
        f['length'] = len(text)
        f['num_lines'] = len(text.split('\n'))
        f['avg_line_length'] = f['length'] / max(1, f['num_lines'])
        
        # 公式特征
        inline_math = len(re.findall(r'\$(.*?)\$', text))
        display_math = len(re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL))
        f['math_blocks'] = inline_math + display_math
        
        # 逻辑词密度
        logical_words = ['because', 'therefore', 'obviously', 'similarly', 'assume', 'thus', 'hence', 'clearly', 'by definition', 'since', 'then', 'so', 'it follows that']
        f['logical_words_count'] = sum(text.lower().count(w) for w in logical_words)
        f['logical_words_density'] = f['logical_words_count'] / max(1, f['length']) * 1000 # 放大千倍以便展示
        
        # 特定宏
        f['num_textbf'] = text.count('\\textbf')
        
        features.append(f)
        
    return pd.DataFrame(features)

def generate_visualizations(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 定义我们要可视化的核心特征和对应的标题
    plot_configs = [
        ('num_lines', '行数分布 (Number of Lines)', '行数'),
        ('avg_line_length', '平均行长分布 (Average Line Length)', '平均字数/行'),
        ('length', '总字符长度分布 (Total Text Length)', '字符数'),
        ('math_blocks', '公式块数量分布 (Math Blocks Count)', '公式数量'),
        ('logical_words_density', '逻辑连接词密度 (Logical Words Density per 1000 chars)', '密度 (千分比)'),
        ('num_textbf', '\\textbf 使用频次 (Textbf Count)', '使用次数')
    ]
    
    palette = {'Human': '#2ca02c', 'Deepseek': '#1f77b4', 'Kimi': '#ff7f0e'}
    
    # 1. 绘制核心特征的箱线图对比
    for feature, title, ylabel in plot_configs:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='label', y=feature, data=df, palette=palette, width=0.5, showfliers=False)
        sns.stripplot(x='label', y=feature, data=df, color='black', alpha=0.3, jitter=True, size=3)
        plt.title(title, fontsize=16)
        plt.xlabel('来源 (Source)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'box_{feature}.png'), dpi=300)
        plt.close()
        
    # 2. 绘制散点图 (2D 决策边界可视化感受)
    # 挑选两个最具区分度的特征: 行数 vs 平均行长
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='num_lines', y='avg_line_length', hue='label', style='label', 
                    data=df, palette=palette, s=80, alpha=0.7)
    plt.title('行数 vs 平均行长 散点分布 (Lines vs Avg Line Length)', fontsize=16)
    plt.xlabel('行数', fontsize=14)
    plt.ylabel('平均字数/行', fontsize=14)
    plt.legend(title='Source', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_lines_vs_avglength.png'), dpi=300)
    plt.close()
    
    # 3. 雷达图 (对比均值)
    numeric_cols = [c[0] for c in plot_configs]
    mean_df = df.groupby('label')[numeric_cols].mean()
    # 标准化到 0-1 之间以便在同一张雷达图上显示
    normalized_mean = (mean_df - mean_df.min()) / (mean_df.max() - mean_df.min() + 1e-9)
    
    labels = normalized_mean.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for idx, row in normalized_mean.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('核心特征均值雷达图 (归一化)', size=16, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_features_mean.png'), dpi=300)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")

if __name__ == '__main__':
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    output_dir = os.path.join(base_dir, 'docs/figures')
    
    if os.path.exists(dataset_path):
        print("Loading data and extracting features...")
        raw_df = load_data(dataset_path)
        feat_df = extract_features(raw_df)
        print("Generating plots...")
        generate_visualizations(feat_df, output_dir)
    else:
        print(f"Error: Could not find dataset at {dataset_path}")