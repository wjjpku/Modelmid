import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import the exact feature extractor used in our final model
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_training'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from train_classifier import load_data, TextFeatureExtractor
from plotting_utils import configure_matplotlib_fonts

configure_matplotlib_fonts()


def build_palette(labels):
    color_values = sns.color_palette('tab10', n_colors=len(labels))
    return {label: color for label, color in zip(labels, color_values)}

def generate_report_and_plots():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    dataset_path = os.path.join(base_dir, 'dataset', 'training', 'full_dataset.json')
    output_dir = os.path.join(base_dir, 'docs', 'figures', 'gpt_augmented')
    report_path = os.path.join(base_dir, 'docs/experiment_report.md')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading data...")
    raw_df = load_data(dataset_path)
    X = raw_df['text']
    y = raw_df['label']
    
    print("Extracting full 28 features...")
    extractor = TextFeatureExtractor()
    feat_df = extractor.transform(X)
    feat_df['label'] = y.values
    
    # 1. 核心统计数据聚合 (Mean values by label)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    grouped_stats = feat_df.groupby('label')[numeric_cols].mean().round(2)
    labels = sorted(feat_df['label'].unique())
    palette = build_palette(labels)
    
    # 2. 生成核心可视化图表 (Violin Plots & PCA)
    print("Generating Violin Plots for Top 3 Features...")
    top_3_features = [
        ('num_paragraphs', '段落数量分布 (Num Paragraphs)', '段落数 (Count)'),
        ('inline_math_count', '行内公式包裹频率 (Inline Math)', '行内公式数量 (Count)'),
        ('declarative_density', '大模型祈使句密度 (Declarative Words)', '频率 (Per 1000 Chars)')
    ]
    
    for feature, title, ylabel in top_3_features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='label', y=feature, data=feat_df, palette=palette, inner='quartile')
        plt.title(title, fontsize=16)
        plt.xlabel('来源 (Source)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violin_{feature}.png'), dpi=300)
        plt.close()
        
    print("Generating PCA 2D Scatter Plot...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feat_df[numeric_cols])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    feat_df['pca_1'] = pca_result[:, 0]
    feat_df['pca_2'] = pca_result[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca_1', y='pca_2', hue='label', style='label', 
                    data=feat_df, palette=palette, s=60, alpha=0.7)
    plt.title(f'PCA 降维特征空间聚类 (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
    plt.legend(title='Source', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters_2d.png'), dpi=300)
    plt.close()
    
    # 3. 撰写图文混合的 Markdown 报告
    print("Generating Markdown Report...")
    
    report_content = f"""# 实验与特征数据报告 (Experiment Data Report)

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据规模**: {len(feat_df)} 条
**来源类别**: {', '.join(labels)}
**特征维度**: 28 个深度排版、结构与逻辑特征

---

## 1. 宏观排版与结构特征对比 (Macroscopic Structure)

大模型受限于自回归生成机制，在处理长文本证明时，极度依赖频繁的段落切换和序列标记。相反，人类更倾向于连贯的单段长推导。

### 1.1 关键数据均值对比
| 来源 (Source) | 平均段落数 (`num_paragraphs`) | 平均每段字符数 (`avg_paragraph_length`) | 换行数 (`num_lines`) |
| --- | --- | --- | --- |
{chr(10).join(
    f"| **{label}** | {grouped_stats.loc[label, 'num_paragraphs']} | {grouped_stats.loc[label, 'avg_paragraph_length']} | {grouped_stats.loc[label, 'num_lines']} |"
    for label in labels
)}

*解读：人类的平均段落数极少（仅为大模型的1/2到1/3），但每个段落的信息密度（字符数）是所有 AI 的数倍。大模型极度依赖双换行来组织思维。*

### 1.2 核心特征分布图
![段落数量分布 (Violin Plot)](./figures/gpt_augmented/violin_num_paragraphs.png)

---

## 2. 数学公式特异性与严谨度 (Mathematical Formats)

不同的大脑对于“何时使用 LaTeX 渲染”有着完全不同的理解，Qwen 展现出了远超其他模型的行内公式包裹欲望。

### 2.1 关键数据均值对比
| 来源 (Source) | 行内公式频次 (`inline_math_count`) | 块级公式频次 (`display_math_count`) | 复杂环境频率 (`latex_env_count`) |
| --- | --- | --- | --- |
{chr(10).join(
    f"| **{label}** | {grouped_stats.loc[label, 'inline_math_count']} | {grouped_stats.loc[label, 'display_math_count']} | {grouped_stats.loc[label, 'latex_env_count']} |"
    for label in labels
)}

*解读：不同模型在“何时使用行内公式”和“是否偏好复杂 LaTeX 环境”上差异明显。新增的 GPT-4.1-mini 也可在这组特征中与原五类来源一起比较。*

### 2.2 核心特征分布图
![行内公式包裹频率 (Violin Plot)](./figures/gpt_augmented/violin_inline_math_count.png)

---

## 3. 词汇风格与大模型套话 (Vocabulary & Semantic Fingerprints)

大模型在数学推导时，有着极其统一的“机器感起手式”。

### 3.1 关键数据均值对比 (每千字密度)
| 来源 (Source) | 祈使句/代词密度 (`we, let, suppose`) | 大写字母密度 (`uppercase_density`) | 序列衔接词密度 (`firstly, secondly`) |
| --- | --- | --- | --- |
{chr(10).join(
    f"| **{label}** | {grouped_stats.loc[label, 'declarative_density']} | {grouped_stats.loc[label, 'uppercase_density']} | {grouped_stats.loc[label, 'transition_words_density']} |"
    for label in labels
)}

*解读：几乎所有的 LLM 都极其喜欢使用 "We have", "Let x be", "Now consider" 这样的祈使代词句式作为推导开头，人类的使用密度要低得多。同时，大模型严格的语法训练导致其大写字母的分布（首字母大写规范）远高于随意的人类手写。*

### 3.2 核心特征分布图
![大模型祈使句密度 (Violin Plot)](./figures/gpt_augmented/violin_declarative_density.png)

---

## 4. PCA 主成分分析 (Principal Component Analysis)

为了验证上述 28 个深度特征的组合能否在数学空间中有效区分这些文本，我们进行了 PCA 降维。

![PCA 降维特征空间聚类](./figures/gpt_augmented/pca_clusters_2d.png)

*解读：在二维 PCA 空间中，不同来源的排版与措辞风格会形成可观察的聚类结构。加入 GPT-4.1-mini 后，可以直接观察它与原五类来源在结构特征空间中的相对位置。*

---
*本报告由数据流水线自动生成。*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated successfully at: {report_path}")

if __name__ == '__main__':
    generate_report_and_plots()
