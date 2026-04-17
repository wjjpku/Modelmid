import pandas as pd
import numpy as np
import re
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载并整理数据
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for row in data:
        # 添加 human
        if row.get('human') and str(row['human']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['human']).strip(),
                'label': 'Human'
            })
        # 添加 deepseek
        if row.get('deepseek') and str(row['deepseek']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['deepseek']).strip(),
                'label': 'Deepseek'
            })
        # 添加 kimi 
        if row.get('kimi') and str(row['kimi']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['kimi']).strip(),
                'label': 'Kimi'
            })
        # 添加 GLM
        if row.get('glm') and str(row['glm']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['glm']).strip(),
                'label': 'GLM'
            })
        # 添加 Qwen
        if row.get('qwen') and str(row['qwen']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['qwen']).strip(),
                'label': 'Qwen'
            })
    return pd.DataFrame(records)

# 2. 深度优化的自定义特征提取器
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        features = []
        for text in X:
            f = {}
            # 基础长度特征
            f['length'] = len(text)
            f['num_lines'] = len(text.split('\n'))
            f['avg_line_length'] = f['length'] / max(1, f['num_lines'])
            
            # 段落与结构特异性特征
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            f['num_paragraphs'] = len(paragraphs)
            f['avg_paragraph_length'] = f['length'] / max(1, max(1, len(paragraphs)))
            f['uppercase_density'] = sum(1 for c in text if c.isupper()) / max(1, f['length']) * 1000
            f['digit_density'] = sum(1 for c in text if c.isdigit()) / max(1, f['length']) * 1000
            
            # 词汇与句子复杂度
            words = re.findall(r'\b\w+\b', text.lower())
            f['word_count'] = len(words)
            f['vocab_richness'] = len(set(words)) / max(1, len(words))  # 词汇丰富度
            f['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words)) # 平均单词长度
            
            # 标点符号频率 (用于反映句子切割频率)
            f['comma_density'] = text.count(',') / max(1, f['length']) * 1000
            f['period_density'] = text.count('.') / max(1, f['length']) * 1000
            
            # 数学公式细分: 行内(inline) vs 块级(display)
            inline_math = len(re.findall(r'(?<!\$)\$(?!\$)(.*?)\$', text))
            display_math = len(re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{align\}(.*?)\\end\{align\}', text, re.DOTALL))
            f['inline_math_count'] = inline_math
            f['display_math_count'] = display_math
            f['math_density'] = (inline_math + display_math) / max(1, f['length']) * 1000
            
            # LaTeX 宏频率与复杂环境特征
            f['num_frac'] = text.count('\\frac')
            f['num_textbf'] = text.count('\\textbf')
            f['latex_env_count'] = len(re.findall(r'\\begin\{.*?\}', text)) # 复杂环境频率
            f['left_right_brackets'] = len(re.findall(r'\\left[\[\(\\{]', text)) # 严谨的大括号使用
            f['math_cal_bb'] = len(re.findall(r'\\math(cal|bb|bf|rm|frak)\{.*?\}', text)) # 数学字体丰富度
            f['implication_arrows'] = len(re.findall(r'\\(Rightarrow|implies|rightarrow|iff|Leftrightarrow)', text)) # 逻辑箭头
            f['qed_symbols'] = len(re.findall(r'\\(blacksquare|square|qed)|Q\.E\.D\.', text, re.IGNORECASE)) # 结束符倾向
            
            # 连接词/逻辑词频率
            logical_words = ['because', 'therefore', 'obviously', 'similarly', 'assume', 'thus', 'hence', 'clearly', 'by definition', 'since', 'then', 'so', 'it follows that']
            f['logical_words_count'] = sum(text.lower().count(w) for w in logical_words)
            f['logical_words_density'] = f['logical_words_count'] / max(1, f['length']) * 1000
            
            # 祈使词与代词频率 (大模型爱用的表达)
            declarative_words = ['we', 'let', 'suppose', 'consider', 'now', 'note']
            f['declarative_density'] = sum(words.count(w) for w in declarative_words) / max(1, f['length']) * 1000
            
            # 序列衔接词频率 (First, Second, Finally)
            transition_words = ['firstly', 'secondly', 'moreover', 'furthermore', 'additionally', 'next', 'finally']
            f['transition_words_density'] = sum(text.lower().count(w) for w in transition_words) / max(1, f['length']) * 1000
            
            # 结论引导词频率
            conclusion_words = ['in conclusion', 'to sum up', 'the final answer is', 'we can conclude', 'summary']
            f['conclusion_words_density'] = sum(text.lower().count(w) for w in conclusion_words) / max(1, f['length']) * 1000
            
            # 结构化列表特征
            f['num_list_items'] = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|\(\d+\)|first|second|step\s*\d+)', text, re.IGNORECASE))
            
            features.append(f)
        return pd.DataFrame(features)

    def get_feature_names_out(self):
        return ['length', 'num_lines', 'avg_line_length', 'num_paragraphs', 'avg_paragraph_length', 
                'uppercase_density', 'digit_density', 'word_count', 'vocab_richness', 'avg_word_length',
                'comma_density', 'period_density', 'inline_math_count', 'display_math_count', 'math_density',
                'num_frac', 'num_textbf', 'latex_env_count', 'left_right_brackets', 'math_cal_bb', 
                'implication_arrows', 'qed_symbols', 'logical_words_count', 'logical_words_density', 
                'declarative_density', 'transition_words_density', 'conclusion_words_density', 'num_list_items']

# 2. 提取特征并查看随机森林特征重要性 (可解释性)
def analyze_features_and_interpretability(df):
    X = df['text']
    y = df['label']
    
    print(f"Data Distribution:\n{y.value_counts()}\n")
    
    extractor = TextFeatureExtractor()
    X_features = extractor.transform(X)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_features, y)
    
    importances = clf.feature_importances_
    feature_names = extractor.get_feature_names_out()
    
    print("--- Feature Importances (Random Forest on Custom Features) ---")
    sorted_idx = np.argsort(importances)[::-1]
    
    # 打印前 10 名
    for idx in sorted_idx[:15]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
    # 保存特征重要性柱状图
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
    plt.title("Feature Importances for Differentiating Math Answer Sources", fontsize=16)
    plt.xlabel("Random Forest Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'figures', 'feature_importances.png'), dpi=300)
    plt.close()
    print("Feature importance plot saved.")

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, "toarray") else X

# 3. 构建并保存最好的模型 (HistGradientBoostingClassifier)
def train_and_save_best_model(df):
    X = df['text']
    y = df['label']
    
    X_df = pd.DataFrame({'text': X})
    
    # 移除自定义停用词，允许模型利用作弊词（Trick Words）来进行分类，以便对比最终结果
    combined_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                  token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+',
                                  stop_words=None), 'text'),
        ('custom', Pipeline([
            ('extractor', TextFeatureExtractor()),
            ('scaler', StandardScaler())
        ]), 'text')
    ])

    # 经过多模型实验，HistGradientBoostingClassifier 效果最好 (95.28%)
    best_pipeline = Pipeline([
        ('features', combined_features),
        ('to_dense', DenseTransformer()),
        ('clf', HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ])
    
    print("\n--- Training Final Combined Gradient Boosting Model (5-Class) ---")
    
    best_pipeline.fit(X_df, y)
    
    # Save the complete pipeline
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'best_classifier_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Model successfully saved to {model_path}")
    
    # Evaluate on training set
    y_pred = best_pipeline.predict(X_df)
    print("\nClassification Report (Training Set):")
    print(classification_report(y, y_pred))
    
    # Cross validation
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # scores = cross_val_score(best_pipeline, X_df, y, cv=cv, scoring='accuracy', n_jobs=-1)
    # print(f"\n5-Fold CV Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == '__main__':
    df = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'full_dataset.json'))
    if len(df) > 0:
        analyze_features_and_interpretability(df)
        train_and_save_best_model(df)
    else:
        print("No valid data found.")
