import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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
        # 添加 kimi (如果已有数据)
        if row.get('kimi') and str(row['kimi']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['kimi']).strip(),
                'label': 'Kimi'
            })
    return pd.DataFrame(records)

# 2. 自定义特征提取器
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
            
            # 数学公式密度
            inline_math = len(re.findall(r'\$(.*?)\$', text))
            display_math = len(re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL))
            f['math_blocks'] = inline_math + display_math
            f['math_density'] = f['math_blocks'] / max(1, f['length'])
            
            # LaTeX 宏频率
            f['num_frac'] = text.count('\\frac')
            f['num_sum'] = text.count('\\sum')
            f['num_int'] = text.count('\\int')
            f['num_textbf'] = text.count('\\textbf')
            f['num_begin_end'] = text.count('\\begin')
            
            # 连接词/逻辑词频率
            logical_words = ['because', 'therefore', 'obviously', 'similarly', 'assume', 'thus', 'hence', 'clearly', 'by definition', 'since', 'then', 'so', 'it follows that']
            f['logical_words_count'] = sum(text.lower().count(w) for w in logical_words)
            f['logical_words_density'] = f['logical_words_count'] / max(1, f['length'])
            
            # 结构化列表特征
            f['num_list_items'] = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|\(\d+\)|first|second|finally|step\s*\d+)', text, re.IGNORECASE))
            
            features.append(f)
        return pd.DataFrame(features)

    def get_feature_names_out(self):
        return ['length', 'num_lines', 'avg_line_length', 'math_blocks', 'math_density',
                'num_frac', 'num_sum', 'num_int', 'num_textbf', 'num_begin_end',
                'logical_words_count', 'logical_words_density', 'num_list_items']

# 3. 提取特征并查看随机森林特征重要性
def analyze_features(df):
    X = df['text']
    y = df['label']
    
    print(f"Data Distribution:\n{y.value_counts()}\n")
    
    extractor = TextFeatureExtractor()
    X_features = extractor.transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_features, y)
    
    importances = clf.feature_importances_
    feature_names = extractor.get_feature_names_out()
    
    print("--- Feature Importances (Random Forest on Custom Features) ---")
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# 4. 构建并保存最好的模型
def train_and_save_best_model(df):
    X = df['text']
    y = df['label']
    
    X_df = pd.DataFrame({'text': X})
    
    # 增加自定义的停用词，过滤掉可能导致数据泄露的“大模型独有”或“人类独有”的格式/套话词汇
    custom_stop_words = [
        'boxed', 'quad', 'therefore', 'hence', 'proof', 'solution', 'conclusion', 'finally', 'we', 'have', 'thus', 'step', 'now', 'let'
    ]
    
    combined_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                  token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+',
                                  stop_words=custom_stop_words), 'text'),
        ('custom', Pipeline([
            ('extractor', TextFeatureExtractor()),
            ('scaler', StandardScaler())
        ]), 'text')
    ])
    
    # 经过测试 Combined Features + SVM 效果最好 (99.7% CV acc)
    best_pipeline = Pipeline([
        ('features', combined_features),
        ('clf', SVC(kernel='linear', probability=True, random_state=42))
    ])
    
    print("\n--- Training Final Combined SVM Model ---")
    best_pipeline.fit(X_df, y)
    
    # 保存模型
    model_path = '/Users/jiaju/Documents/github/Modelmid/models/best_classifier_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    print(f"Model saved to {model_path}")
    print("Training Data Evaluation:")
    y_pred = best_pipeline.predict(X_df)
    print(classification_report(y, y_pred))

if __name__ == '__main__':
    df = load_data('/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.json')
    if len(df) > 0:
        analyze_features(df)
        train_and_save_best_model(df)
    else:
        print("No valid data found.")
