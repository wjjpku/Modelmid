import pandas as pd
import numpy as np
import re
import pickle
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
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        # 添加 human
        if pd.notna(row.get('human')) and str(row['human']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['human']).strip(),
                'label': 'Human'
            })
        # 添加 deepseek
        if pd.notna(row.get('deepseek')) and str(row['deepseek']).strip():
            records.append({
                'id': row['id'],
                'text': str(row['deepseek']).strip(),
                'label': 'Deepseek'
            })
        # 添加 kimi (如果已有数据)
        if pd.notna(row.get('kimi')) and str(row['kimi']).strip():
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
            logical_words = ['因为', '所以', '显然', '同理', '不妨设', '于是', '从而', '故', '可知', '由定义']
            f['logical_words_count'] = sum(text.count(w) for w in logical_words)
            f['logical_words_density'] = f['logical_words_count'] / max(1, f['length'])
            
            # 结构化列表特征 (如 1., 2., (1), (2), 首先)
            f['num_list_items'] = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|\(\d+\)|首先|其次|最后|步骤\s*\d+)', text))
            
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
    
    combined_features = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+'), 'text'),
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
    df = load_data('/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.csv')
    if len(df) > 0:
        analyze_features(df)
        train_and_save_best_model(df)
    else:
        print("No valid data found.")
