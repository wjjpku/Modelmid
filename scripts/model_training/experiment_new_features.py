import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

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
        if row.get('qwen') and str(row['qwen']).strip():
            records.append({'id': row['id'], 'text': str(row['qwen']).strip(), 'label': 'Qwen'})
    return pd.DataFrame(records)

# ----------------- BASELINE (Old features) -----------------
class BaselineFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        features = []
        for text in X:
            f = {}
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

# ----------------- ENHANCED (New features added) -----------------
class EnhancedFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        features = []
        for text in X:
            f = {}
            # 1. 基础特征 (继承 Baseline)
            f['length'] = len(text)
            lines = text.split('\n')
            f['num_lines'] = len(lines)
            f['avg_line_length'] = f['length'] / max(1, f['num_lines'])
            
            words = re.findall(r'\b\w+\b', text.lower())
            f['word_count'] = len(words)
            f['vocab_richness'] = len(set(words)) / max(1, len(words))
            f['avg_word_length'] = sum(len(w) for w in words) / max(1, len(words))
            
            f['comma_density'] = text.count(',') / max(1, f['length']) * 1000
            f['period_density'] = text.count('.') / max(1, f['length']) * 1000
            
            # 2. 段落结构与字符特异性特征 (NEW)
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            f['num_paragraphs'] = len(paragraphs)
            f['avg_paragraph_length'] = f['length'] / max(1, len(paragraphs))
            f['uppercase_density'] = sum(1 for c in text if c.isupper()) / max(1, f['length']) * 1000
            f['digit_density'] = sum(1 for c in text if c.isdigit()) / max(1, f['length']) * 1000
            
            # 3. 深度 LaTeX 公式格式特征 (NEW)
            inline_math = len(re.findall(r'(?<!\$)\$(?!\$)(.*?)\$', text))
            display_math = len(re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{align\}(.*?)\\end\{align\}', text, re.DOTALL))
            f['inline_math_count'] = inline_math
            f['display_math_count'] = display_math
            f['math_density'] = (inline_math + display_math) / max(1, f['length']) * 1000
            f['num_frac'] = text.count('\\frac')
            f['num_textbf'] = text.count('\\textbf')
            
            f['latex_env_count'] = len(re.findall(r'\\begin\{.*?\}', text)) # 复杂环境频率
            f['left_right_brackets'] = len(re.findall(r'\\left[\[\(\\{]', text)) # 严谨的大括号使用
            f['math_cal_bb'] = len(re.findall(r'\\math(cal|bb|bf|rm|frak)\{.*?\}', text)) # 数学字体丰富度
            f['implication_arrows'] = len(re.findall(r'\\(Rightarrow|implies|rightarrow|iff|Leftrightarrow)', text)) # 逻辑箭头
            f['qed_symbols'] = len(re.findall(r'\\(blacksquare|square|qed)|Q\.E\.D\.', text, re.IGNORECASE)) # 结束符倾向
            
            # 4. 语义风格特征 (NEW/Enhanced)
            logical_words = ['because', 'therefore', 'obviously', 'similarly', 'assume', 'thus', 'hence', 'clearly', 'by definition', 'since', 'then', 'so', 'it follows that']
            f['logical_words_count'] = sum(text.lower().count(w) for w in logical_words)
            f['logical_words_density'] = f['logical_words_count'] / max(1, f['length']) * 1000
            
            declarative_words = ['we', 'let', 'suppose', 'consider', 'now', 'note']
            f['declarative_density'] = sum(words.count(w) for w in declarative_words) / max(1, f['length']) * 1000
            
            transition_words = ['firstly', 'secondly', 'moreover', 'furthermore', 'additionally', 'next', 'finally']
            f['transition_words_density'] = sum(text.lower().count(w) for w in transition_words) / max(1, f['length']) * 1000
            
            conclusion_words = ['in conclusion', 'to sum up', 'the final answer is', 'we can conclude', 'summary']
            f['conclusion_words_density'] = sum(text.lower().count(w) for w in conclusion_words) / max(1, f['length']) * 1000
            
            f['num_list_items'] = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|\(\d+\)|first|second|step\s*\d+)', text, re.IGNORECASE))
            
            features.append(f)
        return pd.DataFrame(features)
    
    def get_feature_names_out(self):
        return ['length', 'num_lines', 'avg_line_length', 'word_count', 'vocab_richness', 'avg_word_length',
                'comma_density', 'period_density', 'num_paragraphs', 'avg_paragraph_length', 
                'uppercase_density', 'digit_density', 'inline_math_count', 'display_math_count', 'math_density',
                'num_frac', 'num_textbf', 'latex_env_count', 'left_right_brackets', 'math_cal_bb',
                'implication_arrows', 'qed_symbols', 'logical_words_count', 'logical_words_density',
                'declarative_density', 'transition_words_density', 'conclusion_words_density', 'num_list_items']

def run_experiment():
    print("Loading dataset...")
    df = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')/dataset/full_dataset.json')
    X_df = pd.DataFrame({'text': df['text']})
    y = df['label']
    print(f"Total samples: {len(df)}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def build_pipeline(extractor_class, tfidf_stop_words):
        combined_features = ColumnTransformer([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                      token_pattern=r'(?u)\b\w+\b|\\[a-zA-Z]+',
                                      stop_words=tfidf_stop_words), 'text'),
            ('custom', Pipeline([
                ('extractor', extractor_class()),
                ('scaler', StandardScaler())
            ]), 'text')
        ])
        return Pipeline([
            ('features', combined_features),
            ('clf', SVC(kernel='linear', probability=True, random_state=42))
        ])

    print("\n--- Model 1: Baseline (Old Features + stop_words='english') ---")
    pipe1 = build_pipeline(BaselineFeatureExtractor, 'english')
    scores1 = cross_val_score(pipe1, X_df, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy: {scores1.mean():.4f} (+/- {scores1.std():.4f})")

    print("\n--- Model 2: Enhanced (New Features + stop_words='english') ---")
    pipe2 = build_pipeline(EnhancedFeatureExtractor, 'english')
    scores2 = cross_val_score(pipe2, X_df, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy: {scores2.mean():.4f} (+/- {scores2.std():.4f})")
    
    print("\n--- Model 3: Enhanced (New Features + stop_words=None) MAX PERFORMANCE ---")
    pipe3 = build_pipeline(EnhancedFeatureExtractor, None)
    scores3 = cross_val_score(pipe3, X_df, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy: {scores3.mean():.4f} (+/- {scores3.std():.4f})")
    
    # Analyze Feature Importance of Enhanced Features
    print("\n--- Analyzing Enhanced Feature Importances (Random Forest) ---")
    from sklearn.ensemble import RandomForestClassifier
    extractor = EnhancedFeatureExtractor()
    X_features = extractor.transform(X_df['text'])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_features, y)
    importances = clf.feature_importances_
    feature_names = extractor.get_feature_names_out()
    sorted_idx = np.argsort(importances)[::-1]
    
    for idx in sorted_idx[:15]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

if __name__ == '__main__':
    run_experiment()
