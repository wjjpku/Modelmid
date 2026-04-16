import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        if pd.notna(row.get('human')) and str(row['human']).strip():
            records.append({'text': str(row['human']).strip(), 'label': 'Human'})
        if pd.notna(row.get('deepseek')) and str(row['deepseek']).strip():
            records.append({'text': str(row['deepseek']).strip(), 'label': 'Deepseek'})
        if pd.notna(row.get('kimi')) and str(row['kimi']).strip():
            records.append({'text': str(row['kimi']).strip(), 'label': 'Kimi'})
    return pd.DataFrame(records)

df = load_data('/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.csv')
X = df['text']
y = df['label']

print("Extracting TF-IDF features to find potential tricky words...")
# 使用和训练模型中一致的参数
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b|\\\[a-zA-Z]+')
X_tfidf = vectorizer.fit_transform(X)
feature_names = vectorizer.get_feature_names_out()

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tfidf, y)

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n--- Top 30 TF-IDF Features ---")
for i in range(30):
    idx = sorted_idx[i]
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
