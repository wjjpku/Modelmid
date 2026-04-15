import pandas as pd

df = pd.read_csv('/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.csv')
human_count = df['human'].dropna().apply(lambda x: len(str(x).strip()) > 0).sum()
deepseek_count = df['deepseek'].dropna().apply(lambda x: len(str(x).strip()) > 0).sum()
kimi_count = df['kimi'].dropna().apply(lambda x: len(str(x).strip()) > 0).sum()

print(f"Human answers: {human_count}")
print(f"Deepseek answers: {deepseek_count}")
print(f"Kimi answers: {kimi_count}")
