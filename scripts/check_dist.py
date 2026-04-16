import json
import pandas as pd

with open('/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

human_count = sum(1 for row in data if row.get('human') and str(row['human']).strip())
deepseek_count = sum(1 for row in data if row.get('deepseek') and str(row['deepseek']).strip())
kimi_count = sum(1 for row in data if row.get('kimi') and str(row['kimi']).strip())
glm_count = sum(1 for row in data if row.get('glm') and str(row['glm']).strip())

print(f"Total problems: {len(data)}")
print(f"Human answers: {human_count}")
print(f"Deepseek answers: {deepseek_count}")
print(f"Kimi answers: {kimi_count}")
print(f"GLM answers: {glm_count}")
