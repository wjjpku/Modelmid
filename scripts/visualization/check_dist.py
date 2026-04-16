import json
import json
import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
with open(os.path.join(base_dir, 'dataset/full_dataset.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

human_count = sum(1 for row in data if row.get('human') and str(row['human']).strip())
deepseek_count = sum(1 for row in data if row.get('deepseek') and str(row['deepseek']).strip())
kimi_count = sum(1 for row in data if row.get('kimi') and str(row['kimi']).strip())
glm_count = sum(1 for row in data if row.get('glm') and str(row['glm']).strip())
qwen_count = sum(1 for row in data if row.get('qwen') and str(row['qwen']).strip())

print(f"Total problems: {len(data)}")
print(f"Human answers: {human_count}")
print(f"Deepseek answers: {deepseek_count}")
print(f"Kimi answers: {kimi_count}")
print(f"GLM answers: {glm_count}")
print(f"Qwen answers: {qwen_count}")
