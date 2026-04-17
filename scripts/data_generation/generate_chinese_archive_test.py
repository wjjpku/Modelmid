import os
import sys
import pandas as pd
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

# Setup API Keys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
load_dotenv(os.path.join(base_dir, '.env'))

GLM_API_KEY = os.environ.get("GLM_API_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")

NORMAL_PROMPT = "你是一个数学专家。请用中文详细地一步步解答以下数学问题。使用标准的中文和 LaTeX 公式。"

def call_glm(prompt, max_retries=3):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    data = {"model": "glm-4", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    for i in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 ** i)
    return ""

def call_qwen(prompt, max_retries=3):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {QWEN_API_KEY}"}
    data = {
        "model": "qwen-plus", 
        "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], 
        "temperature": 0.5, "max_tokens": 2048
    }
    for i in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 ** i)
    return ""

def process_archive_data():
    csv_path = os.path.join(base_dir, 'archive', 'full_dataset_cn.csv')
    output_path = os.path.join(base_dir, 'dataset', 'test_100_chinese_archive_questions.json')
    
    if os.path.exists(output_path):
        print(f"Found existing {output_path}. Skipping generation.")
        return
        
    df = pd.read_csv(csv_path)
    
    # We need 100 questions. The dataset has 219. Let's sample 100 randomly.
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(df)} questions from archive.")
    
    chinese_data = []
    lock = threading.Lock()
    
    def process_row(idx, row):
        q_id = str(row['id'])
        problem_zh = str(row['content'])
        human_zh = str(row['human'])
        deepseek_zh = str(row['deepseek'])
        kimi_zh = str(row['kimi'])
        
        prompt = f"问题：\n{problem_zh}\n\n请提供详细的数学解答过程。"
        
        result = {
            'id': q_id,
            'subject': str(row['course']),
            'problem': problem_zh,
            'human': human_zh,
            'deepseek': deepseek_zh,
            'kimi': kimi_zh
        }
        
        # Missing models are GLM and Qwen
        result['glm'] = call_glm(prompt)
        result['qwen'] = call_qwen(prompt)
        
        with lock:
            chinese_data.append(result)
            if len(chinese_data) % 5 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chinese_data, f, ensure_ascii=False, indent=4)
            print(f"✅ Completed {q_id} [{len(chinese_data)}/100]")
            
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_row, i, row) for i, row in df.iterrows()]
        for future in as_completed(futures):
            future.result()
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chinese_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Chinese Archive Test Dataset generated: {output_path}")

if __name__ == '__main__':
    process_archive_data()