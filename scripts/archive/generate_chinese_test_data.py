import os
import sys
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
import random

# Setup API Keys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
load_dotenv(os.path.join(base_dir, '.env'))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY")
GLM_API_KEY = os.environ.get("GLM_API_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")

NORMAL_PROMPT = "你是一个数学专家。请用中文详细地一步步解答以下数学问题。使用标准的中文和 LaTeX 公式。"

def call_deepseek(prompt, max_retries=3):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {"model": "deepseek-chat", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    for i in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 ** i)
    return ""

def call_kimi(prompt, max_retries=3):
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KIMI_API_KEY}"}
    data = {"model": "moonshot-v1-8k", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    for i in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 ** i)
    return ""

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

def translate_to_chinese(text, max_retries=3):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    prompt = f"Please translate the following mathematical text into natural and professional Chinese. Maintain all LaTeX formulas intact. Do not output anything else other than the translation.\n\nText:\n{text}"
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 2048}
    for i in range(max_retries):
        try:
            res = requests.post(url, headers=headers, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 ** i)
    return text # fallback to english if failed

def generate_chinese_test_dataset():
    output_path = os.path.join(base_dir, 'dataset', 'test_100_chinese_questions.json')
    
    # Check if we already have it
    if os.path.exists(output_path):
        print(f"Found existing {output_path}. Skipping generation.")
        return
        
    print("Loading Hendrycks Math dataset to sample 100 questions for Chinese translation...")
    dataset = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    dataset = dataset.shuffle(seed=999) # different seed
    
    selected_samples = []
    count = 0
    for i in range(len(dataset)):
        q_id = f"math_algebra_zh_test_{i}"
        selected_samples.append({
            'id': q_id,
            'subject': 'algebra',
            'problem': dataset[i]['problem'],
            'human': dataset[i]['solution']
        })
        count += 1
        if count >= 100:
            break
            
    print(f"Selected {len(selected_samples)} questions. Translating and calling APIs...")
    
    chinese_data = []
    lock = threading.Lock()
    
    def process_question(row):
        q_id = row['id']
        problem_en = row['problem']
        human_en = row['human']
        
        # Translate to Chinese
        problem_zh = translate_to_chinese(problem_en)
        human_zh = translate_to_chinese(human_en)
        
        prompt = f"问题：\n{problem_zh}\n\n请提供详细的数学解答过程。"
        
        result = {
            'id': q_id,
            'subject': row['subject'],
            'problem': problem_zh,
            'human': human_zh
        }
        
        result['deepseek'] = call_deepseek(prompt)
        result['kimi'] = call_kimi(prompt)
        result['glm'] = call_glm(prompt)
        result['qwen'] = call_qwen(prompt)
        
        with lock:
            chinese_data.append(result)
            if len(chinese_data) % 5 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chinese_data, f, ensure_ascii=False, indent=4)
        print(f"✅ Completed {q_id} [{len(chinese_data)}/100]")
        
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_question, row) for row in selected_samples]
        for future in as_completed(futures):
            future.result()
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chinese_data, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Chinese Test Dataset generated: {output_path}")

if __name__ == '__main__':
    generate_chinese_test_dataset()
