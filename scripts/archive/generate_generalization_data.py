import os
import sys
import json
import time
import requests
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY")
GLM_API_KEY = os.environ.get("GLM_API_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")

NORMAL_PROMPT = "You are a helpful and expert mathematician. Please solve the following math problem step by step. Use standard English and LaTeX for any mathematical expressions."

def call_deepseek(prompt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {"model": "deepseek-chat", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_kimi(prompt):
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KIMI_API_KEY}"}
    data = {"model": "moonshot-v1-8k", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_glm(prompt):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    data = {"model": "glm-4", "messages": [{"role": "system", "content": NORMAL_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_qwen(prompt):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {QWEN_API_KEY}"}
    data = {
        "model": "qwen-plus", 
        "messages": [
            {"role": "system", "content": NORMAL_PROMPT},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.5, 
        "max_tokens": 2048
    }
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def generate_generalization_data():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    output_path = os.path.join(base_dir, 'dataset/generalization_dataset.json')
    
    print("Loading Hendrycks MATH dataset...")
    subjects = ["algebra", "geometry", "number_theory", "precalculus"]
    samples_per_subject = 10
    
    selected_samples = []
    
    for subject in subjects:
        print(f"Loading subset: {subject}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        # Select first 10
        for i in range(samples_per_subject):
            selected_samples.append({
                'id': f"math_{subject}_{i}",
                'subject': subject,
                'problem': dataset[i]['problem'],
                'human': dataset[i]['solution']
            })
            
    print(f"Selected {len(selected_samples)} questions across {len(subjects)} subjects.")
    
    generalization_data = []
    lock = threading.Lock()
    
    def process_question(row):
        q_id = row['id']
        problem = row['problem']
        prompt = f"Problem:\n{problem}\n\nPlease provide a clear mathematical solution."
        
        result = {
            'id': q_id,
            'subject': row['subject'],
            'problem': problem,
            'human': row['human']
        }
        
        try: result['deepseek'] = call_deepseek(prompt)
        except Exception as e: print(f"Deepseek error on {q_id}: {e}")
            
        try: result['kimi'] = call_kimi(prompt)
        except Exception as e: print(f"Kimi error on {q_id}: {e}")
            
        try: result['glm'] = call_glm(prompt)
        except Exception as e: print(f"GLM error on {q_id}: {e}")
            
        try: result['qwen'] = call_qwen(prompt)
        except Exception as e: print(f"Qwen error on {q_id}: {e}")
            
        with lock:
            generalization_data.append(result)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(generalization_data, f, ensure_ascii=False, indent=4)
        
        print(f"Completed {q_id} ({row['subject']})")
        
    print("Starting AI generation for generalization test...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_question, row) for row in selected_samples]
        for future in as_completed(futures):
            future.result()
            
    print(f"Generalization data generation complete! Saved to {output_path}")

if __name__ == '__main__':
    generate_generalization_data()