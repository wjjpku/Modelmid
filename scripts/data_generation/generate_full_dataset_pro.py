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

NORMAL_PROMPT = "You are a helpful and expert mathematician. Please solve the following math problem step by step. Use standard English and LaTeX for any mathematical expressions."

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

def generate_pro_dataset():
    output_path = os.path.join(base_dir, 'dataset', 'training', 'full_dataset_pro.json')
    original_dataset_path = os.path.join(base_dir, 'dataset', 'training', 'full_dataset.json')
    
    # 1. Load original 1000 items
    print(f"Loading original 1000 items from {original_dataset_path}...")
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    for item in original_data:
        item['subject'] = 'general_math'  # Add default subject to original data
        
    # Check if we already have a partial pro dataset
    existing_ids = {str(item['id']) for item in original_data}
    pro_data = original_data.copy()
    
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, resuming...")
        with open(output_path, 'r', encoding='utf-8') as f:
            partial_pro_data = json.load(f)
        # Update pro_data with already generated new items
        for item in partial_pro_data:
            if str(item['id']) not in existing_ids:
                pro_data.append(item)
                existing_ids.add(str(item['id']))
    
    print(f"Currently have {len(pro_data)} items in pro_data.")
    
    # 2. Select 1000 new questions from Hendrycks MATH
    target_new = 1000
    already_generated = len(pro_data) - len(original_data)
    need_to_generate = target_new - already_generated
    
    if need_to_generate <= 0:
        print("Already have 1000+ new items. Skipping generation.")
        return
        
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    samples_per_subject = target_new // len(subjects) + 1
    
    print(f"Need to generate {need_to_generate} new items across {len(subjects)} subjects (~{samples_per_subject} per subject).")
    
    selected_samples = []
    
    for subject in subjects:
        print(f"Loading subset: {subject}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        # Shuffle with a seed to get diverse questions
        dataset = dataset.shuffle(seed=42)
        count = 0
        for i in range(len(dataset)):
            q_id = f"math_{subject}_{i}"
            if str(q_id) in existing_ids:
                continue
            
            selected_samples.append({
                'id': q_id,
                'subject': subject,
                'problem': dataset[i]['problem'],
                'human': dataset[i]['solution']
            })
            count += 1
            if count >= samples_per_subject:
                break
                
    # Truncate to exact needed amount
    selected_samples = selected_samples[:need_to_generate]
    print(f"Selected {len(selected_samples)} new questions.")
    
    # 3. Concurrent Generation
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
        
        result['deepseek'] = call_deepseek(prompt)
        result['kimi'] = call_kimi(prompt)
        result['glm'] = call_glm(prompt)
        result['qwen'] = call_qwen(prompt)
        
        # Only save if we got at least 2 valid answers
        valid_answers = sum([1 for k in ['deepseek', 'kimi', 'glm', 'qwen'] if len(result[k]) > 50])
        if valid_answers >= 2:
            with lock:
                pro_data.append(result)
                # Incremental save every 10 items
                if len(pro_data) % 10 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(pro_data, f, ensure_ascii=False, indent=4)
            print(f"✅ Completed {q_id} ({row['subject']}) [{len(pro_data)}/2000]")
        else:
            print(f"❌ Failed to get enough valid answers for {q_id}. Skipping.")
            
    print(f"Starting concurrent API calls with 50 workers...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_question, row) for row in selected_samples]
        for future in as_completed(futures):
            future.result()
            
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pro_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Dataset PRO generation complete! Total items: {len(pro_data)}. Saved to {output_path}")

if __name__ == '__main__':
    generate_pro_dataset()
