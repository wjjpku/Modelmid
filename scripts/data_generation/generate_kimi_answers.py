import os
import json
import time
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')/.env'
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key_val = line.split('=', 1)
                    if len(key_val) == 2:
                        os.environ[key_val[0].strip()] = key_val[1].strip().strip('"').strip("'")

load_env()

KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://api.moonshot.cn/v1/chat/completions"

def generate_answer_with_kimi(problem_content: str, current_id: int) -> dict:
    system_prompt = (
        "You are a professional mathematics professor. Please provide a detailed and rigorous mathematical derivation and proof using standard LaTeX format.\n"
        "Requirements:\n"
        "1. The solution must be logically clear with complete steps.\n"
        "2. Use standard LaTeX syntax for all mathematical formulas (e.g., $...$ or \\[...\\]).\n"
        "3. Do not output redundant explanations or pleasantries; directly output the problem-solving steps in English.\n"
    )
    
    user_prompt = f"Please provide a detailed solution:\n\n{problem_content}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KIMI_API_KEY}"
    }
    
    data = {
        "model": "moonshot-v1-8k",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, 
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        print(f"Successfully generated for ID {current_id}.")
        return {"id": current_id, "answer": answer, "error": None}
    except Exception as e:
        print(f"Error calling Kimi API for ID {current_id}: {e}")
        return {"id": current_id, "answer": "", "error": str(e)}

def process_dataset(file_path: str, max_workers: int = 3):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
            
    print(f"Loaded {len(data)} records from {file_path}")
    
    tasks = []
    for row in data:
        if not row.get('kimi'):
            tasks.append({
                'id': row.get('id'),
                'problem': row.get('problem', '')
            })
            
    if not tasks:
        print("All problems already have Kimi answers.")
        return
        
    print(f"Found {len(tasks)} problems needing Kimi answers. Starting {max_workers} concurrent workers...")
    
    results_map = {}
    completed_count = 0
    lock = threading.Lock()
    
    def save_single_result(res_id, res_answer):
        # 内存操作：直接更新内存中的 data，然后原子性地写回文件
        with lock:
            for r in data:
                if r.get('id') == res_id:
                    r['kimi'] = res_answer
                    break
                    
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(generate_answer_with_kimi, task['problem'], task['id']): task['id'] 
            for task in tasks
        }
        
        for future in as_completed(future_to_id):
            task_id = future_to_id[future]
            try:
                res = future.result()
                if res['answer']:
                    results_map[res['id']] = res['answer']
                    save_single_result(res['id'], res['answer'])
                    completed_count += 1
                    print(f"--- Saved progress for ID {res['id']} ({completed_count}/{len(tasks)}) ---")
            except Exception as exc:
                print(f"Task ID {task_id} generated an exception: {exc}")
                
    print(f"\nAll tasks completed! Added {len(results_map)} new Kimi answers.")

if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')/dataset/full_dataset.json'
    if KIMI_API_KEY == "YOUR_API_KEY_HERE" or not KIMI_API_KEY:
        print("WARNING: API KEY not set.")
    else:
        # Kimi API rate limit is strict, so keep max_workers low
        process_dataset(dataset_path, max_workers=50)
