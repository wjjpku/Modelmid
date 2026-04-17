import os
import sys
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv

# Load API keys
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY")
GLM_API_KEY = os.environ.get("GLM_API_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")

# --- Advanced Anti-Detection Prompt based on our Feature Engineering Insights ---
# We teach the AI what features to avoid without giving it human examples.
ANTI_DETECTION_SYSTEM_PROMPT = (
    "You are an expert mathematician solving a proof. However, you must strictly write your solution to mimic natural human handwriting habits, avoiding typical AI structural fingerprints.\n\n"
    "CRITICAL STRUCTURAL RULES (Must Follow Exactly):\n"
    "1. **Do NOT over-paragraph**: Do not start a new paragraph for every minor logical step. Combine steps into long, dense, continuous paragraphs. Limit yourself to 2-4 paragraphs maximum for the entire proof.\n"
    "2. **Do NOT overuse inline math ($...$)**: Only use inline math when absolutely necessary. Do not wrap every single variable or number in math mode. Write naturally.\n"
    "3. **Avoid AI Initiation Words**: NEVER start sentences with 'We have', 'Let', 'Suppose', 'Consider', 'Now', or 'Note that'. Start directly with the mathematical deduction or noun.\n"
    "4. **Avoid Mechanical Transitions**: NEVER use 'Firstly', 'Secondly', 'Moreover', 'Furthermore', or 'Finally'.\n"
    "5. **Avoid Mechanical Conclusions**: Do not write a concluding summary paragraph like 'In conclusion' or 'Therefore, the proof is complete'. Just end the proof naturally when the final result is reached.\n"
    "6. **Do NOT use lists**: Do not use numbered lists (1. 2. 3.) or bullet points. Write in continuous prose.\n"
    "7. **Minimize complex LaTeX environments**: Try to write out equations inline or on a simple new line rather than using complex \\begin{align} or \\begin{cases} environments unless absolutely unavoidable.\n\n"
    "Solve the problem directly in English using standard LaTeX syntax, but strictly adhere to the human-like formatting rules above."
)

def call_deepseek(prompt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {"model": "deepseek-chat", "messages": [{"role": "system", "content": ANTI_DETECTION_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_kimi(prompt):
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KIMI_API_KEY}"}
    data = {"model": "moonshot-v1-8k", "messages": [{"role": "system", "content": ANTI_DETECTION_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_glm(prompt):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GLM_API_KEY}"}
    data = {"model": "glm-4", "messages": [{"role": "system", "content": ANTI_DETECTION_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 2048}
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_qwen(prompt):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {QWEN_API_KEY}"}
    # Qwen might be strict about system prompts, so we merge it into user prompt
    data = {
        "model": "qwen-plus", 
        "messages": [
            {"role": "user", "content": ANTI_DETECTION_SYSTEM_PROMPT + "\n\n" + prompt}
        ], 
        "temperature": 0.5, 
        "max_tokens": 2048
    }
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def generate_stealth_data():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    stealth_output_path = os.path.join(base_dir, 'dataset/stealth_dataset.json')
    
    # Load original data to get 50 random test questions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    import random
    random.seed(42)
    # We select 50 random questions from the dataset to perform the anti-detection experiment
    sample_questions = random.sample(data, 50)
    
    stealth_data = []
    
    lock = threading.Lock()
    
    def process_question(row):
        q_id = row['id']
        problem = row['problem']
        prompt = f"Please solve the following math problem:\n\n{problem}"
        
        result = {
            'id': q_id,
            'problem': problem,
            'human': row['human'], # Keep original human for baseline comparison
        }
        
        # Try Deepseek
        try:
            result['deepseek_stealth'] = call_deepseek(prompt)
        except Exception as e:
            print(f"Deepseek error on ID {q_id}: {e}")
            
        # Try Kimi
        try:
            result['kimi_stealth'] = call_kimi(prompt)
        except Exception as e:
            print(f"Kimi error on ID {q_id}: {e}")
            
        # Try GLM
        try:
            result['glm_stealth'] = call_glm(prompt)
        except Exception as e:
            print(f"GLM error on ID {q_id}: {e}")
            
        # Try Qwen
        try:
            result['qwen_stealth'] = call_qwen(prompt)
        except Exception as e:
            print(f"Qwen error on ID {q_id}: {e}")
            
        with lock:
            stealth_data.append(result)
            with open(stealth_output_path, 'w', encoding='utf-8') as f:
                json.dump(stealth_data, f, ensure_ascii=False, indent=4)
        
        print(f"Completed stealth generation for ID {q_id}")
        
    print(f"Starting generation of {len(sample_questions)} stealth questions using 4 LLMs...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_question, row) for row in sample_questions]
        for future in as_completed(futures):
            future.result()
            
    print(f"Stealth data generation complete! Saved to {stealth_output_path}")

if __name__ == '__main__':
    generate_stealth_data()