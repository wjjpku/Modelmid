import os
import csv
import time
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试读取 .env 文件中的配置
def load_env():
    env_path = '/Users/jiaju/Documents/github/Modelmid/.env'
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key_val = line.split('=', 1)
                    if len(key_val) == 2:
                        os.environ[key_val[0].strip()] = key_val[1].strip().strip('"').strip("'")

load_env()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://api.deepseek.com/chat/completions"

def generate_answer_with_deepseek(problem_content: str, course: str, current_id: str) -> dict:
    """调用 Deepseek API 生成数学解答"""
    
    system_prompt = (
        "你是一个专业的数学教授。请用规范的 LaTeX 格式提供详细、严谨的数学推导和证明。\n"
        "要求：\n"
        "1. 解答需逻辑清晰，步骤完整。\n"
        "2. 所有数学公式请使用规范的 LaTeX 语法（如 $...$ 或 \\[...\\]）。\n"
        "3. 不要输出多余的解释或寒暄，直接输出解题步骤。\n"
    )
    
    user_prompt = f"这是【{course}】的一道题目，请给出详细解答：\n\n{problem_content}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, 
        "max_tokens": 2048
    }
    
    try:
        # print(f"Sending request for {current_id}...")
        response = requests.post(API_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        print(f"Successfully generated for {current_id}.")
        return {"id": current_id, "answer": answer, "error": None}
    except Exception as e:
        print(f"Error calling Deepseek API for {current_id}: {e}")
        return {"id": current_id, "answer": "", "error": str(e)}

def process_dataset(file_path: str, max_workers: int = 5):
    """读取 CSV 文件，并发调用 API 补充 Deepseek 解答，最后保存"""
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    data: List[Dict[str, str]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            data.append(row)
            
    if not fieldnames or 'id' not in fieldnames or 'content' not in fieldnames or 'deepseek' not in fieldnames:
        print("CSV format error: missing 'id', 'content' or 'deepseek' columns.")
        return
        
    print(f"Loaded {len(data)} records from {file_path}")
    
    # 筛选出需要生成解答的任务
    tasks = []
    for row in data:
        if not row.get('deepseek'):
            tasks.append({
                'id': row.get('id', ''),
                'content': row.get('content', ''),
                'course': row.get('course', '')
            })
            
    if not tasks:
        print("All problems already have Deepseek answers.")
        return
        
    print(f"Found {len(tasks)} problems needing Deepseek answers. Starting {max_workers} concurrent workers...")
    
    # 并发执行请求
    results_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_id = {
            executor.submit(generate_answer_with_deepseek, task['content'], task['course'], task['id']): task['id'] 
            for task in tasks
        }
        
        # 获取结果
        for future in as_completed(future_to_id):
            task_id = future_to_id[future]
            try:
                res = future.result()
                if res['answer']:
                    results_map[res['id']] = res['answer']
            except Exception as exc:
                print(f"{task_id} generated an exception: {exc}")
                
    # 更新数据
    updated_count = 0
    for row in data:
        row_id = row.get('id', '')
        if row_id in results_map:
            row['deepseek'] = results_map[row_id]
            updated_count += 1

    # 如果有更新，写回 CSV
    if updated_count > 0:
        print(f"\nSaving {updated_count} new Deepseek answers to {file_path}...")
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print("Save completed!")
    else:
        print("\nNo new answers were generated successfully.")

if __name__ == '__main__':
    dataset_path = '/Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.csv'
    
    if DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE" or not DEEPSEEK_API_KEY:
        print("WARNING: You haven't set your DEEPSEEK_API_KEY in .env file or environment variables.")
    else:
        # 考虑到 API 的速率限制，设置默认并发数为 5，可以根据实际账号额度进行调整
        process_dataset(dataset_path, max_workers=20)