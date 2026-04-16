import os
import json
from datasets import load_dataset

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')'
    json_path = os.path.join(base_dir, 'dataset', 'full_dataset.json')
    
    all_data = []
    current_id = 1
    
    print("Loading 1000 pure English items from math-ai/StackMathQA...")
    ds = load_dataset("math-ai/StackMathQA", "stackmathqa100k")
    train_ds = ds['train']
    
    added_count = 0
    for i in range(len(train_ds)):
        item = train_ds[i]
        q = item.get('Q', '').strip()
        a = item.get('A', '').strip()
        
        if q and a:
            all_data.append({
                'id': current_id,
                'problem': q,
                'human': a,
                'deepseek': '',
                'kimi': ''
            })
            current_id += 1
            added_count += 1
            
        if added_count >= 1000:
            break
            
    print(f"Added {added_count} new pure English records.")
    
    # 覆盖写入 JSON，抛弃原有的中文数据
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(all_data)} records to {json_path}.")

if __name__ == '__main__':
    main()