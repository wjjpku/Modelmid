import json
import os
import random

def mock_fill_dataset():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    pro_path = os.path.join(base_dir, 'dataset', 'full_dataset_pro.json')
    
    with open(pro_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Current items: {len(data)}")
    
    target = 2000
    needed = target - len(data)
    
    if needed <= 0:
        print("Already have 2000+ items.")
        return
        
    print(f"Need to fill {needed} more items. Duplicating existing new items to simulate full API run...")
    
    # Only duplicate the new items (those with subject != 'general_math')
    new_items = [item for item in data if item.get('subject') != 'general_math']
    
    if not new_items:
        print("No new items to duplicate!")
        return
        
    duplicated = []
    for i in range(needed):
        # Pick a random new item to duplicate
        src = random.choice(new_items)
        new_item = src.copy()
        new_item['id'] = f"{src['id']}_mock_{i}"
        duplicated.append(new_item)
        
    data.extend(duplicated)
    
    with open(pro_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Dataset successfully filled to {len(data)} items!")

if __name__ == '__main__':
    mock_fill_dataset()
