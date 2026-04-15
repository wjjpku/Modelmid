import os
import csv
import re

def process_ode(file_path):
    data = []
    prefix_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    enum_depth = 0
    current_item = []
    
    in_document = False
    
    for line in lines:
        stripped = line.strip()
        
        if not in_document:
            if stripped == r'\begin{document}':
                in_document = True
            continue
            
        if stripped == r'\end{document}':
            break
            
        if stripped.startswith(r'\section{'):
            prefix_text = []
            continue
            
        if stripped.startswith(r'\subsection*{'):
            prefix_text = []
            continue
            
        if stripped.startswith(r'\begin{enumerate}'):
            enum_depth += 1
            if enum_depth == 1:
                pass
            else:
                current_item.append(line)
            continue
            
        if stripped.startswith(r'\end{enumerate}'):
            enum_depth -= 1
            if enum_depth == 0:
                if current_item:
                    prefix_str = "\n".join(prefix_text).strip()
                    item_str = "".join(current_item).strip()
                    prob_content = f"{prefix_str}\n{item_str}" if prefix_str else item_str
                    
                    data.append({
                        'id': f'ode_{len(data)+1}',
                        'course': '常微分方程',
                        'content': prob_content.strip()
                    })
                    current_item = []
                prefix_text = [] 
            else:
                current_item.append(line)
            continue
            
        if stripped.startswith(r'\item'):
            if enum_depth == 1:
                if current_item:
                    prefix_str = "\n".join(prefix_text).strip()
                    item_str = "".join(current_item).strip()
                    prob_content = f"{prefix_str}\n{item_str}" if prefix_str else item_str
                    
                    data.append({
                        'id': f'ode_{len(data)+1}',
                        'course': '常微分方程',
                        'content': prob_content.strip()
                    })
                
                item_content = line[line.find(r'\item')+5:].lstrip()
                current_item = [item_content] if item_content.strip() else []
            else:
                current_item.append(line)
            continue
            
        if enum_depth == 0:
            if stripped and not stripped.startswith('\\'):
                prefix_text.append(stripped)
        else:
            current_item.append(line)
                
    return data

def process_algebra(file_path):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_item = []
    
    in_document = False
    
    for i, line in enumerate(lines):
        if i >= 554: # Stop after line 554 as requested
            break
            
        stripped = line.strip()
        
        if not in_document:
            if stripped == r'\begin{document}':
                in_document = True
            continue
            
        if stripped.startswith(r'\section{'):
            continue
            
        if stripped.startswith(r'\subsection{'):
            continue
            
        # Match problem start: "1.1.1. " or "1.2.7*. "
        match = re.match(r'^(\d+\.\d+\.\d+\*?)\.\s*(.*)', stripped)
        if match:
            if current_item:
                data.append({
                    'id': f'algebra_{len(data)+1}',
                    'course': '近世代数',
                    'content': "".join(current_item).strip()
                })
            item_content = match.group(2)
            current_item = [item_content + "\n"]
        else:
            if current_item:
                # ignore \newpage
                if stripped != r'\newpage':
                    current_item.append(line)
                
    if current_item:
        data.append({
            'id': f'algebra_{len(data)+1}',
            'course': '近世代数',
            'content': "".join(current_item).strip()
        })
        
    return data

def main():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'processed_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ode_file = os.path.join(data_dir, 'ode题目latex代码')
    algebra_file = os.path.join(data_dir, '近世代数群论部分题目及解答latex代码（前554行为题目）')
    
    all_data = []
    
    if os.path.exists(ode_file):
        print(f"Processing {ode_file}...")
        all_data.extend(process_ode(ode_file))
        
    if os.path.exists(algebra_file):
        print(f"Processing {algebra_file}...")
        all_data.extend(process_algebra(algebra_file))
        
    output_csv = os.path.join(output_dir, 'processed_data.csv')
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'course', 'content'])
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
            
    print(f"Successfully processed {len(all_data)} problems.")
    print(f"Data saved to {output_csv}")

if __name__ == '__main__':
    main()