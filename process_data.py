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
                        'content': prob_content.strip(),
                        'answer': '',
                        'source': ''
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
                        'content': prob_content.strip(),
                        'answer': '',
                        'source': ''
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
        # We parse the file starting from line 554 since the second part
        # contains BOTH the questions and their respective answers.
        lines = f.readlines()[554:]
        
    current_id = None
    current_content = []
    current_answer = []
    in_answer = False
    
    prob_re = re.compile(r'^(\d+\.\d+\.\d+\*?)\.\s*(.*)')
    ans_re = re.compile(r'^\s*\\textbf\{(证|解|注)\}\s*(.*)')
    
    for line in lines:
        stripped = line.strip()
        
        # Skip section headers in part 2 if any, but only if we are not inside a problem content/answer
        if not current_id:
            if stripped.startswith(r'\section') or stripped.startswith(r'\subsection') or stripped.startswith(r'\addcontentsline') or stripped == r'\newpage':
                continue
            
        match_prob = prob_re.match(stripped)
        if match_prob:
            if current_id:
                ans_str = "".join(current_answer).strip()
                data.append({
                    'id': f'algebra_{len(data)+1}',
                    'course': '近世代数',
                    'content': "".join(current_content).strip(),
                    'answer': ans_str,
                    'source': 'Human' if ans_str else ''
                })
            
            current_id = match_prob.group(1)
            current_content = [match_prob.group(2) + "\n"]
            current_answer = []
            in_answer = False
            continue
            
        match_ans = ans_re.match(stripped)
        if match_ans and current_id:
            in_answer = True
            current_answer.append(line)
            continue
            
        if current_id:
            if in_answer:
                current_answer.append(line)
            else:
                current_content.append(line)

    if current_id:
        ans_str = "".join(current_answer).strip()
        data.append({
            'id': f'algebra_{len(data)+1}',
            'course': '近世代数',
            'content': "".join(current_content).strip(),
            'answer': ans_str,
            'source': 'Human' if ans_str else ''
        })
        
    return data

def main():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'dataset')
    
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
        
    output_csv = os.path.join(output_dir, 'full_dataset.csv')
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'course', 'content', 'answer', 'source'])
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
            
    print(f"Successfully processed {len(all_data)} problems.")
    print(f"Data saved to {output_csv}")

if __name__ == '__main__':
    main()