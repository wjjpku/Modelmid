import os
import csv
import re

def process_algebra(file_path):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 我们从包含题目和答案的部分开始解析
        lines = f.readlines()[554:]
        
    current_id = None
    current_content = []
    current_answer = []
    in_answer = False
    
    # 匹配题号，例如："1.1.1. 设 \(B\)..."
    prob_re = re.compile(r'^(\d+\.\d+\.\d+\*?)\.\s*(.*)')
    # 匹配解答开头，例如："\textbf{证} ..."
    ans_re = re.compile(r'^\s*\\textbf\{(证|解|注)\}\s*(.*)')
    
    def clean_answer(ans_lines):
        """清洗解答内容，去掉开头或者中间残留的 \textbf{证} 等个人习惯"""
        ans_str = "".join(ans_lines).strip()
        # 将所有的 \textbf{证}、\textbf{解}、\textbf{注} 去掉
        ans_str = re.sub(r'\\textbf\{(证|解|注)\}\s*', '', ans_str)
        return ans_str.strip()
    
    for line in lines:
        stripped = line.strip()
        
        # 如果还没匹配到题目，跳过一些大标题
        if not current_id:
            if stripped.startswith(r'\section') or stripped.startswith(r'\subsection') or stripped.startswith(r'\addcontentsline') or stripped == r'\newpage':
                continue
            
        match_prob = prob_re.match(stripped)
        if match_prob:
            # 当匹配到新题目时，保存上一个题目及它的解答
            if current_id:
                ans_str = clean_answer(current_answer)
                data.append({
                    'id': f'algebra_{current_id}',
                    'course': '近世代数',
                    'content': "".join(current_content).strip(),
                    'human': ans_str,
                    'deepseek': '',
                    'kimi': ''
                })
            
            # 开始记录新题目
            current_id = match_prob.group(1)
            current_content = [match_prob.group(2) + "\n"]
            current_answer = []
            in_answer = False
            continue
            
        match_ans = ans_re.match(stripped)
        if match_ans and current_id:
            in_answer = True
            # 不要存 \textbf{证} 这一行，只存后面的内容
            if match_ans.group(2):
                current_answer.append(match_ans.group(2) + "\n")
            continue
            
        if current_id:
            if in_answer:
                current_answer.append(line)
            else:
                current_content.append(line)

    # 加上最后一题
    if current_id:
        ans_str = clean_answer(current_answer)
        data.append({
            'id': f'algebra_{current_id}',
            'course': '近世代数',
            'content': "".join(current_content).strip(),
            'human': ans_str,
            'deepseek': '',
            'kimi': ''
        })
        
    return data

def main():
    base_dir = '/Users/jiaju/Documents/github/Modelmid'
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'dataset')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    algebra_file = os.path.join(data_dir, '近世代数群论部分题目及解答latex代码（前554行为题目）')
    
    all_data = []
    
    if os.path.exists(algebra_file):
        print(f"Processing {algebra_file}...")
        all_data.extend(process_algebra(algebra_file))
    else:
        print(f"Error: Could not find {algebra_file}")
        
    # 如果文件已经存在，我们需要保留之前的 deepseek 答案！
    output_csv = os.path.join(output_dir, 'full_dataset.csv')
    existing_data = {}
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row['id']] = row

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'course', 'content', 'human', 'deepseek', 'kimi'])
        writer.writeheader()
        for row in all_data:
            # 恢复已有的 deepseek 或 kimi 数据
            if row['id'] in existing_data:
                row['deepseek'] = existing_data[row['id']].get('deepseek', '')
                row['kimi'] = existing_data[row['id']].get('kimi', '')
            writer.writerow(row)
            
    print(f"Successfully processed {len(all_data)} problems.")
    print(f"Data saved to {output_csv}")

if __name__ == '__main__':
    main()