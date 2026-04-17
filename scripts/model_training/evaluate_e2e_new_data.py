import os
import sys
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

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

class MathTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_e2e_new_data():
    output_path = os.path.join(base_dir, 'dataset', 'test_100_new_questions.json')
    original_dataset_path = os.path.join(base_dir, 'dataset', 'full_dataset_pro.json')
    
    print(f"Loading original dataset from {original_dataset_path} to avoid overlapping...")
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    existing_ids = {str(item['id']) for item in original_data}
    new_data = []
    
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, loading...")
        with open(output_path, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
    
    target_new = 100
    already_generated = len(new_data)
    need_to_generate = target_new - already_generated
    
    if need_to_generate > 0:
        print(f"Need to generate {need_to_generate} new items.")
        subjects = ["algebra", "counting_and_probability", "geometry", "number_theory", "precalculus"]
        samples_per_subject = need_to_generate // len(subjects) + 1
        
        selected_samples = []
        
        for subject in subjects:
            print(f"Loading subset: {subject}...")
            dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
            dataset = dataset.shuffle(seed=123)  # Use a different seed
            count = 0
            for i in range(len(dataset)):
                q_id = f"math_{subject}_test_{i}"
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
                    
        selected_samples = selected_samples[:need_to_generate]
        print(f"Selected {len(selected_samples)} new test questions.")
        
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
            
            with lock:
                new_data.append(result)
                if len(new_data) % 5 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=4)
            print(f"✅ Completed {q_id} [{len(new_data)}/{target_new}]")
                
        print(f"Starting concurrent API calls...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(process_question, row) for row in selected_samples]
            for future in as_completed(futures):
                future.result()
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
    else:
        print("Data already generated.")
        
    print("\n--- Testing Model on New Data ---")
    
    # Format to DataFrame
    records = []
    for row in new_data:
        q_id = row['id']
        if row.get('human'): records.append({'id': q_id, 'text': row['human'], 'label': 'Human'})
        if row.get('deepseek'): records.append({'id': q_id, 'text': row['deepseek'], 'label': 'Deepseek'})
        if row.get('kimi'): records.append({'id': q_id, 'text': row['kimi'], 'label': 'Kimi'})
        if row.get('glm'): records.append({'id': q_id, 'text': row['glm'], 'label': 'GLM'})
        if row.get('qwen'): records.append({'id': q_id, 'text': row['qwen'], 'label': 'Qwen'})
        
    df = pd.DataFrame(records)
    print(f"Total test records: {len(df)}")
    
    le = LabelEncoder()
    le.fit(['Deepseek', 'GLM', 'Human', 'Kimi', 'Qwen'])
    test_labels = le.transform(df['label'].values)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_path = os.path.join(base_dir, 'models', 'e2e_transformer_best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    test_dataset = MathTextDataset(df['text'].values, test_labels, tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    final_preds = []
    final_true = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            outputs = model(b_input_ids, attention_mask=b_attn_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            
            final_preds.extend(preds.cpu().numpy())
            final_true.extend(b_labels.cpu().numpy())
            
    acc = accuracy_score(final_true, final_preds)
    print(f"\nFinal Test Accuracy on 100 new questions (500 records): {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(final_true, final_preds, target_names=le.classes_))

if __name__ == '__main__':
    evaluate_e2e_new_data()
