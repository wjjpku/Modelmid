import os
import sys
import json
import random
import pickle
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_classifier import TextFeatureExtractor, DenseTransformer

# Setup API
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY")

def call_deepseek(system_prompt, user_prompt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {
        "model": "deepseek-chat", 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ], 
        "temperature": 0.7, 
        "max_tokens": 2048
    }
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def call_kimi(system_prompt, user_prompt):
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KIMI_API_KEY}"}
    data = {
        "model": "moonshot-v1-8k", 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ], 
        "temperature": 0.5, 
        "max_tokens": 2048
    }
    res = requests.post(url, headers=headers, json=data, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

def run_iterative_experiment(num_rounds=8, sample_size=15):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    model_path = os.path.join(base_dir, 'models/best_classifier_model.pkl')
    dataset_path = os.path.join(base_dir, 'dataset/full_dataset.json')
    history_path = os.path.join(base_dir, 'iterative_adversarial_experiment/data/kimi_data_driven_stealth_history.json')
    
    print("Loading model and dataset...")
    with open(model_path, 'rb') as f:
        classifier_model = pickle.load(f)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        
    # Initial Prompt (A simple baseline to start with)
    current_prompt = (
        "You are an expert mathematician solving a proof. Strictly write your solution to mimic natural human handwriting habits, "
        "avoiding typical AI structural fingerprints. Solve the problem directly in English using standard LaTeX syntax."
    )
    
    history = []
    
    # Meta-Optimizer Prompt
    optimizer_system_prompt = (
        "You are a strict, DATA-DRIVEN Prompt Optimizer. Your goal is to rewrite a 'System Prompt' to help the Kimi AI generator evade an AI-text detector.\n"
        "CRITICAL RULE: You MUST base your modifications EXACTLY and ONLY on the numerical feature feedback provided below.\n"
        "DO NOT rely on your prior assumptions about 'human-like' writing. Specifically, DO NOT instruct the AI to use slang, typos, 'umm', 'wait', 'hmm', or conversational tangents.\n"
        "If the feedback says 'paragraphs are too high', add a strict constraint like 'Write exactly 1-2 paragraphs'.\n"
        "If 'inline math is too high', add 'Limit inline math expressions to a maximum of 5'.\n"
        "Focus purely on structural and numerical adjustments (word counts, paragraph counts, specific forbidden words).\n"
        "Output ONLY the new, improved System Prompt. Do not include any other text, reasoning, or explanations."
    )
    
    best_bypass_rate = 0.0
    patience = 3
    patience_counter = 0
    
    for round_idx in range(num_rounds):
        print(f"\n" + "="*50)
        print(f"--- Round {round_idx + 1}/{num_rounds} ---")
        print(f"Current System Prompt:\n{current_prompt}\n")
        
        # Sample questions randomly
        sample_questions = random.sample(full_data, min(sample_size, len(full_data)))
        
        # Generation using Kimi
        print(f"Generating {len(sample_questions)} answers using Kimi...")
        results = []
        lock = threading.Lock()
        
        def process_q(row):
            prompt = f"Please solve the following math problem:\n\n{row['problem']}"
            try:
                ans = call_kimi(current_prompt, prompt)
                with lock:
                    results.append({'text': ans, 'true_label': 'Kimi'})
            except Exception as e:
                print(f"API Error for ID {row.get('id', 'unknown')}: {e}")
                
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(process_q, sample_questions))
            
        if not results:
            print("Failed to generate any answers. Aborting round.")
            break
            
        df = pd.DataFrame(results)
        
        # Evaluation
        X = pd.DataFrame({'text': df['text']})
        print("Evaluating generated texts with classifier...")
        y_pred = classifier_model.predict(X)
        df['predicted_label'] = y_pred
        
        # Extract features to understand failure reasons
        extractor = TextFeatureExtractor()
        features_df = extractor.transform(X['text'])
        features_df['predicted_label'] = y_pred
        
        fooled_count = len(df[df['predicted_label'] == 'Human'])
        bypass_rate = fooled_count / len(df)
        print(f"Result: {fooled_count}/{len(df)} evaded detection (Bypass Rate: {bypass_rate*100:.2f}%)")
        
        failed_df = features_df[features_df['predicted_label'] != 'Human']
        avg_failed_features = {}
        
        if len(failed_df) == 0:
            print("100% Bypass achieved! No failed samples to analyze.")
            history.append({
                'round': round_idx + 1,
                'prompt': current_prompt,
                'bypass_rate': bypass_rate,
                'failed_features': {}
            })
            break
        else:
            avg_failed_features = failed_df.mean(numeric_only=True).to_dict()
            
            # Construct feedback for Optimizer
            feedback_prompt = (
                f"Current System Prompt:\n{current_prompt}\n\n"
                f"Bypass Rate: {bypass_rate*100:.2f}%\n"
                f"STRICT NUMERICAL FEEDBACK (Failed generated texts vs Human Baseline):\n"
                f"- Paragraphs: Generated {avg_failed_features.get('num_paragraphs', 0):.2f} vs Human Target ~1.5 (Action: Constrain paragraph count)\n"
                f"- Inline Math Count: Generated {avg_failed_features.get('inline_math_count', 0):.2f} vs Human Target ~5.0 (Action: Restrict inline math symbol usage)\n"
                f"- Declarative Density (we/let/suppose): Generated {avg_failed_features.get('declarative_density', 0):.2f} vs Human Target ~2.0 (Action: Ban these specific words)\n"
                f"- Logical Words Density (because/therefore): Generated {avg_failed_features.get('logical_words_density', 0):.2f} vs Human Target ~1.0 (Action: Ban logical connectors)\n"
                f"- Transition Words (firstly/moreover): Generated {avg_failed_features.get('transition_words_density', 0):.2f} vs Human Target ~0.0 (Action: Completely ban transition words)\n"
                f"- List Items Count: Generated {avg_failed_features.get('num_list_items', 0):.2f} vs Human Target ~0.0 (Action: Completely ban lists/bullet points)\n\n"
                "Task: Rewrite the Current System Prompt. Keep the effective rules, but add or strengthen STRICT formatting and word-ban constraints to fix exactly the numerical discrepancies above.\n"
                "DO NOT add unstructured conversational advice. Output ONLY the new prompt text."
            )
            
            # Early stopping logic
            if bypass_rate > best_bypass_rate:
                best_bypass_rate = bypass_rate
                patience_counter = 0
            else:
                patience_counter += 1
                
            history.append({
                'round': round_idx + 1,
                'prompt': current_prompt,
                'bypass_rate': bypass_rate,
                'failed_features': avg_failed_features
            })
            
            # Save history iteratively
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
                
            if bypass_rate >= 0.8:
                print("Target bypass rate (>=80%) achieved! Stopping early.")
                break
                
            if patience_counter >= patience:
                print(f"Early stopping triggered: No improvement for {patience} rounds. Best bypass rate was {best_bypass_rate*100:.2f}%.")
                break
            
            # Optimize Prompt
            print("Optimizing prompt based on feedback...")
            try:
                new_prompt = call_deepseek(optimizer_system_prompt, feedback_prompt)
                current_prompt = new_prompt
                print(f"New Prompt for next round:\n{current_prompt[:150]}...\n")
            except Exception as e:
                print(f"Optimizer API Error: {e}")
                print("Stopping experiment due to optimizer error.")
                break
            
    print(f"\nExperiment Complete! Best Bypass Rate: {best_bypass_rate*100:.2f}%")

if __name__ == '__main__':
    # Run Kimi experiment with a max of 8 rounds and 15 samples per round
    run_iterative_experiment(num_rounds=8, sample_size=15)