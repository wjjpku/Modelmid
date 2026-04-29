import argparse
import json
import os
import pickle
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_TRAINING_DIR = BASE_DIR / "scripts" / "model_training"
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.append(str(MODEL_TRAINING_DIR))

from train_classifier import DenseTransformer, TextFeatureExtractor  # noqa: E402,F401


GENERATOR_MODEL = "gpt-4.1-mini"
OPENAI_MAX_RPS = 10


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    print(f"[{timestamp}][{thread_name}] {message}", flush=True)


class RateLimiter:
    def __init__(self, max_requests_per_second: int):
        self.max_requests_per_second = max_requests_per_second
        self.timestamps = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= 1.0:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_requests_per_second:
                    self.timestamps.append(now)
                    return
                sleep_for = max(0.01, 1.0 - (now - self.timestamps[0]))
            time.sleep(sleep_for)


def load_local_env() -> dict[str, str]:
    env_values: dict[str, str] = {}
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip().strip('"').strip("'")
    return env_values


def get_env_var(name: str, loaded_env: dict[str, str]) -> str | None:
    return os.environ.get(name) or loaded_env.get(name)


def call_openai(
    client: OpenAI,
    rate_limiter: RateLimiter,
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 5,
) -> str:
    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            rate_limiter.acquire()
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_output_tokens=2048,
            )
            text = (response.output_text or "").strip()
            if text:
                elapsed = time.monotonic() - started
                log(f"OpenAI call succeeded in {elapsed:.2f}s on attempt {attempt + 1}/{max_retries}")
                return text
            raise ValueError("Empty response from OpenAI")
        except Exception as exc:
            elapsed = time.monotonic() - started
            if attempt == max_retries - 1:
                log(f"OpenAI call failed permanently after {elapsed:.2f}s: {exc}")
                raise
            backoff = min(30, 2**attempt)
            log(f"OpenAI call failed after {elapsed:.2f}s: {exc}. Retrying in {backoff:.2f}s")
            time.sleep(backoff)
    return ""


def call_deepseek(system_prompt: str, user_prompt: str, api_key: str, max_retries: int = 5) -> str:
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
            elapsed = time.monotonic() - started
            log(f"DeepSeek optimizer call succeeded in {elapsed:.2f}s on attempt {attempt + 1}/{max_retries}")
            return text
        except Exception as exc:
            elapsed = time.monotonic() - started
            if attempt == max_retries - 1:
                log(f"DeepSeek optimizer call failed permanently after {elapsed:.2f}s: {exc}")
                raise
            backoff = min(30, 2**attempt)
            log(f"DeepSeek optimizer call failed after {elapsed:.2f}s: {exc}. Retrying in {backoff:.2f}s")
            time.sleep(backoff)
    return ""


def run_iterative_experiment(
    num_rounds: int,
    sample_size: int,
    workers: int,
    seed: int,
    optimizer: str,
) -> Path:
    loaded_env = load_local_env()
    openai_key = get_env_var("OPENAI_API_KEY", loaded_env)
    deepseek_key = get_env_var("DEEPSEEK_API_KEY", loaded_env)
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not available.")
    if optimizer == "deepseek" and not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not available for optimizer=deepseek.")

    model_path = BASE_DIR / "models" / "best_classifier_model.pkl"
    dataset_path = BASE_DIR / "dataset" / "training" / "full_dataset.json"
    history_path = (
        BASE_DIR
        / "iterative_adversarial_experiment"
        / "data"
        / "gpt41mini_data_driven_stealth_history.json"
    )

    log(f"Loading discriminator from {model_path}")
    with model_path.open("rb") as f:
        classifier_model = pickle.load(f)
    with dataset_path.open("r", encoding="utf-8") as f:
        full_data = json.load(f)

    random.seed(seed)
    openai_client = OpenAI(api_key=openai_key)
    openai_rate_limiter = RateLimiter(OPENAI_MAX_RPS)

    current_prompt = (
        "You are an expert mathematician solving a proof. Strictly write your solution to mimic natural human handwriting habits, "
        "avoiding typical AI structural fingerprints. Solve the problem directly in English using standard LaTeX syntax."
    )

    optimizer_system_prompt = (
        "You are a strict, DATA-DRIVEN Prompt Optimizer. Your goal is to rewrite a 'System Prompt' to help the GPT-4.1-mini generator evade an AI-text detector.\n"
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
    history: list[dict] = []

    for round_idx in range(num_rounds):
        log("=" * 60)
        log(f"Round {round_idx + 1}/{num_rounds}")
        log(f"Current system prompt: {current_prompt}")

        sample_questions = random.sample(full_data, min(sample_size, len(full_data)))
        results: list[dict] = []
        lock = threading.Lock()

        def process_question(row: dict) -> None:
            user_prompt = f"Please solve the following math problem:\n\n{row['problem']}"
            try:
                answer = call_openai(
                    openai_client,
                    openai_rate_limiter,
                    current_prompt,
                    user_prompt,
                    GENERATOR_MODEL,
                )
                with lock:
                    results.append(
                        {
                            "id": row["id"],
                            "text": answer,
                            "true_label": "GPT41Mini",
                        }
                    )
            except Exception as exc:
                log(f"Generator error for ID {row.get('id')}: {exc}")

        log(f"Generating {len(sample_questions)} answers using {GENERATOR_MODEL}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(process_question, sample_questions))

        if not results:
            log("No texts were generated. Aborting experiment.")
            break

        df = pd.DataFrame(results)
        x_eval = pd.DataFrame({"text": df["text"]})
        y_pred = classifier_model.predict(x_eval)
        df["predicted_label"] = y_pred

        fooled_count = int((df["predicted_label"] == "Human").sum())
        bypass_rate = fooled_count / len(df)
        log(f"Bypass result: {fooled_count}/{len(df)} misclassified as Human ({bypass_rate * 100:.2f}%)")

        extractor = TextFeatureExtractor()
        features_df = extractor.transform(x_eval["text"])
        features_df["predicted_label"] = y_pred
        failed_df = features_df[features_df["predicted_label"] != "Human"]

        round_record = {
            "round": round_idx + 1,
            "generator_model": GENERATOR_MODEL,
            "optimizer": optimizer,
            "sample_size": len(df),
            "prompt": current_prompt,
            "bypass_rate": bypass_rate,
            "fooled_count": fooled_count,
            "predicted_label_counts": df["predicted_label"].value_counts().to_dict(),
            "sample_ids": df["id"].tolist(),
        }

        if failed_df.empty:
            if bypass_rate > best_bypass_rate:
                best_bypass_rate = bypass_rate
            log("100% bypass achieved. Stopping.")
            round_record["failed_features"] = {}
            history.append(round_record)
            history_path.write_text(json.dumps(history, ensure_ascii=False, indent=4), encoding="utf-8")
            break

        avg_failed_features = failed_df.mean(numeric_only=True).to_dict()
        round_record["failed_features"] = avg_failed_features
        history.append(round_record)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=4), encoding="utf-8")

        feedback_prompt = (
            f"Current System Prompt:\n{current_prompt}\n\n"
            f"Bypass Rate: {bypass_rate*100:.2f}%\n"
            "STRICT NUMERICAL FEEDBACK (Failed generated texts vs Human Baseline):\n"
            f"- Paragraphs: Generated {avg_failed_features.get('num_paragraphs', 0):.2f} vs Human Target ~1.5 (Action: Constrain paragraph count)\n"
            f"- Inline Math Count: Generated {avg_failed_features.get('inline_math_count', 0):.2f} vs Human Target ~5.0 (Action: Restrict inline math symbol usage)\n"
            f"- Declarative Density (we/let/suppose): Generated {avg_failed_features.get('declarative_density', 0):.2f} vs Human Target ~2.0 (Action: Ban these specific words)\n"
            f"- Logical Words Density (because/therefore): Generated {avg_failed_features.get('logical_words_density', 0):.2f} vs Human Target ~1.0 (Action: Ban logical connectors)\n"
            f"- Transition Words (firstly/moreover): Generated {avg_failed_features.get('transition_words_density', 0):.2f} vs Human Target ~0.0 (Action: Completely ban transition words)\n"
            f"- List Items Count: Generated {avg_failed_features.get('num_list_items', 0):.2f} vs Human Target ~0.0 (Action: Completely ban lists/bullet points)\n\n"
            "Task: Rewrite the Current System Prompt. Keep the effective rules, but add or strengthen STRICT formatting and word-ban constraints to fix exactly the numerical discrepancies above.\n"
            "DO NOT add unstructured conversational advice. Output ONLY the new prompt text."
        )

        if bypass_rate > best_bypass_rate:
            best_bypass_rate = bypass_rate
            patience_counter = 0
        else:
            patience_counter += 1

        if bypass_rate >= 0.8:
            log("Target bypass rate >=80% achieved. Stopping early.")
            break
        if patience_counter >= patience:
            log(f"Early stopping: no improvement for {patience} rounds. Best bypass = {best_bypass_rate*100:.2f}%")
            break

        log("Optimizing prompt for next round")
        if optimizer == "deepseek":
            current_prompt = call_deepseek(optimizer_system_prompt, feedback_prompt, deepseek_key)
        else:
            current_prompt = call_openai(
                openai_client,
                openai_rate_limiter,
                optimizer_system_prompt,
                feedback_prompt,
                GENERATOR_MODEL,
            )

    log(f"Experiment complete. Best bypass rate = {best_bypass_rate*100:.2f}%")
    log(f"History saved to {history_path}")
    return history_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the iterative stealth experiment with GPT-4.1-mini as generator.")
    parser.add_argument("--rounds", type=int, default=8, help="Maximum number of optimization rounds.")
    parser.add_argument("--sample-size", type=int, default=15, help="Questions sampled per round.")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent generator calls.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling.")
    parser.add_argument(
        "--optimizer",
        choices=["openai", "deepseek"],
        default="openai",
        help="Prompt optimizer backend. Defaults to OpenAI so the experiment can run with OPENAI_API_KEY only.",
    )
    args = parser.parse_args()

    run_iterative_experiment(
        num_rounds=args.rounds,
        sample_size=args.sample_size,
        workers=args.workers,
        seed=args.seed,
        optimizer=args.optimizer,
    )


if __name__ == "__main__":
    main()
