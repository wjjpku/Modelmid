import argparse
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from prompt_templates import STEALTH_SYSTEM_PROMPT, STEALTH_USER_PROMPT_TEMPLATE


MODEL_NAME = "gpt-4.1-mini"
FIELD_NAME = "gpt_4_1_mini_stealth"


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


def load_api_key() -> str | None:
    base_dir = Path(__file__).resolve().parents[2]
    env_path = base_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return os.environ.get("OPENAI_API_KEY")


def load_system_prompt(
    system_prompt_file: str | None,
    history_json: str | None,
) -> str:
    if system_prompt_file:
        return Path(system_prompt_file).read_text(encoding="utf-8").strip()

    if history_json:
        history = json.loads(Path(history_json).read_text(encoding="utf-8"))
        if not history:
            raise ValueError(f"No prompt history found in {history_json}")
        return str(history[-1]["prompt"]).strip()

    return STEALTH_SYSTEM_PROMPT


def generate_answer(
    problem: str,
    system_prompt: str,
    api_key: str,
    rate_limiter: RateLimiter,
    max_retries: int = 5,
) -> str:
    client = OpenAI(api_key=api_key)
    user_prompt = STEALTH_USER_PROMPT_TEMPLATE.format(problem=problem)

    for attempt in range(max_retries):
        try:
            rate_limiter.acquire()
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_output_tokens=2048,
            )
            answer = (response.output_text or "").strip()
            if answer:
                return answer
            raise ValueError("Empty response from OpenAI Responses API")
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(30, 2**attempt))

    return ""


def process_dataset(
    dataset_path: Path,
    system_prompt: str,
    api_key: str,
    field_name: str,
    max_workers: int,
    max_items: int | None,
    max_rps: int,
) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = [
        {"id": row.get("id"), "problem": row.get("problem", "")}
        for row in data
        if not str(row.get(field_name, "")).strip()
    ]
    if max_items is not None:
        tasks = tasks[:max_items]

    if not tasks:
        print(f"All rows in {dataset_path} already contain '{field_name}'.")
        return

    lock = threading.Lock()
    completed = 0
    rate_limiter = RateLimiter(max_rps)

    def save_answer(row_id, answer):
        nonlocal completed
        with lock:
            for row in data:
                if row.get("id") == row_id:
                    row[field_name] = answer
                    break
            with dataset_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            completed += 1
            print(f"Saved {field_name} for ID {row_id} ({completed}/{len(tasks)})")

    print(
        f"Generating {field_name} for {len(tasks)} rows in {dataset_path} "
        f"with {max_workers} workers and a global limit of {max_rps} req/s."
    )
    print(f"Using system prompt preview: {system_prompt[:200]}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_answer, task["problem"], system_prompt, api_key, rate_limiter): task["id"]
            for task in tasks
        }
        for future in as_completed(future_map):
            row_id = future_map[future]
            try:
                answer = future.result()
                if answer:
                    save_answer(row_id, answer)
            except Exception as exc:
                print(f"Failed on ID {row_id}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Fill GPT-4.1-mini stealth answers into a dataset field.")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[2] / "dataset" / "adversarial" / "stealth_dataset.json"),
        help="Path to the stealth dataset JSON file to update.",
    )
    parser.add_argument("--workers", type=int, default=5, help="Concurrent worker count.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for debugging or partial runs.")
    parser.add_argument("--max-rps", type=int, default=10, help="Global request-rate cap across all threads.")
    parser.add_argument(
        "--field-name",
        default=FIELD_NAME,
        help="Dataset field to write, e.g. gpt_4_1_mini_stealth or gpt_4_1_mini_zero_shot_stealth.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional UTF-8 text file containing the exact system prompt to use.",
    )
    parser.add_argument(
        "--history-json",
        default=None,
        help="Optional iterative experiment history JSON; the last round prompt will be used.",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    system_prompt = load_system_prompt(args.system_prompt_file, args.history_json)
    process_dataset(
        Path(args.dataset),
        system_prompt,
        api_key,
        args.field_name,
        args.workers,
        args.max_items,
        args.max_rps,
    )


if __name__ == "__main__":
    main()
