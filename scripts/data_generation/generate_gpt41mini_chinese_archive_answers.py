import argparse
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from prompt_templates import (
    CHINESE_ARCHIVE_SYSTEM_PROMPT,
    CHINESE_ARCHIVE_USER_PROMPT_TEMPLATE,
)


MODEL_NAME = "gpt-4.1-mini"
FIELD_NAME = "gpt_4_1_mini"


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


def generate_answer(
    row_id,
    problem: str,
    api_key: str,
    rate_limiter: RateLimiter,
    max_retries: int = 5,
) -> str:
    client = OpenAI(api_key=api_key)
    user_prompt = CHINESE_ARCHIVE_USER_PROMPT_TEMPLATE.format(problem=problem)

    for attempt in range(max_retries):
        request_started = time.monotonic()
        try:
            log(f"ID {row_id}: waiting for rate limiter before attempt {attempt + 1}/{max_retries}")
            rate_limiter.acquire()
            log(f"ID {row_id}: sending request attempt {attempt + 1}/{max_retries} (prompt_chars={len(user_prompt)})")
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": CHINESE_ARCHIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_output_tokens=2048,
            )
            answer = (response.output_text or "").strip()
            if answer:
                elapsed = time.monotonic() - request_started
                log(
                    f"ID {row_id}: request succeeded on attempt {attempt + 1}/{max_retries} "
                    f"in {elapsed:.2f}s (answer_chars={len(answer)})"
                )
                return answer
            raise ValueError("Empty response from OpenAI Responses API")
        except Exception as exc:
            if attempt == max_retries - 1:
                elapsed = time.monotonic() - request_started
                log(
                    f"ID {row_id}: final failure on attempt {attempt + 1}/{max_retries} "
                    f"after {elapsed:.2f}s: {exc}"
                )
                raise
            backoff = min(30, 2**attempt)
            elapsed = time.monotonic() - request_started
            log(
                f"ID {row_id}: attempt {attempt + 1}/{max_retries} failed after {elapsed:.2f}s: {exc}. "
                f"Retrying in {backoff:.2f}s"
            )
            time.sleep(backoff)

    return ""


def process_dataset(
    dataset_path: Path,
    api_key: str,
    max_workers: int,
    max_items: int | None,
    max_rps: int,
) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = [
        {"id": row.get("id"), "problem": row.get("problem", "")}
        for row in data
        if not str(row.get(FIELD_NAME, "")).strip()
    ]
    if max_items is not None:
        tasks = tasks[:max_items]

    if not tasks:
        print(f"All rows in {dataset_path} already contain '{FIELD_NAME}'.")
        return

    lock = threading.Lock()
    completed = 0
    rate_limiter = RateLimiter(max_rps)

    def save_answer(row_id, answer):
        nonlocal completed
        with lock:
            for row in data:
                if row.get("id") == row_id:
                    row[FIELD_NAME] = answer
                    break
            with dataset_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            completed += 1
            log(f"Saved {FIELD_NAME} for ID {row_id} ({completed}/{len(tasks)})")

    log(
        f"Generating {FIELD_NAME} for {len(tasks)} rows in {dataset_path} "
        f"with {max_workers} workers and a global limit of {max_rps} req/s."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_answer, task["id"], task["problem"], api_key, rate_limiter): task["id"]
            for task in tasks
        }
        for future in as_completed(future_map):
            row_id = future_map[future]
            try:
                answer = future.result()
                if answer:
                    save_answer(row_id, answer)
            except Exception as exc:
                log(f"Failed on ID {row_id}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill gpt_4_1_mini answers into the Chinese archive generalization dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "generalization"
            / "test_100_chinese_archive_questions.json"
        ),
        help="Path to the Chinese archive dataset JSON file to update.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker count.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for debugging or partial runs.")
    parser.add_argument("--max-rps", type=int, default=10, help="Global request-rate cap across all threads.")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    process_dataset(Path(args.dataset), api_key, args.workers, args.max_items, args.max_rps)


if __name__ == "__main__":
    main()
