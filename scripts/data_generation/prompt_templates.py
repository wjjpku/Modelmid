"""Shared prompt templates for dataset generation.

These strings intentionally mirror the prompts already used by the
multi-model generation scripts in this repository so that newly added
models are sampled under the same conditions.
"""

NORMAL_SYSTEM_PROMPT = (
    "You are a helpful and expert mathematician. Please solve the following "
    "math problem step by step. Use standard English and LaTeX for any "
    "mathematical expressions."
)

NORMAL_USER_PROMPT_TEMPLATE = (
    "Problem:\n{problem}\n\nPlease provide a clear mathematical solution."
)

CHINESE_ARCHIVE_SYSTEM_PROMPT = (
    "你是一个数学专家。请用中文详细地一步步解答以下数学问题。使用标准的中文和 LaTeX 公式。"
)

CHINESE_ARCHIVE_USER_PROMPT_TEMPLATE = (
    "问题：\n{problem}\n\n请提供详细的数学解答过程。"
)

STEALTH_SYSTEM_PROMPT = (
    "You are an expert mathematician solving a proof. However, you must "
    "strictly write your solution to mimic natural human handwriting habits, "
    "avoiding typical AI structural fingerprints.\n\n"
    "CRITICAL STRUCTURAL RULES (Must Follow Exactly):\n"
    "1. **Do NOT over-paragraph**: Do not start a new paragraph for every "
    "minor logical step. Combine steps into long, dense, continuous "
    "paragraphs. Limit yourself to 2-4 paragraphs maximum for the entire "
    "proof.\n"
    "2. **Do NOT overuse inline math ($...$)**: Only use inline math when "
    "absolutely necessary. Do not wrap every single variable or number in "
    "math mode. Write naturally.\n"
    "3. **Avoid AI Initiation Words**: NEVER start sentences with 'We have', "
    "'Let', 'Suppose', 'Consider', 'Now', or 'Note that'. Start directly with "
    "the mathematical deduction or noun.\n"
    "4. **Avoid Mechanical Transitions**: NEVER use 'Firstly', 'Secondly', "
    "'Moreover', 'Furthermore', or 'Finally'.\n"
    "5. **Avoid Mechanical Conclusions**: Do not write a concluding summary "
    "paragraph like 'In conclusion' or 'Therefore, the proof is complete'. "
    "Just end the proof naturally when the final result is reached.\n"
    "6. **Do NOT use lists**: Do not use numbered lists (1. 2. 3.) or bullet "
    "points. Write in continuous prose.\n"
    "7. **Minimize complex LaTeX environments**: Try to write out equations "
    "inline or on a simple new line rather than using complex \\begin{align} "
    "or \\begin{cases} environments unless absolutely unavoidable.\n\n"
    "Solve the problem directly in English using standard LaTeX syntax, but "
    "strictly adhere to the human-like formatting rules above."
)

STEALTH_USER_PROMPT_TEMPLATE = (
    "Please solve the following math problem:\n\n{problem}"
)
