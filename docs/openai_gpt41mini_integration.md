# GPT-4.1-mini Data Generation

This repository now includes OpenAI-based dataset generation scripts that use
`OPENAI_API_KEY` and the `gpt-4.1-mini` model.

## Added fields

- `gpt_4_1_mini`: standard answer generated with the same normal prompt used by
  the existing multi-model generation scripts.
- `gpt_4_1_mini_stealth`: adversarial "human-like" answer generated with the
  same stealth prompt family used by the existing anti-detection experiment.

## Scripts

- `scripts/data_generation/generate_gpt41mini_answers.py`
- `scripts/data_generation/generate_gpt41mini_stealth_answers.py`
- `scripts/data_generation/prompt_templates.py`

## Usage

```bash
python scripts/data_generation/generate_gpt41mini_answers.py
python scripts/data_generation/generate_gpt41mini_answers.py --dataset dataset/training/full_dataset_pro.json
python scripts/data_generation/generate_gpt41mini_stealth_answers.py
```

Optional flags:

- `--workers`: concurrent request count.
- `--max-items`: run on only the first N missing rows for testing.

## Important note

The published reports and current classifier scripts in this repository are
still framed around the original 5-source setup (`Human`, `Deepseek`, `Kimi`,
`GLM`, `Qwen`). The new GPT-4.1-mini fields are additive dataset extensions and
do not retroactively change those reported 5-class results.
