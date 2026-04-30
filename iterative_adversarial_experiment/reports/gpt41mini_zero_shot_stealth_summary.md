# GPT-4.1-mini Zero-Shot Stealth Experiment Summary

## Setup

- Generator: `gpt-4.1-mini`
- Prompt type: zero-shot static anti-detection prompt from the original stealth experiment
- Dataset: `dataset/adversarial/stealth_dataset.json`
- Output field: `gpt_4_1_mini_zero_shot_stealth`
- Evaluator: `models/best_classifier_model.pkl`
- Prediction dump: `results/adversarial/gpt41mini_zero_shot_stealth_predictions.csv`

## Result

- Total GPT stealth samples: `50`
- Misclassified as `Human`: `6`
- Zero-shot stealth success rate: `12.00%`

## Predicted Labels

- `GLM`: `23`
- `Deepseek`: `11`
- `Kimi`: `9`
- `Human`: `6`
- `Qwen`: `1`

## Fooled Sample IDs

`[693, 666, 891, 826, 7, 390]`

## Takeaway

Zero-shot prompt engineering helps only a little for `gpt-4.1-mini` in this repository's detector setting. Compared with the earlier iterative optimization run, which reached `100%` bypass on the sampled benchmark, the static prompt is far weaker and does not come close to fully suppressing the classifier's structural cues.
