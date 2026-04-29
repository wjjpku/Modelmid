# GPT-4.1-mini Iterative Adversarial Experiment Summary

## Setup

- Generator: `gpt-4.1-mini`
- Optimizer: `gpt-4.1-mini`
- Discriminator: `models/best_classifier_model.pkl` rebuilt locally from `scripts/model_training/train_classifier.py`
- Dataset sampled from: `dataset/training/full_dataset.json`
- Sample size per round: `15`
- Maximum rounds: `8`
- Random seed: `42`

## Round-by-round bypass rates

| Round | Bypass Rate | Fooled as Human |
| --- | ---: | ---: |
| 1 | 0.00% | 0 / 15 |
| 2 | 20.00% | 3 / 15 |
| 3 | 46.67% | 7 / 15 |
| 4 | 60.00% | 9 / 15 |
| 5 | 100.00% | 15 / 15 |

## Result

The iterative, data-driven prompt optimization process reached `100%` bypass on round 5.

## Artifacts

- History JSON: `iterative_adversarial_experiment/data/gpt41mini_data_driven_stealth_history.json`
- Runner script: `iterative_adversarial_experiment/scripts/run_iterative_stealth_gpt41mini.py`

## Note

This rerun keeps the iterative stealth logic and discriminator analysis from the existing project, but uses OpenAI tooling for both generation and prompt optimization so the experiment can run with `OPENAI_API_KEY` alone.
