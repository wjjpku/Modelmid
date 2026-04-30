# 数学推导文本溯源分类器

本仓库用于研究“数学解答是由谁写的”：`Human / Deepseek / Kimi / GLM / Qwen` 为主线五分类，另补充了 `GPT-4.1-mini` 的数据生成与对抗实验扩展。

项目核心包括：

- 配对构建数学题与多来源解答数据集
- 提取 `TF-IDF + 结构/LaTeX/逻辑风格特征`
- 训练传统机器学习与端到端深度学习分类器
- 评估跨题库、跨学科、跨语言泛化
- 进行零样本与迭代式防检测对抗实验

## 当前目录结构

状态标记：

- `GPT 已补充`：文件中已实际写入 GPT 生成内容
- `GPT 未补充`：当前文件仍只有原五分类或旧实验内容
- `主线使用`：当前实验流程默认会读写这里
- `归档`：保留历史材料，不参与当前主线复现

```text
Model_mid/
├── archive/
│   ├── cache/                         # 归档: __pycache__、.pyc、编译缓存，主线不使用
│   ├── legacy_generated_data/        # 归档: 早期中间 JSON / 旧生成结果，未并入当前主线
│   ├── raw_chinese_sources/          # 归档: 未采用的中文原始题库、LaTeX 素材
│   ├── system_files/                 # 归档: .DS_Store 等系统杂项
│   └── data/                         # 归档: 旧目录名保留，当前基本为空
├── dataset/
│   ├── training/
│   │   ├── full_dataset.json
│   │   │   # 主线使用: 原始 1000 题五分类训练集
│   │   │   # 字段: human / deepseek / kimi / glm / qwen / gpt_4_1_mini
│   │   │   # GPT 已补充: gpt_4_1_mini = 1000/1000（由 full_dataset_pro 前1000题复制）
│   │   └── full_dataset_pro.json
│   │       # 主线使用: 扩容 2000 题训练/泛化数据集
│   │       # 字段: human / deepseek / kimi / glm / qwen / gpt_4_1_mini / subject
│   │       # GPT 已补充: gpt_4_1_mini = 2000/2000
│   ├── generalization/
│   │   ├── test_100_new_questions.json
│   │   │   # 主线使用: 全新英文题库泛化测试集
│   │   │   # 字段: human / deepseek / kimi / glm / qwen / gpt_4_1_mini / subject
│   │   │   # GPT 已补充: gpt_4_1_mini = 100/100（英文 prompt，与其他 LLM 一致）
│   │   └── test_100_chinese_archive_questions.json
│   │       # 主线使用: 中文归档题零样本泛化测试集
│   │       # 字段: human / deepseek / kimi / glm / qwen / gpt_4_1_mini / subject
│   │       # GPT 已补充: gpt_4_1_mini = 100/100（中文 prompt，与其他 LLM 一致）
│   └── adversarial/
│       └── stealth_dataset.json
│           # 主线使用: 静态 stealth / 防检测实验数据
│           # 字段: deepseek_stealth / kimi_stealth / glm_stealth / qwen_stealth
│           # GPT 已补充:
│           #   - gpt_4_1_mini_stealth = 50/50（迭代优化 prompt）
│           #   - gpt_4_1_mini_zero_shot_stealth = 50/50（零样本静态 prompt）
├── docs/
│   ├── figures/                      # 主线使用: PCA、混淆矩阵、小提琴图等可视化输出
│   │   ├── gpt_augmented/            # 已加入 GPT 数据后重新生成的新版图片
│   │   │   # 包含: feature_importances / PCA / 3 张 violin / confusion_matrix_ml
│   │   └── legacy_pre_gpt/           # 历史旧图，仍对应原五分类或旧实验阶段
│   ├── reports/
│   │   ├── comprehensive_evaluation_report.md
│   │   │   # 主线使用: 综合实验汇总报告
│   │   ├── midreport.pdf
│   │   │   # 主线使用: 中期报告 PDF
│   │   └── midreport_full.txt
│   │       # 主线使用: 中期报告纯文本版，便于检索和改写
│   ├── openai_gpt41mini_integration.md
│   │   # GPT 扩展说明: OpenAI / GPT-4.1-mini 接入与生成脚本说明
│   └── task.md
│       # 课程原始任务说明
├── iterative_adversarial_experiment/
│   ├── data/                         # 主线使用: 迭代 prompt 历史、绕过率轨迹、失败特征反馈
│   ├── reports/                      # 主线使用: 对抗实验总结与说明
│   │   # GPT 已补充:
│   │   #   - gpt41mini_adversarial_experiment_summary.md
│   │   #   - gpt41mini_zero_shot_stealth_summary.md
│   └── scripts/                      # 主线使用: Kimi / GPT 的迭代对抗脚本
├── models/                           # 主线使用: 训练好的模型权重/管线
│   # GPT 后已刷新:
│   #   - best_classifier_model.pkl（基于含 GPT 的 6 类训练集重新训练）
│   # 尚未刷新:
│   #   - 当前工作区未提供可直接评估的 DL checkpoint
├── results/
│   ├── classification/               # 主线使用: 主分类、模型对比、消融实验结果
│   │   # 尚未刷新:
│   │   #   - ablation_results.csv
│   │   #   - e2e_dl_results.csv
│   │   #   - ml_vs_dl_comparison.csv
│   │   # 以上 CSV 仍主要对应原五分类主线，不是 GPT 扩充后的新版结果
│   ├── generalization/               # 主线使用: 泛化实验预测结果
│   │   # GPT 后已刷新:
│   │   #   - clean_test_predictions.csv（6 类 clean set，当前仅含 ML 预测列）
│   │   # 尚未刷新:
│   │   #   - generalization_predictions.csv（仍是旧主线结果）
│   └── adversarial/                  # 主线使用: 防检测实验预测结果
│       # GPT 已补充:
│       #   - gpt41mini_zero_shot_stealth_predictions.csv
│       # GPT 部分未单独补齐:
│       #   - 传统 stealth_predictions.csv 默认仍是原四模型主线结果
├── scripts/
│   ├── archive/                      # 归档: 已废弃或一次性探索脚本，路径多数未按新结构维护
│   ├── data_generation/              # 主线使用: 数据生成脚本
│   │   # GPT 已补充:
│   │   #   - generate_gpt41mini_answers.py
│   │   #   - generate_gpt41mini_chinese_archive_answers.py
│   │   #   - generate_gpt41mini_stealth_answers.py
│   │   #   - prompt_templates.py
│   ├── model_training/               # 主线使用: 训练、评估、消融、混淆矩阵脚本
│   │   # GPT 已补充:
│   │   #   - evaluate_stealth.py 支持单独评估 GPT stealth 字段
│   └── visualization/                # 主线使用: 图表与可视化脚本
└── README.md                         # 当前仓库导航与状态说明
```

## 数据说明

### 训练数据

- `dataset/training/full_dataset.json`
  - 主五分类训练集，约 `1000` 题，每题包含 `human / deepseek / kimi / glm / qwen`
  - GPT 状态：已补充 `gpt_4_1_mini`
  - 当前覆盖：`1000 / 1000`
- `dataset/training/full_dataset_pro.json`
  - 扩容后的泛化数据集，含 `subject` 字段
  - GPT 状态：已补充 `gpt_4_1_mini`
  - 当前覆盖：`2000 / 2000`

### 泛化测试数据

- `dataset/generalization/test_100_new_questions.json`
  - 全新英文题库测试集
  - GPT 状态：已补充 `gpt_4_1_mini`
  - 当前覆盖：`100 / 100`
  - Prompt：与其他英文 LLM 一致的 normal prompt
- `dataset/generalization/test_100_chinese_archive_questions.json`
  - 中文归档题零样本测试集
  - GPT 状态：已补充 `gpt_4_1_mini`
  - 当前覆盖：`100 / 100`
  - Prompt：与现有中文 GLM/Qwen 生成脚本一致的中文 prompt

### 对抗数据

- `dataset/adversarial/stealth_dataset.json`
  - 静态 stealth prompt 生成的数据
  - 原四模型字段：
    - `deepseek_stealth`
    - `kimi_stealth`
    - `glm_stealth`
    - `qwen_stealth`
  - GPT 已补充字段：
    - `gpt_4_1_mini_stealth`
      - 覆盖：`50 / 50`
      - 来源：迭代优化后的最终 stealth prompt
    - `gpt_4_1_mini_zero_shot_stealth`
      - 覆盖：`50 / 50`
      - 来源：原始零样本静态 stealth prompt

## GPT 覆盖速览

- 已补充
  - `dataset/training/full_dataset.json`
  - `dataset/training/full_dataset_pro.json`
  - `dataset/generalization/test_100_new_questions.json`
  - `dataset/generalization/test_100_chinese_archive_questions.json`
  - `dataset/adversarial/stealth_dataset.json`
  - `iterative_adversarial_experiment/reports/` 下两份 GPT 总结
  - `results/generalization/clean_test_predictions.csv`
  - `results/adversarial/gpt41mini_zero_shot_stealth_predictions.csv`
- 还未补充
  - `results/generalization/generalization_predictions.csv`
  - `results/classification/` 下主表仍以原五分类论文主线为主

## GPT 加入后已刷新的结果

- `models/best_classifier_model.pkl`
  - 已基于 `dataset/training/full_dataset.json` 的 6 类数据重新训练
- `docs/experiment_report.md`
  - 已按 6 类数据重新生成，报告中包含 `GPT-4.1-mini`
- `docs/figures/gpt_augmented/feature_importances.png`
- `docs/figures/gpt_augmented/violin_num_paragraphs.png`
- `docs/figures/gpt_augmented/violin_inline_math_count.png`
- `docs/figures/gpt_augmented/violin_declarative_density.png`
- `docs/figures/gpt_augmented/pca_clusters_2d.png`
- `docs/figures/gpt_augmented/confusion_matrix_ml.png`
- `results/generalization/clean_test_predictions.csv`
  - 已更新为 6 类 clean set 的 ML 预测结果
  - 由于本机暂未重训 DL，这个文件当前不含 `DL_Prediction` 列

## 仍是旧主线或未刷新结果

- `models/` 中当前没有可直接评估 6 类任务的 DL checkpoint
  - 因此这次只刷新了 ML 结果，没有重跑 DL 混淆矩阵或 DL 泛化预测
- `docs/figures/legacy_pre_gpt/confusion_matrix_dl.png`
  - 仍是旧版本 DL 图
- `results/classification/ablation_results.csv`
- `results/classification/e2e_dl_results.csv`
- `results/classification/ml_vs_dl_comparison.csv`
- `results/generalization/generalization_predictions.csv`
- `results/adversarial/stealth_predictions.csv`

## 结果目录

- `results/classification/`
  - `ablation_results.csv`
  - `e2e_dl_results.csv`
  - `ml_vs_dl_comparison.csv`
- `results/generalization/`
  - `clean_test_predictions.csv`
  - `generalization_predictions.csv`
- `results/adversarial/`
  - `stealth_predictions.csv`
  - `gpt41mini_zero_shot_stealth_predictions.csv`

## 主要脚本

### 数据生成

- `scripts/data_generation/generate_deepseek_answers.py`
- `scripts/data_generation/generate_kimi_answers.py`
- `scripts/data_generation/generate_glm_answers.py`
- `scripts/data_generation/generate_qwen_answers.py`
- `scripts/data_generation/generate_gpt41mini_answers.py`
- `scripts/data_generation/generate_gpt41mini_chinese_archive_answers.py`
- `scripts/data_generation/generate_full_dataset_pro.py`
- `scripts/data_generation/generate_test_100_new_questions.py`
- `scripts/data_generation/generate_chinese_archive_test.py`
- `scripts/data_generation/generate_stealth_answers.py`
- `scripts/data_generation/generate_gpt41mini_stealth_answers.py`

### 训练与评估

- `scripts/model_training/train_classifier.py`
- `scripts/model_training/train_e2e_transformer.py`
- `scripts/model_training/compare_classifiers.py`
- `scripts/model_training/run_ablation_experiment.py`
- `scripts/model_training/evaluate_cross_lingual.py`
- `scripts/model_training/evaluate_stealth.py`
- `scripts/model_training/plot_confusion_matrices.py`

### 可视化

- `scripts/visualization/visualize_features.py`
- `scripts/visualization/plot_stealth.py`

### 迭代对抗

- `iterative_adversarial_experiment/scripts/run_iterative_stealth.py`
- `iterative_adversarial_experiment/scripts/run_iterative_stealth_gpt41mini.py`

## 常用运行入口

```bash
python scripts/model_training/train_classifier.py
python scripts/model_training/run_ablation_experiment.py
python scripts/model_training/train_e2e_transformer.py
python scripts/model_training/plot_confusion_matrices.py
python scripts/model_training/evaluate_cross_lingual.py
python scripts/model_training/evaluate_stealth.py
```

GPT-4.1-mini 数据生成：

```bash
python scripts/data_generation/generate_gpt41mini_answers.py
python scripts/data_generation/generate_gpt41mini_chinese_archive_answers.py
python scripts/data_generation/generate_gpt41mini_stealth_answers.py
```

迭代对抗实验：

```bash
python iterative_adversarial_experiment/scripts/run_iterative_stealth.py
python iterative_adversarial_experiment/scripts/run_iterative_stealth_gpt41mini.py
```

## 归档说明

以下内容已主动移出主线目录，避免干扰复现实验：

- 未采用的中文原始题库与 LaTeX 素材
- 旧版中间 JSON
- `__pycache__`、`.pyc`、`.DS_Store`
- 历史探索脚本

如果你只关心当前可复现主线，请优先查看：

1. `dataset/`
2. `scripts/data_generation/`
3. `scripts/model_training/`
4. `results/`
5. `iterative_adversarial_experiment/`
6. `docs/reports/`

## 接下来待补实验计划

1. 重新完成 DL 六分类训练。
   - 在补齐 `transformers` 与可下载基础模型的环境后，基于加入 GPT 的训练集重训 `DistilBERT`。
   - 产出新的 `e2e_transformer_best.pt`、`e2e_dl_results.csv` 和 `confusion_matrix_dl.png`。

2. 刷新 ML vs DL 对比实验。
   - 以六分类版本重新生成 `ml_vs_dl_comparison.csv`。
   - 明确比较 GPT 加入前后，传统特征模型与端到端模型的精度、耗时和收敛情况。

3. 重新跑六分类消融实验。
   - 更新 `ablation_results.csv`。
   - 检查加入 GPT 后，段落结构、公式密度、逻辑词等特征的重要性是否发生变化。

4. 补齐新版泛化实验结果。
   - 使用加入 GPT 的两个 `generalization` 数据集重跑主评估。
   - 刷新 `generalization_predictions.csv`，并视 DL 是否可用决定是否同时补全 `DL_Prediction`。

5. 补齐 GPT 加入后的对抗检测评估汇总。
   - 在六分类分类器设定下，重新评估 `stealth_dataset.json` 中的 GPT 与原模型 stealth 样本。
   - 更新 `stealth_predictions.csv` 或新增更清晰的六分类对抗结果文件。

6. 统一刷新综合报告。
   - 将 `docs/reports/comprehensive_evaluation_report.md` 中仍保留旧五分类口径的段落逐步改写为与新版图表一致的表述。
   - 在最终版 README 中补一张“实验完成状态表”，区分已完成、部分完成和待完成项。
