<div align="center">
  <h1>🧠 数学推导文本溯源分类器 (Mathematical Proof Origin Classifier)</h1>
  <p><b>Human vs. Deepseek vs. Kimi vs. GLM vs. Qwen</b></p>
  <p><i>基于“宏观排版与微观语义特征工程”的大语言模型文本检测与反制研究</i></p>
</div>

---

## 📖 项目简介 (Introduction)

随着大语言模型（LLM）的普及，如何准确判断一段严谨的数学推导或证明是由**人类手写**还是由**某个特定 AI 模型**生成，已成为维护学术诚信的关键难题。

本项目摒弃了传统的“黑盒”深度学习文本分类模型，采用 **专家经验驱动的特征工程 + 传统树集成机器学习 (Hist Gradient Boosting)** 架构，实现了极高的可解释性与鲁棒性。我们的研究表明，顶级大模型在生成数学解答时，会留下深刻的“结构与物理指纹”（如不可控的换段频率、极端的公式包裹欲望、以及模板化的起手式）。基于此，我们的分类器在 **5 分类任务**中取得了高达 **95.5%** 的准确率，并在全新的数学交叉学科泛化实验中保持了极佳的稳定性。

进一步地，本项目还探索了**“防检测与反制干预”**（Counter-Intervention）：通过提炼出的可解释特征，我们编写了零样本（Zero-shot）系统提示词指导大模型进行特征伪装，成功使 Qwen 以 98% 的成功率骗过了高精度分类器，为未来的 AI 文本检测系统提出了深远挑战。

## 📁 目录结构 (Repository Structure)
- 采集同一批数学题的人类解答与多个大语言模型解答，构建严格配对的数据集。
- 提取能反映“证明写作风格”的结构、公式和语义特征，而不是只依赖词汇主题。
- 训练并比较多种机器学习分类器，识别解答来源。
- 通过 PCA、小提琴图、特征重要性等方式解释模型到底在利用什么特征。
- 进一步设计“防检测与反制干预实验”，检验大语言模型能否通过提示词规避现有检测器。

## 2. 项目思路

本项目的核心思想可以概括为三句话：

- **同题不同来源**：让同一道数学题分别对应人类与多个模型的解答，尽量压缩“题目内容差异”，放大“表达风格差异”。
- **双轨特征建模**：同时使用 `TF-IDF` 词汇特征和手工设计的结构化特征，既保留文本局部模式，又显式建模证明排版、LaTeX 使用和推理习惯。
- **分类 + 解释 + 对抗**：不仅训练分类器，还要解释为什么它有效，并进一步测试它在对抗提示词下是否会失效。

从研究路线看，这个项目经历了以下几步：

1. 从早期中文材料转向更稳定的纯英文数学数据。
2. 统一构建 `Human + 4 个 LLM` 的五分类数据集。
3. 从简单词袋分类，逐步发展到“结构特征 + 词汇特征”的混合建模。
4. 从标准分类实验，扩展到数据规模实验和防检测反制实验。

## 3. 当前目录结构

```text
Modelmid/
├── archive/                # 归档的旧版中文数据与废弃脚本
├── dataset/                # 数据集
│   ├── full_dataset.json            # 主训练集 (1000 题 × 5 来源 = 5000 条记录)
│   ├── full_dataset_pro.json        # 扩容泛化数据集 (2000 题 × 5 来源 = 10000 条记录)
│   ├── test_100_new_questions.json  # 独立英文测试集 (100 题全新数据)
│   └── test_100_chinese_archive_questions.json # 独立中文归档测试集 (100 题)
├── docs/                   # 项目综合报告与分析文档
│   ├── comprehensive_evaluation_report.md # 综合评估与实验报告 (合并了消融、泛化、深度学习对比等)
│   └── figures/                     # 实验数据统计图表 (PCA, Violin, 混淆矩阵等)
├── iterative_adversarial_experiment/ # 动态迭代对抗实验专区 (LLM-as-an-Optimizer)
│   ├── data/                        # 实验迭代历史与反馈日志
│   ├── scripts/                     # 自动化迭代生成与特征抽取脚本
│   └── adversarial_experiment_report.md # 实验详细报告与核心发现
├── latex_report/           # 最终排版的学术论文级 LaTeX 报告
│   ├── main.tex                     # 论文源文件
│   └── figures/                     # 论文插图 (PCA, Violin Plot, 学习曲线)
├── models/                 # 已训练好的模型文件
│   ├── best_classifier_model.pkl    # Hist Gradient Boosting 最佳 ML 分类器
│   ├── e2e_transformer_best.pt      # DistilBERT 端到端最佳 DL 分类器
│   ├── pro_feature_extractor.pkl    # 大规模数据集下的特征提取器
│   └── pro_ml_model.pkl             # 大规模数据集下的 ML 分类器
├── results/                # 预测结果与实验打分 (CSV)
│   ├── clean_test_predictions.csv   # 全新独立测试集的 ML vs DL 预测结果
│   ├── generalization_predictions.csv # 泛化实验预测结果
│   └── stealth_predictions.csv      # 防检测对抗实验预测结果
├── scripts/                # Python 源码目录
│   ├── archive/                     # 废弃或一次性探索脚本归档
│   ├── data_generation/             # 并发调用 LLM API 获取解答
│   ├── model_training/              # 特征工程、流水线搭建与模型训练评估
│   └── visualization/               # 数据统计、图表绘制与案例分析
├── .env                    # 环境变量 (存储 API Key)
└── README.md               # 项目主说明文档 (本文档)
```

## 4. 数据设计

### 4.1 主数据集

核心数据文件是 [full_dataset.json](file:///Users/jiaju/Documents/github/Modelmid/dataset/full_dataset.json)。

每道题包含以下信息：

- `id`：题目编号
- `problem`：题干
- `human`：人类答案
- `deepseek`：Deepseek 生成答案
- `kimi`：Kimi 生成答案
- `glm`：GLM 生成答案
- `qwen`：Qwen 生成答案

数据组织方式是：

```text
1 道题  ->  5 份答案  ->  5 个来源标签
1000 题 -> 5000 条记录
```

这种设计的好处是：

- 同一道题在五个来源之间天然配对，便于研究“同题不同风格”。
- 可以按题号切分训练集和测试集，避免同一道题的不同来源同时落入训练和测试造成数据泄漏。

### 4.2 防检测数据集

防检测实验使用 [stealth_dataset.json](file:///Users/jiaju/Documents/github/Modelmid/dataset/stealth_dataset.json)。

其中包含：

- 原始题目与人类答案
- 使用“反制提示词”重新生成的：
  - `deepseek_stealth`
  - `kimi_stealth`
  - `glm_stealth`
  - `qwen_stealth`

其对应的分类结果保存在 [stealth_predictions.csv](file:///Users/jiaju/Documents/github/Modelmid/dataset/stealth_predictions.csv)。

## 5. 代码结构与功能设计

### 5.1 数据生成层

位于 [data_generation](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation)。

- [migrate_to_json.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/migrate_to_json.py)
  - 从 HuggingFace 的 `StackMathQA` 迁移纯英文数学题和人类答案。
  - 生成统一 JSON 数据格式。

- [generate_deepseek_answers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/generate_deepseek_answers.py)
- [generate_kimi_answers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/generate_kimi_answers.py)
- [generate_glm_answers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/generate_glm_answers.py)
- [generate_qwen_answers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/generate_qwen_answers.py)
  - 负责调用对应 API，为题目生成标准英文数学解答。
  - 支持并发生成与实时保存。

- [generate_stealth_answers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/data_generation/generate_stealth_answers.py)
  - 使用防检测提示词，要求模型“写得更像人类”。
  - 是整个对抗实验的关键入口。

### 5.2 模型训练层

位于 [model_training](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training)。

- [train_classifier.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/train_classifier.py)
  - 项目核心训练脚本。
  - 加载五类文本。
  - 提取自定义结构特征。
  - 拼接 TF-IDF 特征。
  - 训练最终模型并保存到 [best_classifier_model.pkl](file:///Users/jiaju/Documents/github/Modelmid/models/best_classifier_model.pkl)。
  - 同时输出特征重要性分析图。

- [compare_classifiers.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/compare_classifiers.py)
  - 对比不同机器学习算法。
  - 重点用于回答“为什么最终模型不是线性模型而是树集成模型”。

- [experiment_new_features.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/experiment_new_features.py)
  - 对比旧特征集合和增强特征集合。
  - 验证新增结构特征是否真的提升效果。

- [run_data_size_experiment.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/run_data_size_experiment.py)
  - 比较不同训练题量下的模型表现。
  - 产出 `5 × 5` 的准确率主表。

- [evaluate_stealth.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/evaluate_stealth.py)
  - 加载已训练好的最佳分类器。
  - 评估防检测样本是否能骗过分类器。
  - 统计各模型的“伪装成功率”以及误判原因。

### 5.3 可视化层

位于 [visualization](file:///Users/jiaju/Documents/github/Modelmid/scripts/visualization)。

- [visualize_features.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/visualization/visualize_features.py)
  - 生成实验图表与 Markdown 版本的实验报告。
  - 核心图包括：
    - PCA 聚类图
    - 小提琴图
    - 统计表

- [plot_stealth.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/visualization/plot_stealth.py)
  - 专门绘制防检测实验中各模型的伪装成功率柱状图。

- [check_dist.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/visualization/check_dist.py)
  - 主要用于数据分布检查。

## 6. 算法设计

### 6.1 为什么不是纯深度学习

本项目刻意没有直接使用 BERT、RoBERTa 一类黑盒编码器作为最终方案，原因有三点：

- 需要强可解释性：希望知道“模型为什么说这篇像 Qwen，而不像 Human”。
- 数据规模相对有限：5000 条记录对于传统机器学习已经足够，但不一定值得引入更重的深度模型。
- 任务本质偏风格识别：很多关键线索不是语义内容，而是段落、公式、连接词和 LaTeX 习惯。

### 6.2 双轨特征流水线

最终分类器采用双轨输入：

1. `TF-IDF` 词袋特征
2. 手工结构特征

其核心思想是：

- `TF-IDF` 负责捕捉局部词汇和短语模式；
- 结构特征负责捕捉证明写作中的“物理风格”。

### 6.3 自定义结构特征

`TextFeatureExtractor` 当前提取约 28 个特征，主要分为三大类。

#### 宏观结构特征
- `length`
- `num_lines`
- `avg_line_length`
- `num_paragraphs`
- `avg_paragraph_length`

这些特征反映：

- 是否频繁换行
- 是否每推进一步就强行分段
- 人类写作是否更倾向于长段连续推导

#### 数学公式与 LaTeX 特征
- `inline_math_count`
- `display_math_count`
- `math_density`
- `latex_env_count`
- `left_right_brackets`
- `num_frac`
- `math_cal_bb`
- `implication_arrows`
- `qed_symbols`

这些特征反映：

- 模型是否过度使用 `$...$`
- 是否频繁使用复杂 LaTeX 环境
- 是否有过度“正规化”的公式包装习惯

#### 语义与论证风格特征
- `logical_words_density`
- `declarative_density`
- `transition_words_density`
- `conclusion_words_density`
- `num_list_items`

这些特征反映：

- 是否频繁使用 `we, let, suppose, consider`
- 是否使用 `firstly, moreover, finally`
- 是否存在机械的列表式推导习惯

### 6.4 分类器设计

本项目测试了五种模型：

- `Linear SVM`
- `Logistic Regression`
- `Random Forest`
- `Gradient Boosting`
- `Hist Gradient Boosting`

当前代码中保存为最终模型的是：

- [train_classifier.py](file:///Users/jiaju/Documents/github/Modelmid/scripts/model_training/train_classifier.py) 中训练并保存的 `HistGradientBoostingClassifier`

之所以最终转向树集成模型，是因为：

- 线性模型擅长高维稀疏词袋，但不擅长复杂非线性交互；
- 本项目中很多关键判别边界本质上是非线性的；
- 树模型对“段落数 + 行内公式 + 祈使句密度”这类交互更敏感。

## 7. 实验流程

整个项目的实验流程可以概括为：

1. 迁移英文数学数据集。
2. 生成四种大模型答案。
3. 统一为五分类 JSON 数据集。
4. 提取词汇特征与结构特征。
5. 比较多种分类器。
6. 做不同训练题量的规模实验。
7. 可视化特征空间。
8. 设计防检测提示词并做对抗评估。
9. 汇总为 Markdown 与 LaTeX 报告。

对应脚本入口如下：

```bash
# 1. 构建主数据集
python3 scripts/data_generation/migrate_to_json.py

# 2. 生成四类模型答案
python3 scripts/data_generation/generate_deepseek_answers.py
python3 scripts/data_generation/generate_kimi_answers.py
python3 scripts/data_generation/generate_glm_answers.py
python3 scripts/data_generation/generate_qwen_answers.py

# 3. 训练主分类器
python3 scripts/model_training/train_classifier.py

# 4. 比较分类器
python3 scripts/model_training/compare_classifiers.py

# 5. 数据规模实验
python3 scripts/model_training/run_data_size_experiment.py

# 6. 可视化
python3 scripts/visualization/visualize_features.py

# 7. 防检测样本生成与评估
python3 scripts/data_generation/generate_stealth_answers.py
python3 scripts/model_training/evaluate_stealth.py
python3 scripts/visualization/plot_stealth.py
```

## 8. 核心实验结果

### 8.1 五分类准确率

项目当前存在两套实验口径，需要区分说明：

- **口径 A：5-fold 交叉验证**
  - 早期组合特征 + SVM 的交叉验证结果约为 `89.88%`
  - 主要用于说明双轨特征工程是有效的

- **口径 B：固定 200 题测试集的数据规模实验**
  - 当前最佳模型为 `Hist Gradient Boosting`
  - 在 `800` 题训练集规模下达到 `95.5%`
  - 这是当前代码体系下最重要的最终结果

### 8.2 数据规模实验主表

固定测试集为 200 题时，五种模型在不同训练题量下的准确率如下：

| 模型 | 20题 | 50题 | 100题 | 400题 | 800题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hist Gradient Boosting | **0.868** | **0.895** | **0.916** | **0.947** | **0.955** |
| XGBoost (sklearn GB) | 0.824 | 0.870 | 0.909 | 0.939 | 0.944 |
| Random Forest | 0.850 | 0.895 | 0.909 | 0.928 | 0.934 |
| Linear SVM | 0.782 | 0.833 | 0.854 | 0.912 | 0.917 |
| Logistic Regression | 0.787 | 0.841 | 0.853 | 0.895 | 0.905 |

这说明：

- 树集成模型在本项目中持续优于线性模型；
- 即使训练集只有 20 题，分类器也已经能学到明显来源差异；
- 随着训练数据增加，Hist Gradient Boosting 的优势进一步扩大。

### 8.4 特征消融实验 (Ablation Study)

为了进一步证明我们的双轨特征工程的有效性，我们对 5000 条主数据集进行了 5-Fold 特征消融验证：

| Configuration | 准确率 (Accuracy) | 相比 Full Model 性能下降 (Drop) |
| :--- | :---: | :---: |
| **Full Model (TF-IDF + Custom)** | **95.36%** | - |
| **Full w/o Macro Structure** | 94.80% | ↓ 0.56% |
| **Full w/o Math/LaTeX** | 94.58% | ↓ 0.78% |
| **TF-IDF Only** | 93.64% | ↓ 1.72% |
| **Custom Features Only** | 90.32% | ↓ 5.04% |

**结论**：
- 仅依靠 `TF-IDF`，准确率停留在 93.64%。加入专家经验构建的自定义特征后，模型性能显著拉高至 **95.36%**。
- 仅使用 `Custom Features` 也能达到 **90.32%**，这证明即使完全不看具体词汇内容，仅靠计算文本的“物理属性”（如段落长短、逻辑词密度、数学公式特征），也足以在五分类中取得超过 90% 的表现。
- 在子模块中，消融掉**数学/LaTeX 特征**带来的性能下降最大 (↓ 0.78%)，说明不同模型在处理数学符号的排版习惯上具有极强的指纹特异性。
- 完整报告请见：[comprehensive_evaluation_report.md](file:///Users/jiaju/Documents/github/Modelmid/docs/comprehensive_evaluation_report.md) 中的“特征消融实验”章节。

## 9. 可视化设计

### 9.1 PCA 聚类图

图像位置：

- [pca_clusters_2d.png](file:///Users/jiaju/Documents/github/Modelmid/docs/figures/pca_clusters_2d.png)
- [latex_report/figures/pca_clusters_2d.png](file:///Users/jiaju/Documents/github/Modelmid/latex_report/figures/pca_clusters_2d.png)

作用：

- 将高维结构特征降到二维
- 观察 Human 与各类模型在特征空间中的分离情况

结论：

- Human 形成更独立、更紧密的簇
- Kimi 和 GLM 更容易重叠
- Qwen 有非常鲜明的风格区分度

### 9.2 小提琴图

代表性图像：

- [violin_num_paragraphs.png](file:///Users/jiaju/Documents/github/Modelmid/docs/figures/violin_num_paragraphs.png)
- [violin_declarative_density.png](file:///Users/jiaju/Documents/github/Modelmid/docs/figures/violin_declarative_density.png)
- [violin_inline_math_count.png](file:///Users/jiaju/Documents/github/Modelmid/docs/figures/violin_inline_math_count.png)

作用：

- 展示关键特征在不同来源上的分布，而不是只展示平均值
- 让“人类与模型风格差异”更直观

### 9.3 防检测实验图

图像位置：

- [stealth_success_rate.png](file:///Users/jiaju/Documents/github/Modelmid/latex_report/figures/stealth_success_rate.png)

作用：

- 展示不同模型在“写得更像人类”之后，骗过分类器的成功率

## 10. 防检测与反制干预实验

这是本项目后期最重要的扩展实验之一，包括静态零样本实验与动态迭代对抗实验。

### 10.1 静态零样本干预

既然我们已经知道分类器最依赖哪些特征，那么就可以反过来把这些结论写进提示词中，要求模型：

- 少分段
- 少用行内公式
- 不用 `We have / Let / Suppose / Consider`
- 不用 `Firstly / Moreover / Finally`
- 写成长段连续推导

也就是说，我们不是给模型“人类答案样例”，而是直接告诉它“如何避开会暴露 AI 身份的结构特征”。在 194 份有效防检测样本中，被误判为 Human 的比例高达 **63.40%**。

### 10.2 动态迭代对抗实验 (Iterative Adversarial Experiment)

为了进一步探索 AI 自我进化的反侦察能力，我们设计了基于 LLM 反馈的迭代对抗实验（LLM-as-an-Optimizer）。
- **流程**：将分类器（判别器）识别失败样本的详细特征（如平均段落数、行内公式数量、祈使句密度等）作为反馈，输入给另一个大模型（优化器），要求其自主改写生成 Prompt。
- **专项测试 (针对 Kimi)**：我们挑选了在静态实验中最难被攻陷的 **Kimi** 作为生成器，并由 **DeepSeek** 作为优化器。我们进行了两阶段的对比实验：
  - **阶段一 (先验主导型 Optimizer)**：当优化器没有被严格限制时，会产生“幻觉叠加”，即盲目引入“口水话/刻意不换行”等非自然要求，导致 Kimi 绕过率止步于 33.33% 并在后期崩溃。
  - **阶段二 (数据驱动型 Optimizer)**：当我们在提示词中彻底封杀了优化器的先验发散，强制其**仅将特征数值的偏差映射为严格的结构化约束**（如明确禁止特定词汇、精确限定公式数量）后，发生了质变。
- **最终结果**：在数据驱动型反馈下，Kimi 仅用 2 轮迭代就**瞬间实现了 100% 的检测绕过率**。这一震撼结果说明，即便是最难伪装的模型，只要提示词能够精准且严格地封锁高精度分类器的“特征锚点”，AI 的结构指纹依然可以被完全抹除。
- **实验数据与报告**：本次动态对抗实验的代码、完整的各阶段 Prompt 进化历史（JSON）以及详细发现，已独立打包至 [iterative_adversarial_experiment](file:///Users/jiaju/Documents/github/Modelmid/iterative_adversarial_experiment) 目录中，以便后续论文写作与引用。详细报告请见 [adversarial_experiment_report.md](file:///Users/jiaju/Documents/github/Modelmid/iterative_adversarial_experiment/adversarial_experiment_report.md)。

### 8.5 大规模跨学科泛化实验 (ML vs DL)

为了测试模型在不同数学子学科上的鲁棒性，我们将题库扩大至 2000 题（共计 10,000 条记录），涵盖了代数 (`algebra`)、概率 (`counting_and_probability`) 等子领域，并构建了深度学习 (Deep Learning) 与传统机器学习 (Machine Learning) 的全面基线对比：

1. **分层抽样**：按题目 ID 分组，1800 题用于训练，200 题用于测试。
2. **算法对比结果 (带有 Early Stopping)**：
   - **RandomForest (ML)**：测试集 **97.70%** (耗时 ~0.96秒)
   - **HistGradientBoosting (ML)**：测试集 **97.40%** (耗时 ~30秒)
   - **ResNet_DNN (DL)**：测试集 **97.40%** (收敛于 54 轮)
   - **Simple_MLP (DL)**：测试集 **97.00%** (收敛于 29 轮)
   - **Conv1D_Net (DL)**：测试集 **96.50%** (收敛于 62 轮)
3. **结论**：
   - **风格指纹的跨学科一致性**：不论是在通用数学还是具体的代数/概率题中，大模型的排版和用词习惯依然极其固定。
   - **树集成依然是结构化特征之王**：在基于我们精心设计的“统计特征+TF-IDF”这种异构拼接数据表上，传统的随机森林（RandomForest）在不到 1 秒内就跑出了最高准确率。深度学习即使上了带 Skip Connection 的 ResNet-like 架构，也仅仅是追平了梯度提升树，且付出了更高的调参和训练成本。这充分证明：**脱离了词向量直接编码，纯靠专家经验特征工程时，树模型是最高效且鲁棒的。**
   - 详细实验报告请见：[comprehensive_evaluation_report.md](file:///Users/jiaju/Documents/github/Modelmid/docs/comprehensive_evaluation_report.md) 中的“扩容数据集与基线对比”章节。

对比成功骗过分类器的样本和依然被识别的样本后可以看到：

- 成功骗过分类器的文本：
  - 平均段落数只有 `1.36`
  - 祈使代词密度只有 `1.29`

- 仍被识别的文本：
  - 平均段落数高达 `8.68`
  - 祈使代词密度高达 `4.08`

这说明误判不是随机的，而是因为这些样本确实被提示词推到了“更像人类”的结构分布区域。

### 8.6 端到端预训练语言模型分类实验 (E2E Transformer)

为了彻底颠覆传统的手工特征提取工程，我们进一步构建了基于预训练语言模型（Pre-trained Language Models）的端到端（End-to-End）深度学习分类器。
在该实验中，我们抛弃了所有手动设计的结构化特征与 TF-IDF 词袋模型，直接将原始数学解答文本输入到 `distilbert-base-uncased` 中进行微调（Fine-tuning）。

1. **实验设置**：
   - 使用包含 10,000 条记录的泛化数据集，按照 1800题/200题（训练/验证）进行严格的题号隔离分层抽样。
   - 限制 `Max Sequence Length = 512`，使用 `Batch Size = 16` 以防止 Mac MPS 显存溢出 (OOM)。
   - 引入了基于 Validation Accuracy 的 Early Stopping (patience=2) 与 Checkpointing 机制。
2. **训练结果与同源验证 (Validation)**：
   - 训练在 Epoch 7 触发了 Early Stopping，而**模型在 Epoch 5 达到了 98.10% 的验证集准确率巅峰**，超越了手工特征提取的机器学习基线 (97.70%)。
   - 在 1000 条测试记录上，Deepseek、Human、Qwen 的 F1-score 高达 0.99。这证明 Transformer 可以从原始序列中隐式学习到更深的特征。
3. **全新测试集独立评估 (Zero-Shot Generalization)**：
   - 为了严谨验证该模型是否存在数据泄露或过拟合，我们额外从 Hendrycks MATH 测试集中抽取了 **100 道全新且未见过的问题**。
   - 重新调用四大模型 API 生成解答，构建了包含 500 条记录的全新独立测试集。
   - E2E Transformer 模型在该全新独立测试集上依然达到了 **96.00%** 的泛化准确率。
4. **跨语言测试挑战 (Zero-Shot Cross-Lingual Evaluation)**：
   - 我们编写了 [generate_chinese_archive_test.py](scripts/data_generation/generate_chinese_archive_test.py) 从归档文件中抽取了 100 道原生的纯中文独立测试题，并调用 API 补齐了 500 份 AI 与 Human 的中文回答。
   - 在未经任何中文微调的情况下，**传统 ML 最佳基线 (HistGB) 准确率降至 54.00%**（TF-IDF 英文词汇全部失效，只能盲猜结构特征）；**E2E 深度学习模型 (DistilBERT) 准确率降至 57.00%**。
   - 这充分反映了模型对于跨语言、未见过词表的泛化极限，同时也反映出 AI 在不同语言中依然残留了极少量可以被深度注意力机制捕获的排版物理指纹（例如 Qwen 的中文识别 F1-score 依然高达 0.77）。
   - 详细评估代码见：[evaluate_cross_lingual.py](scripts/model_training/evaluate_cross_lingual.py)。

## 11. 报告与文档
所有的实验文档已汇总合并至 `docs/comprehensive_evaluation_report.md` 中，内含详细的图表和特征分析，包括：
- [综合评估与实验报告](file:///Users/jiaju/Documents/github/Modelmid/docs/comprehensive_evaluation_report.md)

### LaTeX 报告

- [main.tex](file:///Users/jiaju/Documents/github/Modelmid/latex_report/main.tex)
  - 当前主报告
  - 包含方法、特征工程、规模实验、可解释性分析和反制实验

- [results_table.csv](file:///Users/jiaju/Documents/github/Modelmid/latex_report/results_table.csv)
  - LaTeX 表格的原始数据来源

## 12. 环境与运行说明

### 12.1 API 密钥

根目录 `.env` 需要包含类似配置：

```env
DEEPSEEK_API_KEY="..."
MOONSHOT_API_KEY="..."
GLM_API_KEY="..."
QWEN_API_KEY="..."
```

### 12.2 注意事项

- 当前部分脚本使用了硬编码绝对路径，迁移到其他机器时需要调整。
- `README` 中如果引用实验结果，需要注意区分“5-fold 交叉验证”和“固定测试集实验”两种协议。
- 当前保存的最佳模型是 `HistGradientBoostingClassifier`，而不是旧版本文档里曾提到的 `SVM`。

## 13. 当前结论

基于当前项目代码、实验和报告，可以总结为：

- 数学证明来源识别是可行的，而且准确率很高。
- 真正关键的不是题目内容，而是写作结构、公式包装和推理习惯。
- 树集成模型比线性模型更适合这个任务。
- 当前最强模型在固定测试协议下可以达到 `95.5%`。
- 但高精度检测器并不意味着绝对安全：在提示词驱动下，顶级大模型已经具备明显的“伪装成人类”的能力。

## 14. 后续可以继续做什么

- 引入更多学科或题型，测试跨领域泛化能力。
- 研究更鲁棒的检测器，例如结合逻辑一致性、推理路径和交互验证。
- 设计更系统的对抗训练，让分类器在面对“防检测提示词”时更不容易失效。
