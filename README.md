# 项目：基于文本与逻辑特征的纯英文数学作业来源判别 (Human vs Deepseek vs Kimi vs GLM vs Qwen)

## 1. 项目背景与目标
随着大语言模型（LLM）的普及，利用 ChatGPT、Kimi、Deepseek、GLM、Qwen 等工具辅助完成数学理论推导作业已成为常见现象。
本项目的核心目标是：**通过采集纯英文环境中人类真实的数学作业解答，并利用不同大语言模型生成相同题目的解答，提取其风格、排版与逻辑特征，构建一个高精度的机器学习分类器，实现对解答来源（人类还是某种特定 AI）的精准判别。**

---

## 2. 项目目录结构

```text
Modelmid/
├── archive/                # 归档的旧版中文数据与旧脚本
├── dataset/                # 处理后的结构化数据目录
│   └── full_dataset.json   # 核心数据集 (id, problem, human, deepseek, kimi, glm, qwen)
├── docs/                   # 项目文档与特征可视化图表
├── models/                 # 训练好的机器学习模型
│   └── best_classifier_model.pkl  # 表现最好的组合特征 SVM 模型
├── scripts/                # 自动化处理与模型训练脚本
│   ├── data_generation/    # 数据获取与 LLM 并发生成脚本
│   │   ├── migrate_to_json.py
│   │   ├── generate_deepseek_answers.py
│   │   ├── generate_kimi_answers.py
│   │   ├── generate_glm_answers.py
│   │   └── generate_qwen_answers.py
│   ├── model_training/     # 特征工程与模型训练脚本
│   │   ├── train_classifier.py
│   │   └── analyze_tfidf.py
│   ├── visualization/      # 数据与特征可视化脚本
│   │   ├── visualize_features.py
│   │   └── check_dist.py
│   └── legacy_utils/       # 历史遗留工具脚本
├── .env                    # 环境变量配置文件 (存放 API Keys)
├── experience.md           # 记录项目探索过程、踩坑经验与反思
└── README.md               # 项目主说明文档 (本文档)
```

---

## 3. 核心工作流与运行方式

整个项目分为三个主要阶段：纯英文数据拉取、LLM 数据生成、特征工程与模型训练。

### 阶段一：高质量纯英文数据集构建
我们废弃了之前的小规模中文数据，转而接入 HuggingFace 上的高质量数学推理数据集 `math-ai/StackMathQA`。
脚本会从中提取 1000 条纯英文的数学题目（`problem`）以及真实人类专家给出的标准解答（`human`），并构建为统一的 JSON 格式。
- **运行命令**：
  ```bash
  python3 scripts/data_generation/migrate_to_json.py
  ```

### 阶段二：大语言模型数据生成 (多线程并发 + 实时安全保存)
为了构建用于对比的负样本，脚本会读取 JSON 文件中空缺的 LLM 字段，将题干配合严格的纯英文提示词（禁止寒暄，纯粹 LaTeX 格式输出），**通过多线程并发方式** 发送给对应的 API。
*注：脚本内置了严格的线程锁与原子级文件替换机制，哪怕遇到断电或强制中断，已生成的数据也不会丢失且 JSON 文件不会损坏。*
- **环境配置**：在根目录创建或编辑 `.env` 文件，填入你的 API 密钥：
  ```env
  DEEPSEEK_API_KEY="your_deepseek_key_here"
  MOONSHOT_API_KEY="your_kimi_key_here"
  GLM_API_KEY="your_glm_key_here"
  QWEN_API_KEY="your_qwen_key_here"
  ```
- **运行命令**：
  ```bash
  python3 scripts/data_generation/generate_deepseek_answers.py
  python3 scripts/data_generation/generate_kimi_answers.py
  python3 scripts/data_generation/generate_glm_answers.py
  python3 scripts/data_generation/generate_qwen_answers.py
  ```

### 阶段三：模型训练与特征工程
在凑齐 Human, Deepseek, Kimi, GLM, Qwen 五个维度的均衡纯英文数据集（共 5000 条）后，我们执行了多层次的特征提取和模型训练。
- **运行命令**：
  ```bash
  python3 scripts/model_training/train_classifier.py
  ```
- **输出结果**：在终端打印 5-fold 交叉验证的准确率、详细的分类报告，并将最优模型固化保存至 `models/best_classifier_model.pkl`。

---

## 4. 模型设计与实验分析
在凑齐 Human (219条)、Deepseek (219条)、Kimi (219条) 的均衡三分类数据集后，我们执行了多层次的特征提取和模型训练。

您可以运行以下命令生成特征的可视化图表，结果将保存在 `docs/figures/` 中：
```bash
python3 scripts/visualization/visualize_features.py
```

### 4.1 领域特异性与防数据泄露特征工程
除了传统的 **TF-IDF** (词袋特征)，我们发现基于数学文本排版的**自定义结构特征**具有极其强大的区分度：
1. **基础排版特征**：回答总长度 (`length`)、行数 (`num_lines`)、平均每行长度 (`avg_line_length`)。
2. **公式特征**：行内和块级公式的总数 (`math_blocks`)、公式密度 (`math_density`)。
3. **特定 LaTeX 宏**：`\textbf`、`\frac`、`\sum` 等的使用频率。
4. **逻辑词频**：推理连词（如：因为、所以、显然、同理、从而、故等）的出现次数 (`logical_words_count`) 和密度 (`logical_words_density`)。

**TF-IDF 深度清洗与防作弊（Data Leakage Prevention）**：
我们在对 TF-IDF 的特征重要性进行深入排查时，发现了一些能让模型“走捷径”的作弊词（Tricky Features）。例如：
- LLMs 极度喜欢在最后加上 `\boxed{}` 来框住答案，或在推导中使用 `\quad` 空格。
- LLMs 喜欢用“综上所述”、“接下来”、“我们需要证明”等固定的机器套话。
为了逼迫模型去学习**真正底层的排版逻辑与数学思维差异**，而不是依赖这些表面的格式套话，我们在最终的模型流水线中（`train_classifier.py`）将这些特征作为 `stop_words` 进行了强行屏蔽。

### 4.2 实验结果 (五分类：Human vs Deepseek vs Kimi vs GLM vs Qwen)
在引入了全新的 **词汇丰富度**、**公式细分（行内vs块级）**、**标点符号密度**和**大模型祈使句特征**后，我们在全英文数据集（5000 条记录）下进行了 5-fold 交叉验证：
- **最佳模型**：组合特征 (TF-IDF + 深度自定义特征) + SVM 分类器。
- **交叉验证准确率**：稳定在 **88%**。
- **训练集 F1-score 详情**：
  - Human：0.97 (人类和 AI 依然极其容易区分)
  - Qwen：0.98 (极其突出的个体风格)
  - Deepseek：0.95
  - Kimi：0.89
  - GLM：0.88

*(注：Kimi 和 GLM 之间的准确率有所下降，是因为这两种模型在输出全英文数学推导时的特征空间高度重合。而 Qwen 则表现出了极强的行文辨识度！)*

### 4.3 核心特征洞察与可解释性
为了增强模型的可解释性，我们提取了随机森林对新特征的基尼重要性（Gini Importance）。区分这五个来源最核心的前五大特征如下：
1. **换行习惯 (`num_lines`, 16.5%)**：依然是最强大的分类器，不同模型和人类分段的底层逻辑差异巨大。
2. **行内公式数量 (`inline_math_count`, 10.5%)**：Qwen 的加入使得模型开始极度关注公式包裹频率。
3. **大模型祈使句密度 (`declarative_density`, 8.6%)**：我们新引入的特征（如 *we, let, suppose, consider, now*）。大模型在推导时比人类更喜欢频繁使用这些起手式。
4. **数学公式总密度 (`math_density`, 8.4%)**：衡量整篇文章中数学字符的比例。
5. **平均行长 (`avg_line_length`, 8.1%)**：大段不换行的特征。

*(更多关于特征重要性排序，请查看 `docs/figures/feature_importances.png`)*

### 4.4 全新可视化展示 (可解释性视角)
为了进一步探索模型是如何分类的，我们生成了全新的图表视角（位于 `docs/figures/`）：

**1. PCA 2D 聚类降维 (特征空间可解释性)**  
我们对所有自定义特征进行了主成分分析（PCA）。从 `pca_clusters_2d.png` 可以看出，**Human（绿色）** 形成了极其紧密且独立的聚类簇，与机器截然不同；**Deepseek（蓝色）** 也有自己独立的一片区域；而 **Kimi（橙色）** 和 **GLM（红色）** 的特征点相互交织，解释了为什么它们之间的分类存在一定混淆。

**2. 特征相关性热力图**  
从 `feature_correlation_heatmap.png` 中，我们可以观察到 `length`、`word_count` 和 `display_math_count` 等特征之间的高度共线性，这为后续的特征降维提供了依据。

**3. 核心特征小提琴图 (Violin Plots)**  
相比于箱线图，小提琴图更好地展示了数据分布的密度。例如在 `violin_declarative_density.png` 中，你可以清晰看到人类的“祈使句”密度呈现长尾低频分布，而几个大模型则呈现出高频的纺锤形分布。

*(更多关于模型迭代、踩坑经历与数据清洗的细节，请参阅 [experience.md](experience.md))*

---

## 5. 后续规划 (TODO)
- **泛化性验证**：引入其他理科领域（如：常微分方程、概率论等）的数据，检验这些提取出的排版与逻辑特征是否依然稳健。
- **对抗生成网络 (GAN) 测试**：研究对抗判别器的潜在方法，即尝试通过高级的 Prompt 工程，强制要求大模型去刻意模仿人类的排版习惯和逻辑词频，观察当前分类器是否会被欺骗。