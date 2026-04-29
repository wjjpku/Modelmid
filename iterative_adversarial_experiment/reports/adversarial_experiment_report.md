# 迭代对抗实验研究报告 (Iterative Adversarial Experiment Report)

## 1. 实验背景与动机 (Background & Motivation)

在传统的 AI 文本检测研究中，研究者通常通过收集大量数据训练分类器。然而，随着大型语言模型 (LLM) 能力的提升，一个关键问题浮现：**LLM 是否能通过自我反馈和迭代，自主找到绕过高精度分类器的文本生成策略？**

本项目的初始阶段已经成功构建了一个基于宏观排版（段落数）与微观语义特征（祈使句密度、LaTeX 频率）的强可解释分类器，准确率高达 95.5%。为了测试该分类器的鲁棒性上限，我们设计了“基于 LLM 反馈的迭代对抗实验 (LLM-as-an-Optimizer)”。

在前期静态防检测测试中，我们发现 **Kimi (Moonshot)** 的内部对齐风格极其固化，是最难被调整和伪装的模型。因此，我们挑选 Kimi 作为对抗实验的“生成器 (Generator)”，并使用 **DeepSeek** 作为“优化器 (Optimizer)”。

## 2. 实验设计 (Experimental Design)

实验分为两个截然不同的阶段，旨在对比不同提示策略下优化器的表现。

### 2.1 整体架构
- **判别器 (Discriminator)**：已训练好的 `best_classifier_model.pkl` 及 `TextFeatureExtractor`。
- **生成器 (Generator)**：Kimi (moonshot-v1-8k)。
- **优化器 (Optimizer)**：DeepSeek (deepseek-chat)。
- **反馈循环**：生成器输出 -> 判别器提取失败特征并计算绕过率 -> 将特征偏差组装成反馈 -> 优化器重写生成器的 System Prompt。

### 2.2 阶段一：先验主导型优化器 (Prior-Driven Optimizer)
**设计思路**：
在这一阶段，我们对优化器的约束较少，仅提供平均特征的反馈，并要求其“分析失败原因并重写 Prompt，使其更像人类”。

**现象与问题**：
优化器 (DeepSeek) 出现了严重的 **“幻觉叠加” (Hallucination Stacking)**。
1. 优化器过度依赖自身对“人类行为”的先验刻板印象，认为人类必然会写口水话、不换行、甚至逻辑混乱。
2. 随着迭代深入，Prompt 中出现了诸如“加入口语化词汇 hmm/wait”、“完全不换行”、“随意切断句子”等极端角色扮演指令。
3. 面对这种无理指令，生成器 (Kimi) 出现了 **Style Collapse (风格崩塌)**，不仅难以维持数学推导的严谨性，还在后期由于指令冲突重新暴露了 AI 特征。

**结果**：
绕过率在第 3 轮艰难爬升至 33.33% 后，迅速回落至 0%，触发 Early Stopping 终止。

### 2.3 阶段二：数据驱动型优化器 (Data-Driven Optimizer)
**设计思路**：
基于阶段一的教训，我们意识到：**“大模型认为的人类” ≠ “真实分类器依赖的人类数据分布”**。因此，我们对优化器进行了“降维打击”。
1. **彻底封杀先验发散**：在 Optimizer Prompt 中加入 `CRITICAL RULE`，严禁使用“口语化、语气词、错别字”等假设。
2. **强制显式映射**：在 Feedback Prompt 中，直接将特征的数值偏差转化为硬性约束要求（例如，将“段落数过多”明确映射为“Action: Constrain paragraph count”）。

**现象与突破**：
优化器抛弃了角色扮演的废话，直接输出了高度结构化、数字化的约束条件（如：“Write exactly 1-2 paragraphs”, “Limit inline math to a maximum of 5”, “Banned Words: we, let, because”）。

**结果**：
仅在第 2 轮，生成器 (Kimi) 就在这种“紧箍咒”般的提示词下，**瞬间实现了 100% 的检测绕过率 (15/15 evaded detection)**！

## 3. 核心洞察与未来工作 (Insights & Future Work)

### 3.1 核心洞察
1. **可解释特征的双刃剑**：基于专家经验和统计特征构建的高精度模型，虽然解释性极强，但一旦其“特征锚点”暴露给一个能严格遵循指令的大模型，其防线就会被瞬间突破。纯文本特征检测可能无法作为长期的防御手段。
2. **LLM 优化器的局限性与用法**：在构建 LLM Agent 系统时，不能放任模型自由发散。必须使用强显式的数据和规则对其进行约束，迫使其进行“数值到规则的转化”，而非“基于常识的猜想”。

### 3.2 可用于 Paper 的实验数据支持
本文件夹 (`iterative_adversarial_experiment/data/`) 保留了完整的 JSON 实验记录，为撰写论文提供了丰富的数据支持：
- `stealth_iteration_history.json`: 早期针对 DeepSeek 的初始验证数据。
- `kimi_stealth_iteration_history.json`: 阶段一（先验主导型）的完整失败历史记录，可用于分析大模型刻板印象导致的 Style Collapse。
- `kimi_data_driven_stealth_history.json`: 阶段二（数据驱动型）的成功历史记录，记录了 Prompt 如何被优化为结构化约束以及绕过率飙升至 100% 的轨迹。

### 3.3 下一步计划
- 引入**准确性评估**：即虽然绕过了检测，但文本中的数学推导逻辑是否依然正确。
- 探索**黑盒强化学习 (Black-box RL)**：隐藏具体的特征指标名称，仅返回一个整体评分，观察大模型能否在没有明确指导下探索出逃逸路径。