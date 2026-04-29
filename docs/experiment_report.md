# 实验与特征数据报告 (Experiment Data Report)

**生成时间**: 2026-04-30 05:58:33
**数据规模**: 6000 条
**来源类别**: Deepseek, GLM, GPT-4.1-mini, Human, Kimi, Qwen
**特征维度**: 28 个深度排版、结构与逻辑特征

---

## 1. 宏观排版与结构特征对比 (Macroscopic Structure)

大模型受限于自回归生成机制，在处理长文本证明时，极度依赖频繁的段落切换和序列标记。相反，人类更倾向于连贯的单段长推导。

### 1.1 关键数据均值对比
| 来源 (Source) | 平均段落数 (`num_paragraphs`) | 平均每段字符数 (`avg_paragraph_length`) | 换行数 (`num_lines`) |
| --- | --- | --- | --- |
| **Deepseek** | 36.45 | 117.07 | 126.91 |
| **GLM** | 12.6 | 248.25 | 74.33 |
| **GPT-4.1-mini** | 50.65 | 71.96 | 164.45 |
| **Human** | 1.3 | 699.45 | 12.44 |
| **Kimi** | 16.31 | 175.66 | 58.66 |
| **Qwen** | 50.08 | 88.16 | 147.56 |

*解读：人类的平均段落数极少（仅为大模型的1/2到1/3），但每个段落的信息密度（字符数）是所有 AI 的数倍。大模型极度依赖双换行来组织思维。*

### 1.2 核心特征分布图
![段落数量分布 (Violin Plot)](./figures/gpt_augmented/violin_num_paragraphs.png)

---

## 2. 数学公式特异性与严谨度 (Mathematical Formats)

不同的大脑对于“何时使用 LaTeX 渲染”有着完全不同的理解，Qwen 展现出了远超其他模型的行内公式包裹欲望。

### 2.1 关键数据均值对比
| 来源 (Source) | 行内公式频次 (`inline_math_count`) | 块级公式频次 (`display_math_count`) | 复杂环境频率 (`latex_env_count`) |
| --- | --- | --- | --- |
| **Deepseek** | 1.12 | 15.54 | 0.9 |
| **GLM** | 11.33 | 13.42 | 0.87 |
| **GPT-4.1-mini** | 0.0 | 20.13 | 0.87 |
| **Human** | 7.43 | 2.83 | 0.84 |
| **Kimi** | 5.96 | 11.95 | 0.84 |
| **Qwen** | 49.32 | 14.74 | 1.06 |

*解读：不同模型在“何时使用行内公式”和“是否偏好复杂 LaTeX 环境”上差异明显。新增的 GPT-4.1-mini 也可在这组特征中与原五类来源一起比较。*

### 2.2 核心特征分布图
![行内公式包裹频率 (Violin Plot)](./figures/gpt_augmented/violin_inline_math_count.png)

---

## 3. 词汇风格与大模型套话 (Vocabulary & Semantic Fingerprints)

大模型在数学推导时，有着极其统一的“机器感起手式”。

### 3.1 关键数据均值对比 (每千字密度)
| 来源 (Source) | 祈使句/代词密度 (`we, let, suppose`) | 大写字母密度 (`uppercase_density`) | 序列衔接词密度 (`firstly, secondly`) |
| --- | --- | --- | --- |
| **Deepseek** | 2.6 | 21.13 | 0.01 |
| **GLM** | 2.18 | 19.23 | 0.04 |
| **GPT-4.1-mini** | 1.44 | 19.66 | 0.02 |
| **Human** | 2.47 | 14.87 | 0.06 |
| **Kimi** | 4.48 | 15.07 | 0.13 |
| **Qwen** | 2.48 | 19.8 | 0.02 |

*解读：几乎所有的 LLM 都极其喜欢使用 "We have", "Let x be", "Now consider" 这样的祈使代词句式作为推导开头，人类的使用密度要低得多。同时，大模型严格的语法训练导致其大写字母的分布（首字母大写规范）远高于随意的人类手写。*

### 3.2 核心特征分布图
![大模型祈使句密度 (Violin Plot)](./figures/gpt_augmented/violin_declarative_density.png)

---

## 4. PCA 主成分分析 (Principal Component Analysis)

为了验证上述 28 个深度特征的组合能否在数学空间中有效区分这些文本，我们进行了 PCA 降维。

![PCA 降维特征空间聚类](./figures/gpt_augmented/pca_clusters_2d.png)

*解读：在二维 PCA 空间中，不同来源的排版与措辞风格会形成可观察的聚类结构。加入 GPT-4.1-mini 后，可以直接观察它与原五类来源在结构特征空间中的相对位置。*

---
*本报告由数据流水线自动生成。*
