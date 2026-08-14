# Transformer 架构

## 概述
Transformer 由 Vaswani 等人在 2017 年论文 "Attention Is All You Need" 中提出，是一种基于自注意力机制的序列到序列模型，摒弃了 RNN 的循环结构。

## 核心组件

### 1. 自注意力机制（Self-Attention）
- **Query (Q), Key (K), Value (V)**: 输入通过三个线性变换得到
- **注意力分数**: Attention(Q,K,V) = softmax(QK^T / √d_k) × V
- **缩放因子 √d_k**: 防止点积过大导致 softmax 梯度消失
- **复杂度**: O(n² × d)，n 为序列长度，d 为维度

### 2. 多头注意力（Multi-Head Attention）
- 将 Q, K, V 分成 h 个头，每个头独立计算注意力
- 拼接后通过线性变换输出
- 优点：可以关注不同位置的不同表示子空间

### 3. 位置编码（Positional Encoding）
- Transformer 没有循环结构，需要额外注入位置信息
- 原始方案：正弦/余弦函数
  - PE(pos, 2i) = sin(pos / 10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- 改进方案：RoPE（旋转位置编码）、ALiBi

### 4. 前馈网络（FFN）
- 两层全连接网络：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
- 中间维度通常是隐藏维度的 4 倍
- 变体：GLU（Gated Linear Unit）系列

### 5. 残差连接 + Layer Normalization
- 每个子层都有残差连接：LayerNorm(x + Sublayer(x))
- Pre-Norm vs Post-Norm：现代模型多用 Pre-Norm

## Encoder-Decoder 结构

### Encoder（编码器）
- N 层堆叠，每层包含：
  1. 多头自注意力
  2. 前馈网络
- 代表模型：BERT

### Decoder（解码器）
- N 层堆叠，每层包含：
  1. 带掩码的多头自注意力（防止看到未来信息）
  2. 编码器-解码器交叉注意力
  3. 前馈网络
- 代表模型：GPT 系列

## 重要变体

### BERT（2018）
- 仅使用 Encoder
- 预训练：MLM（掩码语言模型）+ NSP（下一句预测）
- 适用于：分类、NER、问答等理解任务

### GPT 系列
- 仅使用 Decoder
- 预训练：自回归语言模型（预测下一个 token）
- GPT-3 (175B) → GPT-4 → GPT-4o

### LLaMA / Qwen / DeepSeek
- 现代开源 LLM，基于 Decoder-only 架构
- 改进：RoPE、SwiGLU、RMSNorm、GQA

## 关键论文
1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Devlin et al., "BERT", NAACL 2019
3. Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020
4. Touvron et al., "LLaMA", 2023
