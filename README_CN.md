<div align="center">

# SR-RAG: 面向航空软件安全需求生成的自适应检索增强框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org)

*通过自适应检索和生成技术智能生成高质量航空软件安全需求的框架*

[**中文文档**](README_CN.md) | [**English**](README.md)

</div>

## 📋 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [快速开始](#快速开始)
- [数据组织](#数据组织)
- [模型链接](#模型链接)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [评估体系](#评估体系)

## 🎯 项目概述

SR-RAG（安全需求 - 检索增强生成）是一个专门为航空软件安全需求生成设计的自适应框架。它结合了多种检索方法和先进的生成模型，能够智能地生成符合航空行业标准的高质量安全需求。

### 核心组件

- **多模态检索**: 集成BM25和HNSW语义搜索
- **自适应生成**: 基于需求复杂度的智能模型选择
- **质量保证**: 内置一致性检查和精炼过程
- **可扩展架构**: 支持并行处理和多模型后端

## 🚀 快速开始

### 系统要求

- Python 3.10+
- CUDA兼容GPU（推荐）
- 至少16GB内存

### 安装配置

```bash
# 克隆仓库
git clone https://github.com/your-username/SR-RAG.git
cd SR-RAG

# 安装依赖
pip install -r requirements.txt

# 初始化目录和配置
python config.py
```

## 📁 数据组织

项目采用结构化的数据组织方式，便于访问和管理：

### 数据集结构

```
datasets/
├── database.json                    # 📚 知识库（10,666条记录）
│   └── 航空安全指南、标准和法规
├── testset/
│   └── gt.json                      # 🧪 测试集（6,800个测试用例）
├── train/                           # 🎯 训练数据（当前为空 - 预留扩展）
├── requirements_processed_hypo.json # 🔄 已处理需求（5,334条记录）
├── docx/                           # 📄 原始文档源
└── stopwords/                      # 🚫 文本处理停用词
```

### 实验结果

```
experiments_results/
├── evaluation_result_SR_RAG.txt     # 🏆 主框架结果
├── evaluation_result_Qwen.txt       # 🤖 Qwen模型结果
├── evaluation_result_QwQ.txt        # 🤖 QwQ模型结果  
├── evaluation_result_llama.txt      # 🤖 LLaMA模型结果
├── evaluation_result_BM25.txt       # 🔍 BM25基线结果
├── evaluation_result_HNSW.txt       # 🔍 HNSW基线结果
└── evaluation_result_*.txt          # 📊 其他实验变体
```

## 🔗 模型链接

### 预训练模型

| 模型类型 | 平台 | 下载链接 | 描述 |
|----------|------|----------|------|
| **SR-RAG完整版** | ModelScope | [🔗 下载](https://www.modelscope.cn/models/lurengu/SR-RAG) | SR-RAG LoRA |

> **📋 说明**: 更多模型变体和训练检查点将逐步发布。

## ⚙️ 配置说明

框架使用`config.py`中的集中配置系统：

### 模型配置

```python
MODEL_CONFIG = {
    "default_model": "qwen",           # 默认生成模型
    "max_tokens": 16384,               # 最大输出令牌数
    "temperature": 0.3,                # 生成随机性
    "max_retries": 5                   # API重试次数
}
```

### 检索配置

```python
RETRIEVAL_CONFIG = {
    "k_retrieval": 20,                 # 初始检索数量
    "k_final": 5,                      # 最终文档数量
    "use_rerank": True,                # 启用重排序
    "consistency_threshold": 0.4       # 一致性过滤阈值
}
```

### 性能配置

```python
CONCURRENCY_CONFIG = {
    "max_workers": 10,                 # 并行处理线程数
    "max_retries": 3                   # 错误重试次数
}
```

## 🛠️ 使用方法

### 命令行界面

```bash
# 在测试集上运行评估
python -m src.evaluation.evaluator

# 处理单个需求
python -m src.modules.inference --requirement "您的需求文本"

# 批量处理
python -m src.modules.inference --input_file "requirements.json"
```

### Python API

```python
from src.modules.retriever.multi_retriever import MultiRetriever
from src.modules.generator.base_generator import BaseGenerator
from src.modules.inference import InferenceEngine

# 初始化组件
retriever = MultiRetriever(knowledge_base_path="datasets/database.json")
generator = BaseGenerator(model_name="qwen")
engine = InferenceEngine(retriever, generator)

# 生成需求
result = engine.process_requirement(
    requirement="系统应确保数据完整性",
    context="飞行控制系统"
)
```

## 📊 评估体系

### 评估指标

框架使用多种指标评估生成的需求：

- **BERT评分**: 语义相似度测量
- **ROUGE评分**: N-gram重叠评估
- **一致性评分**: 内部连贯性评估
- **覆盖度评分**: 知识库利用评估

### 运行评估

```bash
# 在测试集上完整评估
python -m src.evaluation.evaluator --config_path config.py

# 自定义评估
python -m src.evaluation.evaluator \
    --test_file datasets/testset/gt.json \
    --output_dir experiments_results/ \
    --model qwen
```

<div align="center">

**🔗 链接:** [主页](https://your-website.com) | [文档](https://docs.your-website.com) | [问题反馈](https://github.com/your-username/SR-RAG/issues)

为航空安全而制作 ❤️

</div> 