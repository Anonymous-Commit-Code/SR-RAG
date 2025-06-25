<div align=center>

# SR-RAG: An Adaptive Retrieval-Augmented Framework for Aviation Software Safety Requirement Generation

</div>

## 项目概述

SR-RAG是一个自适应的检索增强框架，专门用于航空软件安全需求生成。该框架结合了多种检索方法和生成模型，能够智能地生成高质量的安全需求。

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 确保必要目录存在
python config.py
```

### 2. 基本使用

```python
from src.modules.inference import InferenceEngine
from config import get_data_path

# 初始化推理引擎
engine = InferenceEngine(get_data_path("knowledge_base"))

# 处理单个需求
result = engine.process_requirement("系统应能够处理飞行控制数据")
print(result)
```

### 3. 配置文件

项目使用 `config.py` 管理所有配置项：

- **数据路径**: 知识库、测试数据等
- **模型参数**: 温度、最大令牌数等
- **检索配置**: K值、阈值等
- **并发设置**: 线程数、重试次数等

## 📁 项目结构

```
SR-RAG/
├── config.py                 # 配置文件 (新增)
├── requirements.txt           # 依赖列表 (已更新)
├── src/
│   ├── modules/
│   │   ├── generator/         # 生成器模块
│   │   ├── retriever/         # 检索器模块 (已修复)
│   │   └── inference.py       # 推理引擎 (已优化)
│   ├── utils/                 # 工具函数
│   │   └── dataloader.py      # 数据加载器 (已实现)
│   └── evaluation/            # 评估模块 (已优化)
├── datasets/                  # 数据集
└── experiments/               # 实验结果
```

## 🔧 主要组件

### 1. 多路检索器 (MultiRetriever)
- 集成BM25和HNSW检索
- 支持重排序和阈值过滤
- 可配置的检索参数

### 2. 生成器模块 (Generators)
- 统一的基础生成器接口
- 支持多种模型后端
- 可配置的提示模板

### 3. 推理引擎 (InferenceEngine)
- 端到端的需求处理流程
- 并行处理和错误恢复
- 灵活的配置选项

## 🛠️ 配置选项

### 模型配置
```python
MODEL_CONFIG = {
    "default_model": "qwen",
    "max_tokens": 16384,
    "temperature": 0.3,
    "max_retries": 5
}
```

### 检索配置
```python
RETRIEVAL_CONFIG = {
    "k_retrieval": 20,
    "k_final": 5,
    "use_rerank": True,
    "consistency_threshold": 0.4
}
```
