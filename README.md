<div align="center">

# SR-RAG: An Adaptive Retrieval-Augmented Framework for Aviation Software Safety Requirement Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org)

*An intelligent framework for generating high-quality aviation software safety requirements through adaptive retrieval and generation techniques.*

[**中文文档**](README_CN.md) | [**English**](README.md)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Data Organization](#data-organization)
- [Model Links](#model-links)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)

## 🎯 Overview

SR-RAG (Safety Requirement - Retrieval Augmented Generation) is an adaptive framework specifically designed for aviation software safety requirement generation. It combines multiple retrieval methods with advanced generation models to intelligently produce high-quality safety requirements that comply with aviation industry standards.

### Key Components

- **Multi-Modal Retrieval**: Integrates BM25 and HNSW-based semantic search
- **Adaptive Generation**: Smart model selection based on requirement complexity
- **Quality Assurance**: Built-in consistency checking and refinement processes
- **Scalable Architecture**: Supports parallel processing and multiple model backends

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/SR-RAG.git
cd SR-RAG

# Install dependencies
pip install -r requirements.txt

# Initialize directories and configuration
python config.py
```

## 📁 Data Organization

The project follows a structured data organization for easy access and management:

### Dataset Structure

```
datasets/
├── database.json                    # 📚 Knowledge Base (10,666 entries)
│   └── Aviation safety guidelines, standards, and regulations
├── testset/
│   └── gt.json                      # 🧪 Test Set (6,800 test cases)
├── train/                           # 🎯 Training Data (Currently empty - for future extensions)
├── requirements_processed_hypo.json # 🔄 Processed Requirements (5,334 entries)
├── docx/                           # 📄 Original Document Sources
└── stopwords/                      # 🚫 Stop Words for Text Processing
```

### Experimental Results

```
experiments_results/
├── evaluation_result_SR_RAG.txt     # 🏆 Main Framework Results
├── evaluation_result_Qwen.txt       # 🤖 Qwen Model Results
├── evaluation_result_QwQ.txt        # 🤖 QwQ Model Results  
├── evaluation_result_llama.txt      # 🤖 LLaMA Model Results
├── evaluation_result_BM25.txt       # 🔍 BM25 Baseline Results
├── evaluation_result_HNSW.txt       # 🔍 HNSW Baseline Results
└── evaluation_result_*.txt          # 📊 Other Experimental Variants
```

## 🔗 Model Links

## ⚙️ Configuration

The framework uses a centralized configuration system in `config.py`:

### Model Configuration

```python
MODEL_CONFIG = {
    "default_model": "qwen",           # Default generation model
    "max_tokens": 16384,               # Maximum output tokens
    "temperature": 0.3,                # Generation randomness
    "max_retries": 5                   # API retry attempts
}
```

### Retrieval Configuration

```python
RETRIEVAL_CONFIG = {
    "k_retrieval": 20,                 # Initial retrieval count
    "k_final": 5,                      # Final document count
    "use_rerank": True,                # Enable reranking
    "consistency_threshold": 0.4       # Consistency filtering threshold
}
```

### Performance Configuration

```python
CONCURRENCY_CONFIG = {
    "max_workers": 10,                 # Parallel processing threads
    "max_retries": 3                   # Error retry attempts
}
```

## 🛠️ Usage

### Command Line Interface

```bash
# Run evaluation on test set
python -m src.evaluation.evaluator

# Process single requirement
python -m src.modules.inference --requirement "Your requirement text"

# Batch processing
python -m src.modules.inference --input_file "requirements.json"
```

### Python API

```python
from src.modules.retriever.multi_retriever import MultiRetriever
from src.modules.generator.base_generator import BaseGenerator
from src.modules.inference import InferenceEngine

# Initialize components
retriever = MultiRetriever(knowledge_base_path="datasets/database.json")
generator = BaseGenerator(model_name="qwen")
engine = InferenceEngine(retriever, generator)

# Generate requirements
result = engine.process_requirement(
    requirement="The system shall ensure data integrity",
    context="Flight control system"
)
```

## 📊 Evaluation

### Metrics

The framework evaluates generated requirements using multiple metrics:

- **BERT Score**: Semantic similarity measurement
- **ROUGE Score**: N-gram overlap evaluation  
- **Consistency Score**: Internal coherence assessment
- **Coverage Score**: Knowledge base utilization

### Running Evaluation

```bash
# Full evaluation on test set
python -m src.evaluation.evaluator --config_path config.py

# Custom evaluation
python -m src.evaluation.evaluator \
    --test_file datasets/testset/gt.json \
    --output_dir experiments_results/ \
    --model qwen
```

<div align="center">

**🔗 Links:** [Homepage](https://your-website.com) | [Documentation](https://docs.your-website.com) | [Issues](https://github.com/your-username/SR-RAG/issues)

Made with ❤️ for Aviation Safety

</div>
