import os
from pathlib import Path

try:
    from api_keys import API_KEYS
except ImportError:
    API_KEYS = {}

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据路径配置
DATA_PATHS = {
    "knowledge_base": "datasets/database.json",
    "test_data": "datasets/testset/gt.json",
    "stopwords_dir": "datasets/stopwords",
    "vector_db": "models/vector_db/safety_guidelines.index",
    "embedding_model": "models/BAAI/bge-large-zh-v1.5"
}

# 提示模板路径配置
PROMPT_TEMPLATES = {
    "refine": "src/modules/generator/prompt/prompt_template/refine.txt",
    "classify": "src/modules/generator/prompt/prompt_template/classify.txt",
    "filter": "src/modules/generator/prompt/prompt_template/filter.txt",
    "criterion_rewrite": "src/modules/generator/prompt/prompt_template/criterion_rewrite.txt",
    "consistent": "src/modules/generator/prompt/prompt_template/consistent.txt",
    "safety": "src/modules/generator/prompt/prompt_template/safety.txt",
    "query_rewrite": "src/modules/generator/prompt/prompt_template/query_rewrite.txt",
    "straight_generate_requirement": "src/modules/generator/prompt/prompt_template/straight_generate_requirement.txt",
    "straight_with_retrieval": "src/modules/generator/prompt/prompt_template/straight_with_retrieval.txt",
    "hypo": "src/modules/generator/prompt/prompt_template/hypo.txt"
}

# 模型端点配置
MODEL_ENDPOINTS = {
    "qwen": "http://localhost:8001/v1",
    "qwq": "http://localhost:8002/v1",
    "llama": "http://localhost:8003/v1",
    "vllm": "http://localhost:8004/v1",
    "deepseek-r1": "https://api.deepseek.com/v1",
    "deepseek-v3": "https://api.deepseek.com/v1",
    "rerank": "http://localhost:8001/v1",
    "classify": "http://localhost:8001/v1",
    "filter": "http://localhost:8001/v1",
}

# 模型配置
MODEL_CONFIG = {
    "default_model": "qwen",
    "max_tokens": 16384,
    "temperature": 0.3,
    "top_p": 1.0,
    "max_retries": 5,
    "retry_delay": {"multiplier": 2, "min": 4, "max": 10},
    "api_keys": API_KEYS,
    "endpoints": MODEL_ENDPOINTS,
}

# 检索配置
RETRIEVAL_CONFIG = {
    "k_retrieval": 20,
    "k_final": 5,
    "use_rerank": True,
    "bert_threshold": 0.0,
    "rouge_threshold": 0.0,
    "consistency_threshold": 0.4
}

# HNSW配置
HNSW_CONFIG = {
    "M": 32,
    "efSearch": 32,
    "efConstruction": 32,
    "batch_size": 32
}

# 并发配置
CONCURRENCY_CONFIG = {
    "max_workers": min(10, os.cpu_count() or 1),
    "max_retries": 3
}

# 输出配置
OUTPUT_CONFIG = {
    "output_dir": "output_chunks",
    "final_output": "requirements_processed.json",
    "log_level": "INFO"
}

# 评估配置
EVALUATION_CONFIG = {
    "bert_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "rouge_types": ['rouge1', 'rouge2', 'rougeL'],
    "use_stemmer": False
}


def get_absolute_path(relative_path: str) -> str:
    """将相对路径转换为绝对路径"""
    return str(PROJECT_ROOT / relative_path)


def get_data_path(key: str) -> str:
    """获取数据文件的绝对路径"""
    if key in DATA_PATHS:
        return get_absolute_path(DATA_PATHS[key])
    raise ValueError(f"Unknown data path key: {key}")


def get_prompt_template_path(key: str) -> str:
    """获取提示模板的绝对路径"""
    if key in PROMPT_TEMPLATES:
        return get_absolute_path(PROMPT_TEMPLATES[key])
    raise ValueError(f"Unknown prompt template key: {key}")


def ensure_directories():
    """确保必要的目录存在"""
    dirs_to_create = [
        "models/vector_db",
        "output_chunks",
        "datasets/stopwords",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Sample data path: {get_data_path('knowledge_base')}")
    print(f"Sample template path: {get_prompt_template_path('refine')}") 