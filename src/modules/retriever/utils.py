import json
import jieba
import glob
import os
from typing import List, Set

def load_json_documents(file_path: str) -> List[dict]:
    """加载JSON格式的文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return documents

def get_document_by_id(documents: List[dict], ids: List[int]) -> List[tuple]:
    """根据ID获取文档内容"""
    results = []
    for doc_id in ids:
        if 0 <= doc_id < len(documents):
            results.append((doc_id, documents[doc_id]["分析准则"]))
    return results

def load_single_stopwords(file_path: str) -> Set[str]:
    """加载单个停用词文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: Stopwords file {file_path} not found")
        return set()

def load_stopwords(stopwords_dir: str = "datasets/stopwords") -> set:
    """加载所有停用词文件并取并集"""
    # 默认停用词
    default_stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                        '或', '一个', '没有', '我们', '你们', '他们', '它们', '这个',
                        '那个', '这些', '那些', '这样', '那样', '之', '的话', '说'}
    
    # 获取目录下所有txt文件
    stopwords_files = glob.glob(os.path.join(stopwords_dir, "*.txt"))
    
    if not stopwords_files:
        print(f"Warning: No stopwords files found in {stopwords_dir}, using default stopwords")
        return default_stopwords
    
    # 读取所有停用词文件并取并集
    all_stopwords = set()
    for file_path in stopwords_files:
        stopwords = load_single_stopwords(file_path)
        all_stopwords.update(stopwords)
    
    # 合并默认停用词
    all_stopwords.update(default_stopwords)
    return all_stopwords

# 全局停用词集合
STOPWORDS = load_stopwords()

def remove_stopwords(words: List[str]) -> List[str]:
    """移除停用词"""
    return [word for word in words if word not in STOPWORDS] 