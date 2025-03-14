import sys
import json
sys.path.append("src")
from modules.retriever.hnsw import HNSW
from modules.retriever.rerank import Reranker
from typing import List, Dict, Any
import os

class HierarchicalRetriever:
    """实现分层检索系统的类"""
    
    def __init__(self, file_path):
        """初始化分层检索系统
        
        参数:
            file_path: JSON文档文件路径
        """
        # 存储文档结构
        self.documents = []        # 原始文档列表
        
        # 用于需求的向量数据库
        self.vector_db = HNSW("")
        
        # 用于最终结果的Re-ranker
        self.reranker = Reranker()
        
        # 处理文档并建立索引
        self._build_index(file_path)
    
    def _build_index(self, file_path):
        """根据文档构建分层索引"""
        # 从JSON文件中加载文档
        self.documents = self._load_documents(file_path)
        
        # 将文档添加到向量数据库中
        for doc in self.documents:
            # 使用需求文本作为向量数据库的文档内容
            self.vector_db.add_documents([doc["hopy_requirement"]])
    
    def _load_documents(self, file_path):
        """从JSON文件中加载文档
        
        返回: 文档列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents = json.load(file)
                return documents
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    def retrieve(self, query, k_retrieval=20, k_final=5):
        """使用分层结构进行检索
        
        参数:
            query: 检索查询
            k_retrieval: 首轮检索返回的数量
            k_final: 最终返回的数量
        """
        # 1. 在向量数据库中搜索需求
        requirement_results = self.vector_db.retrieve(query, k_final=k_retrieval)
        
        # 2. 将检索结果与原始文档对应
        structured_results = []
        for req_text in requirement_results:
            # 找到对应的文档
            for doc in self.documents:
                if doc["hopy_requirement"] == req_text:
                    doc_path=os.path.join("datasets/docx",doc["document_name"])
                    with open(doc_path, "r", encoding="utf-8") as json_file:
                        doc_chunk_content=json.load(json_file)[doc["document_chunk_index"]]
                    structured_results.append({
                        "parent_doc": doc_chunk_content,
                        "criterion": doc["safety_criterion"],
                        "requirement": doc["hopy_requirement"],
                        "class": doc["class"]
                    })
                    break
        
        # 3. 对最终结果进行Re-rank
        if len(structured_results) > k_final:
            combined_texts = [
                f"{r['class']} {r['criterion']} {r['requirement']}"
                for r in structured_results
            ]
            scores = self.reranker.cal_score(query, combined_texts)
            results_with_scores = list(zip(structured_results, scores))
            sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
            structured_results = [r for r, _ in sorted_results[:k_final]]
        
        return structured_results[:k_final]


if __name__ == "__main__":
    retriever = HierarchicalRetriever("datasets/table/安全性分析准则_书.json")
    
    results = retriever.retrieve("接口数据分析")
    
    for result in results:
        print("文档:", result["parent_doc"])
        print("分类:", result["class"])
        print("安全准则:", result["criterion"])
        print("需求:", result["requirement"])
        print("---")
