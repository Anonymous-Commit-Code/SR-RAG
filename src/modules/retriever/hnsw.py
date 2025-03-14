import sys

sys.path.append("src")
import faiss
from FlagEmbedding import FlagModel
import numpy as np
from modules.retriever.utils import load_json_documents, get_document_by_id
import os
import atexit


class HNSW:
    def __init__(
        self, file_path="", save_path="", M=32, efSearch=32, efConstruction=32
    ):
        """初始化HNSW索引

        Args:
            file_path: JSON文档路径
            save_path: 向量数据库保存路径
            M: HNSW图的最大出度
            efSearch: 搜索时的候选邻居数
            efConstruction: 构建时的候选邻居数
        """
        self.embedding_model = FlagModel(
            "models/BAAI/bge-large-zh-v1.5",
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True,
        )
        # 注册清理函数
        atexit.register(self.cleanup)

        self.documents = load_json_documents(file_path) if file_path else []

        try:
            if save_path and os.path.exists(save_path):
                print(f"Loading existing index from {save_path}")
                self.vecdb = faiss.read_index(save_path)
            else:
                print("Creating new index")
                self.d = self.embedding_model.encode("test").shape[0]
                self.vecdb = faiss.IndexHNSWFlat(self.d, M)
                self.vecdb.hnsw.efConstruction = efConstruction
                self.vecdb.hnsw.efSearch = efSearch
                if self.documents:
                    self.build_vecdb(
                        [doc["safety_criterion"] for doc in self.documents], save_path
                    )
        except Exception as e:
            print(f"Error initializing index: {e}")
            raise

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.embedding_model, 'stop_self_pool'):
                self.embedding_model.stop_self_pool()
            del self.embedding_model
        except:
            pass

    def __del__(self):
        """析构函数"""
        self.cleanup()

    def tokenize(self, texts):
        """文本编码，支持单个文本或文本列表"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedding_model.encode(texts)
        return embeddings.astype(np.float32)  # 确保类型为float32

    def get_topK(self, query: str, k=5):
        """获取TopK相似文档"""
        query_embedding = self.tokenize(query)
        score_list, index_list = self.vecdb.search(query_embedding, k)
        score_list = score_list.reshape(-1)
        index_list = index_list.reshape(-1)
        ans_list = get_document_by_id(self.documents, index_list)
        return ans_list

    def add_documents(self, documents: list):
        """添加新文档"""
        embeddings = self.tokenize(documents)
        self.vecdb.add(embeddings)
    
    def retrieve(self, query: str, k_final: int = 5) -> list[str]:
        results=[]
        for item in self.get_topK(query, k=k_final):
            results.append(item[1])
        return results

    def build_vecdb(self, texts, save_path=None, batch_size=32):
        """构建向量数据库"""
        texts_len = len(texts)
        n = (texts_len + batch_size - 1) // batch_size
        print(f"Building vector database with {texts_len} documents...")
        
        for i in range(n):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, texts_len)
            print(f"Processing batch {i+1}/{n}")
            
            batch = texts[start_idx:end_idx]
            self.add_documents(batch)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Saving index to {save_path}")
            faiss.write_index(self.vecdb, save_path)


if __name__ == "__main__":
    # 使用示例
    index_path = "models/vector_db/safety_guidelines.index"
    hnsw = HNSW(file_path="datasets/table/安全性分析准则_书.json", save_path=index_path)
    print(hnsw.retrieve("在其它阶段，采用飞行控制规律控制舵偏角达到预定的控制目的。"))

    cr="testet"
    reqs=["sas","dasd","te","test"]
    criterion_vec = hnsw.tokenize(cr)
    requirements_vec = hnsw.tokenize(reqs)
    similarity = criterion_vec @ requirements_vec.T
    print(similarity)
