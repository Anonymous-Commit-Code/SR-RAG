from modules.retriever.bm25 import BM25
from modules.retriever.hnsw import HNSW
from modules.retriever.rerank import Reranker
from config import RETRIEVAL_CONFIG

class MultiRetriever:
    """多路检索器，集成BM25、向量检索和重排序功能"""
    
    def __init__(self, file_path, use_rerank=None):
        """
        初始化多路检索器
        
        :param file_path: 知识库文档路径
        :param use_rerank: 是否使用重排序，默认使用配置文件设置
        """
        if use_rerank is None:
            use_rerank = RETRIEVAL_CONFIG["use_rerank"]
            
        self.bm25 = BM25(file_path)
        self.hnsw = HNSW(file_path=file_path)
        self.reranker = Reranker() if use_rerank else None
        self.use_rerank = use_rerank

    def retrieve(self, query: str, k_retrieval: int = None, k_final: int = None) -> list[str]:
        """
        执行多路检索
        
        :param query: 查询文本
        :param k_retrieval: 每个检索器返回的结果数量
        :param k_final: 最终返回的结果数量
        :return: 检索结果列表
        """
        if k_retrieval is None:
            k_retrieval = RETRIEVAL_CONFIG["k_retrieval"]
        if k_final is None:
            k_final = RETRIEVAL_CONFIG["k_final"]
        
        # 获取BM25检索结果
        bm25_results = self.bm25.get_topK(query, k=k_retrieval)
        
        # 获取向量检索结果
        hnsw_results = self.hnsw.get_topK(query, k=k_retrieval)
        
        if self.use_rerank:
            # 使用重排序器合并结果
            reranked_results = self.reranker.rerank(
                query, 
                bm25_results, 
                hnsw_results, 
                k=k_final
            )
            return [doc[1] for doc in reranked_results]  # 只返回文档内容
        else:
            # 不使用重排序，简单合并结果
            all_docs = set()
            results = []
            
            # 交替添加两个检索器的结果
            for bm25_doc, hnsw_doc in zip(bm25_results, hnsw_results):
                if bm25_doc[1] not in all_docs:
                    results.append(bm25_doc[1])
                    all_docs.add(bm25_doc[1])
                if hnsw_doc[1] not in all_docs:
                    results.append(hnsw_doc[1])
                    all_docs.add(hnsw_doc[1])
                if len(results) >= k_final:
                    break
                    
            return results[:k_final]

    def add_documents(self, documents: list[str]):
        """
        向知识库添加新文档
        
        :param documents: 要添加的文档列表
        """
        self.bm25.add_documents(documents)
        self.hnsw.add_documents(documents)


if __name__ == "__main__":
    # 使用示例
    from config import get_data_path
    retriever = MultiRetriever(get_data_path("knowledge_base"))
    results = retriever.retrieve("系统应能够根据当前飞行状态和目标状态计算出舵偏角δz、δx和δy")
    print(results) 