import sys
sys.path.append(f'src')
from modules.retriever.rerank import Reranker
from modules.retriever.hnsw import HNSW
from modules.retriever.bm25 import BM25

class Retriever:
    def __init__(self,file_path):
        self.filepath=file_path
        self.bm25=BM25(file_path)
        self.hnsw=HNSW(file_path=file_path)
        self.reranker=Reranker()
    
    def search(self, query, k_retrieval=20, k_rerank=5):
        results1=self.bm25.get_topK(query,k=k_retrieval)
        results2=self.hnsw.get_topK(query,k=k_retrieval)
        return [document[1] for document in self.reranker.rerank(query,results1,results2,k=k_rerank)]

if __name__ == "__main__":
    retriever=Retriever('documents.txt')
    print(retriever.search("环境法中关于水污染的规定"))