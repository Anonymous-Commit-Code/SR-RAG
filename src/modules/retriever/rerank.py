import sys
sys.path.append(f'.')
from FlagEmbedding import FlagReranker
import atexit


class Reranker:
    def __init__(self):
        self.model = FlagReranker('models/BAAI/bge-reranker-large', use_fp16=True)
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.model, 'stop_self_pool'):
                self.model.stop_self_pool()
            del self.model
        except:
            pass

    def __del__(self):
        """析构函数"""
        self.cleanup()

    def cal_score(self, query, documents:list[str], batch_size=10):
        query_document_pairs = [[query, document] for document in documents]
        scores = []
        for idx in range(0, len(query_document_pairs), batch_size):
            batch = query_document_pairs[idx:idx+batch_size]
            bscore = self.model.compute_score(batch)
            if len(batch) == 1:
                bscore = [bscore]
            else:
                bscore = list(bscore)
            scores.extend(bscore)
        return scores
    
    def merge_results(self, results1, results2):
        ids, documents=zip(*results1)
        ids=list(ids)
        documents=list(documents)
        for id, document in results2:
            if id not in ids:
                ids.append(id)
                documents.append(document)
        return ids, documents
    
    def rerank(self, query, results1, results2, k=5):
        ids, documents=self.merge_results(results1, results2)
        scores=self.cal_score(query, documents)
        results_score=list(zip(ids,documents, scores))
        sorted_results=sorted(results_score, key=lambda x: x[-1], reverse=True)
        top_k=[(sorted_results[i][0], sorted_results[i][1]) for i in range(min(k, len(sorted_results)))]
        return top_k

if __name__ == "__main__":
    reranker=Reranker()
    question = "test"
    results1=[(1,"rr"),(5,"tt"),(3,"tt"),(10,"test")]
    results2=[(1,"rr"),(78,"txx"),(6,"t"),(32,"testtest")]
    print(reranker.rerank(question,results1,results2))
