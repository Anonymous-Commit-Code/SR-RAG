import sys

sys.path.append("src")
from gensim.models.bm25model import OkapiBM25Model
import jieba, json
from gensim.corpora import Dictionary
from modules.retriever.utils import (
    remove_stopwords,
    get_document_by_id,
    load_json_documents,
)
import functools


def before_method_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.documents_buffer:
            self.dictionary.add_documents(
                [self.cut(doc["分析准则"]) for doc in self.documents_buffer]
            )
            self.documents.extend(self.documents_buffer)
            self.documents_buffer = []
        return func(self, *args, **kwargs)

    return wrapper


class BM25:
    def __init__(self, file_path):
        self.documents = load_json_documents(file_path)
        self.dictionary = Dictionary([])

        # 初始化词典
        for doc in self.documents:
            self.dictionary.add_documents([self.cut(doc["safety_criterion"])])

        self.model = OkapiBM25Model(dictionary=self.dictionary)
        self.documents_buffer = []

    def cut(self, text: str):
        return remove_stopwords(jieba.lcut(text))

    @before_method_decorator
    def tokenize(self, text: str):
        splited_text = remove_stopwords(jieba.lcut(text))
        token = self.dictionary.doc2bow(splited_text)
        return token

    @before_method_decorator
    def get_score(self, query: str, document: str):
        query_token = self.tokenize(query)
        document_token = self.tokenize(document)
        query_words_ids = {x[0] for x in query_token}
        term_frequencies = [x for x in document_token if x[0] in query_words_ids]
        score = 0
        for item in self.model[term_frequencies]:
            score += item[1]
        return score

    @before_method_decorator
    def get_topK(self, query: str, k=5):
        score_list = []
        for index, document in enumerate(self.documents):
            score = self.get_score(query, document["safety_criterion"])
            score_list.append((index, score))

        sorted_score = sorted(score_list, key=lambda x: x[1], reverse=True)
        top_k = [sorted_score[i][0] for i in range(min(k, len(sorted_score)))]
        result_documents = get_document_by_id(self.documents, top_k)
        return result_documents

    def retrieve(self, query: str, k_final: int = 5) -> list[str]:
        results=[]
        for item in self.get_topK(query, k=k_final):
            results.append(item[1])
        return results

    def add_documents(self, documents: list):
        """添加新文档到缓冲区"""
        self.documents_buffer.extend(documents)


if __name__ == "__main__":
    bm25 = BM25("datasets/table/安全性分析准则_书.json")
    print(bm25.retrieve("接口数据分析"))
