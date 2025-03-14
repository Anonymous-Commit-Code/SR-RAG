import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# chroma_client = chromadb.PersistentClient(path="/home/jupyter-yyf-rym/workspace/cail_data/")

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('/home/jupyter-yyf-rym/huggingface/emb_text/BAAI/bge-m3',
                       use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = [model.encode(x)['dense_vecs'].tolist() for x in texts]
        return embeddings


# from datasets import load_dataset
#
# path_to_data = '/home/jupyter-yyf-rym/huggingface/dataset/llm/china-ai-law-challenge/cail2018'
# dataset = load_dataset(path_to_data)
# facts = [data["fact"] for data in dataset['exercise_contest_train']]
# facts_len = len(facts)
# ids = [f"id{i}" for i in range(facts_len)]
#
# batch_size = 1000
# n = facts_len // batch_size + 1
# collection = chroma_client.create_collection(name="exercise_fact_collection", embedding_function=MyEmbeddingFunction())
# for i in range(n):
#     print(i)
#     if (i + 1) * batch_size < facts_len:
#         facts_batch = facts[i * batch_size:(i + 1) * batch_size]
#         ids_batch = ids[i * batch_size:(i + 1) * batch_size]
#     else:
#         facts_batch = facts[i * batch_size:]
#         ids_batch = ids[i * batch_size:]
#     collection.add(
#         documents=facts_batch,
#         ids=ids_batch
#     )

# results = collection.query(
#     query_texts=["Which food is the best?"],
#     n_results=2
# )
#
# print(results)
