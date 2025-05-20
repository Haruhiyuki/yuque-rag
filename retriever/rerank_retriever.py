# retriever/rerank_retriever.py

from langchain.schema import BaseRetriever, Document
from typing import List
import numpy as np

class RerankRetriever(BaseRetriever):
    def __init__(
        self,
        faiss_store,
        bc_embedding_wrapper,
        documents: List[Document],
        top_k_initial: int ,
        top_k_rerank: int
    ):
        super().__init__()
        self._faiss_store = faiss_store
        self._bc_embedding = bc_embedding_wrapper
        self._documents = documents
        self._top_k_initial = top_k_initial
        self._top_k_rerank = top_k_rerank

    def invoke(self, query: str, **kwargs) -> List[Document]:
        # 向量化查询
        query_vec = np.array(self._bc_embedding.embed_texts([query]), dtype=np.float32)

        # 检索 top_k_initial 个文档
        scores, results = self._faiss_store.search(query_vec, self._top_k_initial)
        candidates = results[0]  # 拿第一个 query 的检索结果
        candidate_texts = [doc.page_content for doc in candidates]

        # 使用 Reranker 重排序
        reranked = self._bc_embedding.rerank(query, candidate_texts)
        # 返回 top_k_rerank 的文档对象
        reranked_docs = []
        for text, score in reranked[:self._top_k_rerank]:
            for doc in candidates:
                if doc.page_content.startswith(text):
                    reranked_docs.append(doc)
                    break

        return reranked_docs

    # 如果你还想保留 get_relevant_documents，可以写成调用 invoke
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # 警告：这个方法会被弃用，不建议长期使用
        return self.invoke(query)