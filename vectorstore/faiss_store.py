# vectorstore/faiss_store.py

import os
import faiss
import numpy as np
import pickle

class FaissVectorStore:
    def __init__(self, vector_dim: int, index_path: str):
        """
        vector_dim: 向量维度（例如768）
        index_path: 索引文件存放路径
        """
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.index = None
        self.documents = []  # 与向量顺序对应的文档对象列表

    def create_index(self):
        """创建一个新的FAISS索引"""
        self.index = faiss.IndexFlatIP(self.vector_dim)  # 余弦相似度用内积
        print("FAISS索引创建成功")

    def add_vectors(self, vectors: np.ndarray):
        """
        向索引中添加向量
        vectors: shape (n_samples, vector_dim)
        """
        if self.index is None:
            raise ValueError("索引未初始化，请先调用create_index()或load_index()")
        self.index.add(vectors)
        print(f"向索引添加了 {vectors.shape[0]} 个向量")

    def add_embeddings(self, embeddings: list[list[float]], documents: list):
        """
        同时添加嵌入向量和对应文档
        """
        vectors_np = np.array(embeddings).astype("float32")
        if self.index is None:
            self.create_index()
        self.index.add(vectors_np)
        self.documents.extend(documents)
        print(f"添加了 {len(documents)} 个文本块及其向量")

    def search(self, query_vectors: np.ndarray, top_k: int):
        """
        通过query向量检索top_k结果
        返回: (scores, documents)
        """
        if self.index is None:
            raise ValueError("索引未初始化，无法搜索")
        scores, indices = self.index.search(query_vectors, top_k)
        docs = [[self.documents[i] for i in row] for row in indices]
        return scores, docs

    def save_index(self):
        """保存索引到磁盘"""
        if self.index is None:
            raise ValueError("索引未初始化，无法保存")
        faiss.write_index(self.index, self.index_path)
        print(f"索引已保存到 {self.index_path}")

    def load_index(self):
        """从磁盘加载索引"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"索引已从 {self.index_path} 加载")

    def save(self):
        """保存向量索引和文档到磁盘"""
        self.save_index()
        with open(self.index_path + ".docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        print(f"文档已保存到 {self.index_path}.docs.pkl")

    def load(self):
        """加载向量索引和文档"""
        self.load_index()
        docs_path = self.index_path + ".docs.pkl"
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"文档文件不存在: {docs_path}")
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        print(f"文档已从 {docs_path} 加载")