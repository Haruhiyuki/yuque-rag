# embedder/bc_embedding.py

from BCEmbedding import EmbeddingModel, RerankerModel

class BCEmbeddingWrapper:
    def __init__(self, embed_model_name: str, rerank_model_name: str):
        self.embed_model = EmbeddingModel(model_name_or_path=embed_model_name)
        self.rerank_model = RerankerModel(model_name_or_path=rerank_model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_model.encode(texts)

    def rerank(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        result = self.rerank_model.rerank(query, candidates)

        # 防止是dict结构（compute_score格式）
        if isinstance(result, dict) and 'rerank_passages' in result and 'rerank_scores' in result:
            passages = result['rerank_passages']
            scores = result['rerank_scores']
            return list(zip(passages, scores))

        return result
