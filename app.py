# app.py

from loader.yuque_loader import YuqueLoader
from loader.text_preprocessor import TextPreprocessor
from embedder.bc_embedding import BCEmbeddingWrapper
from vectorstore.faiss_store import FaissVectorStore
from retriever.rerank_retriever import RerankRetriever
from llm.ollama_llm import OllamaLLM
from llm.openai_llm import OpenAILLM

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import (
    FAISS_INDEX_PATH,
    VECTOR_DIM,
    BC_EMBED_MODEL,
    BC_RERANK_MODEL,
    OLLAMA_MODEL,
    USE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    QA_MODE,
    TOP_K_INITIAL,
    TOP_K_RERANK,
    YUQUE_TOKEN,
    YUQUE_GROUP,
    YUQUE_NAMESPACE
)

def initialize_retriever_and_llm():
    # ===== 配置部分 =====
    yuque_token = YUQUE_TOKEN
    group_login = YUQUE_GROUP            # 团队名（加载整个团队知识库）
    namespace = YUQUE_NAMESPACE          # 不为空则加载单一知识库

    embed_model_name = BC_EMBED_MODEL
    rerank_model_name = BC_RERANK_MODEL

    # ===== 尝试加载已有索引 =====
    faiss_store = FaissVectorStore(vector_dim=VECTOR_DIM, index_path=FAISS_INDEX_PATH)
    index_exists = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_PATH + ".docs.pkl")

    # ===== 加载BCEmbedding模型（预处理模式和问答模式都用到）=====
    print("🧠 初始化 BCEmbedding 模型...")
    bc_embedding = BCEmbeddingWrapper(embed_model_name, rerank_model_name)

    # 如果没有找到索引及文档，或关闭了纯问答模式
    if not index_exists or not QA_MODE:
        print("🧩 进入数据持久化模式...")

        # ===== 加载文档 =====
        print("🔍 正在加载语雀知识库文档...")
        loader = YuqueLoader(yuque_token)
        documents = loader.load_documents(group_login=group_login, namespace=namespace)
        print(f"✅ 加载完成，共 {len(documents)} 篇文档")

        # ===== 文本预处理（清洗与切分） =====
        print("🧹 正在清洗与切分文档...")
        preprocessor = TextPreprocessor()
        documents = preprocessor.process_documents(documents)
        print(f"✅ 清洗与切分完成，共 {len(documents)} 段文本")


        # ===== 向量化文档内容 =====
        print("🔢 正在生成文档向量...")
        texts = [doc.page_content for doc in documents]
        embeddings = bc_embedding.embed_texts(texts)

        # ===== 构建向量索引 =====
        print("📦 构建 FAISS 向量索引...")
        faiss_store.add_embeddings(embeddings, documents)
        faiss_store.save()
        print("✅ 数据持久化完成")

    else:
        print("📥 加载已有向量索引...")
        faiss_store.load()

    documents = faiss_store.documents

    # ===== 检索器（带重排序） =====
    retriever = RerankRetriever(
        faiss_store=faiss_store,
        bc_embedding_wrapper=bc_embedding,
        documents=documents,
        top_k_initial=TOP_K_INITIAL,
        top_k_rerank=TOP_K_RERANK
    )

    # ===== 初始化 LLM =====
    if USE_OPENAI:
        print("🌐 使用 OpenAI API 模型")
        llm = OpenAILLM(
            model_name=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_API_BASE  # ✅ 传入 base_url
        )
    else:
        print("🖥️ 使用本地 Ollama 模型")
        llm = OllamaLLM(OLLAMA_MODEL)

    return retriever, llm

def run_cli_loop(retriever, llm):
    # ===== 命令行问答循环 =====
    print("\n🎤 请输入你的问题（输入 exit 退出）")
    while True:
        query = input("\n🧾 你的问题：").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 退出问答系统")
            break

        # 检索文档 → 构造上下文
        relevant_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 构造 Prompt
        prompt = f"根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n\n回答："

        # 调用本地模型生成回答
        answer = llm.generate(prompt)
        print(f"\n🤖 回答：{answer}")

if __name__ == "__main__":
    retriever, llm = initialize_retriever_and_llm()
    run_cli_loop(retriever, llm)