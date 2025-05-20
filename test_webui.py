# test_webui.py

import streamlit as st
from app import initialize_retriever_and_llm

# 初始化模型与检索器
st.set_page_config(page_title="🔍 调试模式：语雀RAG问答", page_icon="🛠️")

st.title("🛠️ 语雀知识库 RAG 调试工具")
st.markdown("输入你的问题，系统将展示完整的RAG处理过程，包括向量检索、重排序、最终回答。")

# 初始化组件
if "retriever" not in st.session_state or "llm" not in st.session_state:
    with st.spinner("初始化模型与向量库..."):
        retriever, llm = initialize_retriever_and_llm()
        st.session_state.retriever = retriever
        st.session_state.llm = llm

# 输入框
query = st.text_area("请输入你的问题：", height=100, placeholder="例如：四月份团队做了哪些内容？")

if st.button("开始调试") and query.strip():
    with st.spinner("处理中..."):
        retriever = st.session_state.retriever
        llm = st.session_state.llm

        # ===== 1. 向量检索 + rerank =====
        st.subheader("📁 检索结果（Top-K 初选文档）")
        query_vec = retriever._bc_embedding.embed_texts([query])
        scores, docs = retriever._faiss_store.search(query_vec, retriever._top_k_initial)

        for i, doc in enumerate(docs[0]):
            st.markdown(f"**[{i+1}] 文本内容：**")
            st.code(doc.page_content)  # 取消字数限制，显示全文

        # ===== 2. 重排序输出 =====
        st.subheader("🔄 Rerank 重排序结果")
        candidate_texts = [doc.page_content for doc in docs[0]]
        rerank_result = retriever._bc_embedding.rerank(query, candidate_texts)

        for i, (text, score) in enumerate(rerank_result[:retriever._top_k_rerank]):
            st.markdown(f"**{i+1}. 分数：{score:.4f}**")
            st.code(text)  # 取消字数限制，显示全文

        # ===== 3. 构造 Prompt 并生成回答 =====
        st.subheader("🤖 LLM 回答生成")
        reranked_docs = []
        for text, score in rerank_result[:retriever._top_k_rerank]:
            for doc in docs[0]:
                if doc.page_content.startswith(text):
                    reranked_docs.append(doc)
                    break

        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        prompt = f"根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n\n回答："

        st.markdown("**📜 Prompt 构造如下：**")
        st.code(prompt)  # 取消字数限制，显示全文

        # 调用模型生成
        answer = llm.generate(prompt)
        st.success("✅ 回答完成")
        st.markdown(f"**🔍 回答内容：**\n\n> {answer}")
