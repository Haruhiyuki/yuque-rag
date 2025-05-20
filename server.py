# server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app import initialize_retriever_and_llm

# 初始化 RAG 模型
retriever, llm = initialize_retriever_and_llm()

app = FastAPI()

# 允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体模型
class QueryRequest(BaseModel):
    question: str


# 对话 API
@app.post("/chat")
def chat(req: QueryRequest):
    query = req.question.strip()
    if not query:
        return {"answer": "❗请输入问题"}

    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n\n回答："
    answer = llm.generate(prompt)

    return {"answer": answer}
