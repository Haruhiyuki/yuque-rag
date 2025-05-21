<h1 align="center">yuque-rag</h1>

<div align="center">结合 <b>语雀知识库</b> 与本地/远程大模型的 <b>RAG（检索增强生成）</b>问答系统。</div>



## ✨ 功能

- 📥 自动加载语雀团队或知识库内容
- 🧹 文档切分、清洗与向量化
- 🔍 两阶段检索（向量 + 重排序）
- 🤖 本地 LLM（如 Ollama）或 OpenAI 接口（如 DeepSeek）
- 📦 FastAPI API 接口
- 🔌 数据持久化

## 🗂️ 项目结构

```bash
yuque_rag_project/
├── app.py                    # 主入口（测试问答）
├── server.py                 # fastapi服务端
├── test_webui.py             # 调试页面，可以输出两阶段检索结果
├── loader/
│   ├── yuque_loader.py       # 语雀知识库导入
│   └── text_preprocessor.py  # 文本切分与清洗
├── embedder/
│   └── bc_embedding.py       # 向量化与重排序模块（BCEmbedding）
├── retriever/
│   └── rerank_retriever.py   # 基于 LangChain Retriever 的重排序包装器
├── vectorstore/
│   └── faiss_store.py        # 向量存储初始化与持久化
├── llm/
│   ├── ollama_llm.py         # 本地模型调用模块（通过 Ollama）
│   └── openai_llm.py         # OpenAI 模型调用模块
├── config.py                 # 配置项
└── requirements.txt          # 项目依赖列表
```

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 配置参数

Token获取请参考[官方文档](https://www.yuque.com/yuque/developer/api#sAVSW)

```python
# 语雀团队名，如果YUQUE_NAMESPACE为空将获取整个团队的知识库。
YUQUE_GROUP = "Your-Group"
# 知识库的名称，格式为“团队名/知识库名”，填写后只会获取单一知识库。
YUQUE_NAMESPACE = None          

# 开启问答模式，在存在索引的情况下避免重建。若要更新知识库索引及文本，请关闭该模式。
QA_MODE = True

# 是否使用 OpenAI 接口（兼容支持OPENAPI的模型，False则使用Ollama）
USE_OPENAI = True

# 检索参数，TOP_K_INITIAL控制检索出的文本段数，TOP_K_RERANK控制重排序后返回的文本段数
TOP_K_INITIAL = 20
TOP_K_RERANK = 10

# 通用 OpenAI API 配置
OPENAI_MODEL = "deepseek-chat"  
OPENAI_API_BASE = "https://api.deepseek.com/v1"  
OPENAI_MAX_TOKENS = 8192

# Ollama 模型配置
OLLAMA_MODEL = "Your-Model"
OLLAMA_MAX_TOKENS = 4096
```

### 3️⃣ 运行主程序
```bash
python app.py
```

首次运行将进行：
- 模型下载（若耗时较久可打开代理）

- 语雀知识库加载

- 文本清洗切分

- 嵌入与向量构建

- FAISS 索引保存

## 🌐 FastAPI 服务

运行：
```bash
uvicorn server:app --reload
```

请求示例（POST `/chat`）：
```json
{
  "question": "四月语雀有哪些更新？"
}
```
返回示例：
```json
{
  "answer": "四月语雀的更新包括新增了团队协作功能，优化了文档编辑体验，以及增强了安全策略。"
}
```

## 🧪 调试页面

```bash
streamlit run test_webui.py
```

功能：

- 输出向量检索结果
- 输出 Rerank 结果
- 输出 prompt 与回答

## 🧠 模型说明

### 📌 嵌入模型：
`maidalun1020/bce-embedding-base_v1`

### 📌 重排序模型：
`maidalun1020/bce-reranker-base_v1`

致谢：[NetEase Youdao. BCEmbedding: Bilingual and Crosslingual Embedding for RAG, 2023.](https://github.com/netease-youdao/BCEmbedding)


## ✅TODO

- [ ] 支持个人用户的知识库获取
- [ ] 累进模式，多次获取叠加并去重
- [ ] 不同知识库数据的单独管理


## 📜 License

本项目采用[Apache 2.0 License](https://github.com/netease-youdao/BCEmbedding/blob/master/LICENSE)
