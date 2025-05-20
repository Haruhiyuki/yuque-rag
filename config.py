# config.py

# 语雀团队TOKEN
YUQUE_TOKEN = "YOUR-TOKEN"
# 语雀团队名，如果YUQUE_NAMESPACE为空将获取整个团队的知识库。
YUQUE_GROUP = "YOUR-GROUP"
# 知识库名，格式为“团队名/知识库名”，填写后只会获取单一知识库的文档。
YUQUE_NAMESPACE = None

# 开启问答模式，在存在索引的情况下避免重建。若要更新知识库索引及文本，请关闭该模式。
QA_MODE = True

# 是否使用 OpenAI 接口（兼容支持OPENAPI的模型，False则使用Ollama）
USE_OPENAI = True

# 检索参数，TOP_K_INITIAL控制检索出的文本段数，TOP_K_RERANK控制重排序后返回的文本段数
TOP_K_INITIAL = 20
TOP_K_RERANK = 10

# 通用 OpenAI API 配置
OPENAI_API_KEY = "YOUR-KEY"
OPENAI_MODEL = "deepseek-chat"  # 或 "gpt-3.5-turbo"
OPENAI_API_BASE = "https://api.deepseek.com/v1"  # DeepSeek 的地址
OPENAI_MAX_TOKENS = 8192

# Ollama 模型配置
OLLAMA_MODEL = "deepseek-r1:7b"
OLLAMA_MAX_TOKENS = 4096

# 检索模型相关
BC_EMBED_MODEL = "maidalun1020/bce-embedding-base_v1"
BC_RERANK_MODEL = "maidalun1020/bce-reranker-base_v1"
VECTOR_DIM = 768
FAISS_INDEX_PATH = "./vectorstore/faiss.index"

