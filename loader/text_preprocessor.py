# loader/text_preprocessor.py

import re
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

_HTML_TAG_RE = re.compile(
    r"""</?                     # < 或 </
        [a-zA-Z][\w\-]*         # 标签名
        (?:\s+[^<>]*?)?         # 可选属性
        /?>                     # 结束 >
    """,
    re.VERBOSE,
)


class TextPreprocessor:
    def __init__(self, chunk_size: int = 460, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " "],
        )

    # ---------- 基础清洗 ---------- #
    def clean_text(self, text: str) -> str:
        """移除 HTML 标签、收缩多空白，不触碰换行"""
        text = _HTML_TAG_RE.sub("", text)
        text = re.sub(r"[ \t\f\v]+", " ", text)  # 合并连续空格
        return text.strip()

    # ---------- 入口 ---------- #
    def process_documents(self, docs: List[Document]) -> List[Document]:
        """清洗 + 切分 + 将元信息注入到每个子块中"""
        # 1) 先整体清洗
        cleaned_docs = [
            Document(page_content=self.clean_text(d.page_content), metadata=d.metadata)
            for d in docs
        ]

        # 2) 切分
        chunks: List[Document] = []
        for doc in cleaned_docs:
            chunks.extend(self.splitter.split_documents([doc]))

        # 3) 把关键信息写入文本头
        enriched_chunks: List[Document] = []
        for c in chunks:
            meta = c.metadata

            # 只挑最能帮助语义判断的字段
            meta_head = " | ".join(
                str(meta[k])
                for k in ("title", "author_name", "created_at")
                if meta.get(k)
            )
            if meta_head:
                c.page_content = f"[{meta_head}]\n{c.page_content}"

            enriched_chunks.append(c)

        return enriched_chunks
