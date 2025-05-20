# loader/text_preprocessor.py

import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class TextPreprocessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        chunk_size: 每段最大字符数
        chunk_overlap: 段落之间的重叠字符数
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " "]  # 中文适配
        )

    def clean_text(self, text: str) -> str:
        """去除 HTML 标签、多余空格等"""
        text = re.sub(r"<[^>]+>", "", text)  # 去除 HTML 标签
        text = re.sub(r"\s+", " ", text)     # 合并多空格为单空格
        return text.strip()

    def process_documents(self, docs: List[Document]) -> List[Document]:
        """清洗并切分所有文档"""
        cleaned_docs = [
            Document(page_content=self.clean_text(doc.page_content), metadata=doc.metadata)
            for doc in docs
        ]
        chunks = []
        for doc in cleaned_docs:
            split_docs = self.splitter.split_documents([doc])
            chunks.extend(split_docs)
        return chunks
