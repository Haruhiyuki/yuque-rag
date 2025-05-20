# loader/yuque_loader.py

import requests
from langchain.schema import Document
from typing import List, Optional


class YuqueLoader:
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://www.yuque.com/api/v2"
        self.headers = {
            "X-Auth-Token": self.token,
            "Content-Type": "application/json"
        }

    def get_repos(self, group_login: str) -> List[dict]:
        """获取某个团队下的所有知识库"""
        url = f"{self.base_url}/groups/{group_login}/repos"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_docs_list(self, namespace: str) -> List[dict]:
        """获取某个知识库下的所有文档元数据"""
        url = f"{self.base_url}/repos/{namespace}/docs"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_doc_content(self, namespace: str, slug: str) -> str:
        """获取指定文档内容（Markdown格式）"""
        url = f"{self.base_url}/repos/{namespace}/docs/{slug}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("body", "")

    def load_documents(self,
                       group_login: Optional[str] = None,
                       namespace: Optional[str] = None) -> List[Document]:
        """
        支持两种加载方式：
        - 传 group_login → 获取该团队下所有知识库和所有文档
        - 传 namespace → 获取该知识库下的所有文档
        """
        documents = []
        if namespace:
            print(f"读取指定知识库：{namespace}")
            docs_meta = self.get_docs_list(namespace)
            for doc in docs_meta:
                content = self.get_doc_content(namespace, doc.get("slug"))
                title = doc.get("title", "")
                documents.append(Document(page_content=f"{title}\n{content}", metadata={"repo": namespace, "doc_id": doc.get("id")}))

        elif group_login:
            repos = self.get_repos(group_login)
            for repo in repos:
                ns = repo.get("namespace")  # e.g. "staff-sqlmik/doc-manual"
                print(f"读取知识库：{ns}")
                docs_meta = self.get_docs_list(ns)
                for doc in docs_meta:
                    content = self.get_doc_content(ns, doc.get("slug"))
                    title = doc.get("title", "")
                    documents.append(Document(page_content=f"{title}\n{content}", metadata={"repo": ns, "doc_id": doc.get("id")}))

        else:
            raise ValueError("必须提供 group_login 或 namespace 其中之一")

        return documents