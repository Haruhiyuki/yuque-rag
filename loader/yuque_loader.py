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
            "Content-Type": "application/json",
        }


    # ---------- 基础 API ---------- #
    def _get(self, url: str) -> dict:
        resp = requests.get(url, headers=self.headers, timeout=20)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_repos(self, group_login: str):
        return self._get(f"{self.base_url}/groups/{group_login}/repos")

    def get_docs_list(self, namespace: str):
        return self._get(f"{self.base_url}/repos/{namespace}/docs")

    def get_doc_content(self, namespace: str, slug: str) -> str:
        url = f"{self.base_url}/repos/{namespace}/docs/{slug}"
        resp = requests.get(url, headers=self.headers, timeout=20)
        resp.raise_for_status()
        return resp.json()["data"]["body"]  # markdown / html

    # ---------- 总入口 ---------- #
    def load_documents(
        self,
        *,
        group_login: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Document]:
        """
        · group_login → 加载该团队下所有 repo
        · namespace   → 仅加载单一 repo
        """
        if not (group_login or namespace):
            raise ValueError("必须提供 group_login 或 namespace 其中之一")

        documents: List[Document] = []

        def _collect(ns: str):
            for meta in self.get_docs_list(ns):
                slug = meta["slug"]
                content = self.get_doc_content(ns, slug)

                metadata = {
                    "repo": ns,
                    "doc_id": meta["id"],
                    "title": meta["title"],
                    "author_name": meta.get("user", {}).get("name", ""),
                    "created_at": meta["created_at"],
                }

                documents.append(Document(page_content=content, metadata=metadata))

        if namespace:
            print(f"读取指定知识库：{namespace}")
            _collect(namespace)
        else:  # group_login mode
            for repo in self.get_repos(group_login):
                ns = repo["namespace"]
                print(f"读取知识库：{ns}")
                _collect(ns)

        return documents
