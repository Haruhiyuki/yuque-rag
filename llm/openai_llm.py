# llm/openai_llm.py
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from config import OPENAI_MAX_TOKENS


class OpenAILLM:
    def __init__(self, model_name: str, api_key: str, api_base: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

        # 基础 system prompt，**不要放时间**，时间在 generate 时再拼
        self.base_prompt = (
            "你是一个基于知识库的智能问答助手。"
            "当用户提问时，你将收到用户的问题以及从知识库中检索到的信息。请结合检索的信息，提供准确且简洁的回答。"
            "如果上下文中没有相关信息，礼貌地告诉用户你无法回答该问题。"
            "避免编造信息，保持回答专业且友好。"
            "每条检索到的知识库信息之间并没有关联，仅根据它们与问题的关联性被检索出，避免错误认定其上下文关系，将不存在的事实安在某条信息头上。"
            "你可以看到每条检索信息的权重，权重越高更有可能相关，并非所有信息都是真的相关，需要结合问题和权重进行甄别。"
            "你获得的信息是从知识库中检索的，不是用户提供的。只有问题是用户提的。"
            "用户有时提出的问题不是真正的问题，而是打招呼或开玩笑。"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = OPENAI_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> str:
        # —— 在此“临时”拼接当前时间（UTC+8） ——
        utc8_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
        system_prompt = f"【当前时间：{utc8_now} (UTC+8)】\n{self.base_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
