# llm/openai_llm.py

from openai import OpenAI
from config import OPENAI_MAX_TOKENS

class OpenAILLM:
    def __init__(self, model_name: str, api_key: str, api_base: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

        self.system_prompt = (
            "你是一个基于知识库的智能问答助手，为凉宫春日应援团服务。"
            "当用户提问时，请结合上下文信息，提供准确且简洁的回答。"
            "如果上下文中没有相关信息，礼貌地告诉用户你无法回答该问题。"
            "避免编造信息，保持回答专业且友好。"
        )

    def generate(self, prompt: str, max_tokens: int = OPENAI_MAX_TOKENS, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
