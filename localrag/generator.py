"""答案生成：默认**不需要任何 LLM**，可插拔升级到本地模型。

- `ExtractiveGenerator`：抽取式作答，零依赖、零出网，答案每句都带引用标记。
  让「没有 GPU / 没有模型」的用户也能立刻用起来。
- `OpenAICompatGenerator`：调用**本地** OpenAI 兼容服务
  （如本仓库配套的 CPUEdgeInference：http://127.0.0.1:8080/v1），
  流量不出本机；用标准库 urllib 实现，不强制安装 openai SDK。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .embeddings import tokenize

_SENT_SPLIT = re.compile(r"(?<=[。．\.!?！？\n])")


class BaseGenerator:
    name = "base"

    def generate(self, query: str, citations) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """抽取式生成器：无模型依赖，答案附带 [n] 引用。"""

    name = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, query: str, citations) -> str:
        q_terms = set(tokenize(query))
        parts = []
        for i, c in enumerate(citations, start=1):
            sents = [s for s in _SENT_SPLIT.split(c.text) if s and s.strip()]
            if not sents:
                sents = [c.text]
            scored = sorted(
                sents,
                key=lambda s: len(q_terms & set(tokenize(s))),
                reverse=True,
            )
            picked = [s.strip() for s in scored[: self.max_sentences] if s.strip()]
            if picked:
                parts.append(" ".join(picked) + f" [{i}]")
        if not parts:
            return "（未检索到与问题相关的内容）"
        return "\n".join(parts)


class OpenAICompatGenerator(BaseGenerator):
    """调用任意 OpenAI 兼容端点：本地服务（如 CPUEdgeInference）或云端厂商均可。

    - `base_url` 为空时由 `LocalRAG` 回退到 `ExtractiveGenerator`（零 LLM、全离线）；
    - 云端厂商（OpenAI / 兼容服务）需传 `api_key`，将作为 `Authorization: Bearer` 发送；
      本地模型通常留空即可。
    全程用标准库 `urllib` 实现，不强制安装 `openai` SDK。
    """

    name = "openai-compat"

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1",
                 model: str = "auto", timeout: float = 60.0, api_key: str = "",
                 system_prompt: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "你是一个严谨的本地知识库助手。只能依据给定的【资料】回答，"
            "并在句末用 [n] 标注引用来源编号；资料中没有的内容必须明确说明不知道。"
        )

    def _build_prompt(self, query: str, citations) -> str:
        blocks = []
        for i, c in enumerate(citations, start=1):
            loc = f"{c.title} p{c.page}" if getattr(c, "page", None) else c.title
            blocks.append(f"[{i}] 来源：{loc}\n{c.text}")
        return "【资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】\n{query}"

    def generate(self, query: str, citations) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_prompt(query, citations)},
            ],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            return (
                f"[LLM 端点不可用：{exc}]\n"
                "回退为抽取式答案：\n" + ExtractiveGenerator().generate(query, citations)
            )
