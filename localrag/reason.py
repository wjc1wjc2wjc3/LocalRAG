"""可选的 LLM 推理式章节导航（P1，受 PageIndex 启发）。

设计原则（与 LocalRAG 保持一致）：
- **纯标准库**（`urllib`），不强制安装 `openai` SDK；
- **可选外挂**：默认关闭；只有当显式开启且本地 OpenAI 兼容端点可达时才生效；
- **安全回退**：任何失败（无模型 / 超时 / 解析异常）都返回 `None`，
  检索主链路随即回退到「向量 + BM25 + 结构感知」的离线管线，绝不中断。

它解决的是 PageIndex 的核心卖点之一——让 LLM 像人翻书一样在章节树上
「推理」出相关章节，而不是被动做相似度匹配。区别在于：LocalRAG 把它做成
*锦上添花*而不是*必需*，没有本地模型时系统依旧完整可用。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class TreeReasoner:
    """用本地 LLM 从候选章节中挑出相关章节（返回 0-based 索引列表）。"""

    name = "tree-reasoner"

    def __init__(self, endpoint: str = "http://127.0.0.1:8080/v1",
                 model: str = "auto", timeout: float = 30.0, api_key: str = "",
                 enabled: bool = True):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.enabled = enabled

    def select(self, query: str, candidates: list) -> list | None:
        """`candidates`: `[{"path": str, "snippet": str}, ...]`。

        返回选中的候选索引（0-based）列表；失败/禁用时返回 `None`。
        """
        if not self.enabled or not candidates:
            return None

        numbered = []
        for i, c in enumerate(candidates, 1):
            snippet = (c.get("snippet") or "").replace("\n", " ").strip()
            numbered.append(f"[{i}] {c.get('path', '')}\n    {snippet[:240]}")
        body = "\n\n".join(numbered)

        system = (
            "你是文档结构导航器。给定用户问题与若干文档章节标题/摘要，"
            "只输出一个 JSON 数组，包含最可能与答案相关的章节编号（1-based）。"
            "不要输出任何额外文字、解释或代码块标记。"
        )
        user = f"问题：{query}\n\n候选章节：\n{body}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            arr = json.loads(self._extract_json(content))
            picked = [i - 1 for i in arr
                      if isinstance(i, int) and 1 <= i <= len(candidates)]
            return picked or None
        except (urllib.error.URLError, KeyError, json.JSONDecodeError,
                IndexError, ValueError):
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """从模型可能的「废话/代码块」输出里抠出 JSON 数组。"""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            return text[start:end + 1]
        return "[]"
