"""检索效果评测（零依赖）。

为什么要自带评测：只有可量化的指标，才能证明「换了切分策略 / 开了混合检索」
到底是变好还是变差。多数轻量 RAG 项目完全靠手感调参，本项目给出
`recall@k` 与 `MRR` 两个指标，配合 `--json` 可进 CI。

qrels 格式（JSON 数组）：
    [
      {"query": "报销需要什么材料", "relevant": ["finance_policy.txt"]},
      {"query": "年假多少天",       "relevant": ["hr_policy.txt"]}
    ]
`relevant` 里写**文档路径或标题的片段**，命中即算相关。
"""
from __future__ import annotations

import json


def evaluate(engine, qrels, top_k: int = 5) -> dict:
    """计算 recall@k 与 MRR。"""
    recalls, mrrs, per_case = [], [], []
    for case in qrels:
        query = case.get("query", "")
        relevant = [str(r).lower() for r in case.get("relevant", [])]
        hits = engine.retrieve(query, top_k=top_k)
        ranks = []
        for i, c in enumerate(hits, start=1):
            source = f"{c.path} {c.title}".lower()
            if any(r in source for r in relevant):
                ranks.append(i)
        hit = bool(ranks)
        recalls.append(1.0 if hit else 0.0)
        mrrs.append(1.0 / ranks[0] if ranks else 0.0)
        per_case.append({"query": query, "recall": 1.0 if hit else 0.0,
                         "first_rank": ranks[0] if ranks else None})
    n = len(qrels) or 1
    return {
        "cases": len(qrels),
        "top_k": top_k,
        f"recall@{top_k}": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "details": per_case,
    }


def load_qrels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
