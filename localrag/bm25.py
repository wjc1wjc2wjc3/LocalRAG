"""BM25 词法检索（**纯标准库**）。

为什么需要它：纯向量检索对「专有名词 / 型号 / 编号 / 报错码」这类
**低频但精确**的查询很不友好（向量相似但词不匹配）。成熟 RAG 方案
（Haystack / LlamaIndex / RAGFlow 等）基本都做**混合检索**，本项目补齐这一环，
且不引入任何依赖。
"""
from __future__ import annotations

import math

from .embeddings import tokenize


class BM25:
    """经典 Okapi BM25。构建一次，可多次打分。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.tf: list = []
        self.df: dict = {}
        self.lens: list = []
        self.avg_len = 0.0

    def build(self, texts) -> "BM25":
        self.tf = []
        self.df = {}
        self.lens = []
        for t in texts:
            toks = tokenize(t)
            self.lens.append(len(toks))
            freq: dict = {}
            for tok in toks:
                freq[tok] = freq.get(tok, 0) + 1
            self.tf.append(freq)
            for tok in freq:
                self.df[tok] = self.df.get(tok, 0) + 1
        self.N = len(self.tf)
        self.avg_len = (sum(self.lens) / self.N) if self.N else 0.0
        return self

    def scores(self, query: str) -> list:
        """返回与 texts 一一对应的 BM25 分数列表。"""
        if self.N == 0:
            return []
        q_terms = set(tokenize(query))
        idf_cache: dict = {}
        out = []
        for i in range(self.N):
            dl = self.lens[i] or 1
            score = 0.0
            for term in q_terms:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                if term not in idf_cache:
                    n = self.df.get(term, 0)
                    idf_cache[term] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1.0))
                score += idf_cache[term] * f * (self.k1 + 1) / denom
            out.append(score)
        return out


def rrf_fuse(rank_lists, k: int = 60):
    """Reciprocal Rank Fusion：把多个排序列表融合成一个分数表。

    RRF 不需要对各路分数做归一化，对「向量 + 词法」这种量纲不同的场景最稳。
    返回 {id: rrf_score}。
    """
    fused: dict = {}
    for ranks in rank_lists:
        for rank, item_id in enumerate(ranks, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused
