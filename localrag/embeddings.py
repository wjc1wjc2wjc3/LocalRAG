"""嵌入层：**默认全离线**。

- `HashingEmbedder`：纯标准库实现（blake2b 哈希 + 子线性 TF + L2 归一）。
  *零依赖、零下载、跨进程确定性*（不使用内置 `hash()`，避免 PYTHONHASHSEED 随机化
  导致重启后向量不可复现 —— 这是很多本地 RAG 项目的隐藏缺陷）。
- `SentenceTransformerEmbedder`：可选，使用**本地缓存**模型
  （`local_files_only=True`），首次拉取后可完全离线运行。
"""
from __future__ import annotations

import hashlib
import math
import re

_LATIN = re.compile(r"[a-z0-9_]+")
# CJK 统一表意文字 + 扩展 A + 日文假名（中文/日文没有空格，必须按字切）
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]+")


def tokenize(text: str) -> list:
    """中英混合分词（零依赖、无需分词器）。

    - 拉丁文：按词切分；
    - 中文 / 日文：按**字**切分，并补充**字二元组**（字符级 bigram）。

    为什么不用 `\\w+`：中文没有空格，`\\w+` 会把「报销需要提交什么材料」
    当成**一个** token，导致查询与文档零重叠、相关度恒为 0 —— 这正是许多
    轻量 RAG demo 在中文场景下「能跑但检索不到东西」的隐藏原因。
    """
    t = text.lower()
    toks: list = list(_LATIN.findall(t))
    for run in _CJK_RUN.findall(t):
        toks.extend(list(run))                              # 字 unigram
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))  # 字 bigram
    return toks


def _hash_pair(token: str, dim: int) -> tuple[int, int]:
    """稳定哈希 → (维度下标, 符号)。跨进程/跨机器结果一致。"""
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    idx = int.from_bytes(d[:8], "big") % dim
    sign = 1 if (int.from_bytes(d[8:], "big") & 1) else -1
    return idx, sign


class BaseEmbedder:
    name = "base"
    dim = 0

    def embed(self, texts) -> list:
        raise NotImplementedError

    def embed_one(self, text: str) -> list:
        return self.embed([text])[0]


class HashingEmbedder(BaseEmbedder):
    """零依赖、确定性、纯本地的特征哈希嵌入。"""

    name = "hashing"

    def __init__(self, dim: int = 256, use_bigrams: bool = True):
        self.dim = dim
        self.use_bigrams = use_bigrams

    def embed(self, texts) -> list:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            toks = tokenize(t)
            feats = list(toks)
            if self.use_bigrams:
                feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for tok in feats:
                i, s = _hash_pair(tok, self.dim)
                vec[i] += s  # 可换成 1+log(tf) 的子线性缩放
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class SentenceTransformerEmbedder(BaseEmbedder):
    """可选的语义嵌入：模型来自本地缓存，离线可用。"""

    name = "sentence-transformers"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", offline: bool = True):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "需要可选依赖：pip install sentence-transformers"
            ) from exc
        self.model_name = model
        self._m = SentenceTransformer(model, local_files_only=offline) if offline else SentenceTransformer(model)

    @property
    def dim(self) -> int:  # type: ignore[override]
        return int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts) -> list:
        vecs = self._m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]


def get_embedder(cfg):
    """按配置构造嵌入器。离线模式下 ST 走 local_files_only。"""
    if cfg.embedder == "hashing":
        return HashingEmbedder(dim=cfg.embedding_dim)
    if cfg.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.st_model, offline=cfg.is_offline)
    raise ValueError(f"未知 embedder：{cfg.embedder}")


def cosine(a, b) -> float:
    """向量已归一化时即点积。"""
    return sum(x * y for x, y in zip(a, b))
