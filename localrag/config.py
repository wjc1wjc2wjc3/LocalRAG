"""LocalRAG 配置。

默认策略：**全离线、零遥测**。索引与检索全过程不产生任何出网请求；
`offline_only=True` 时会同时关闭 HuggingFace / Transformers 的联网与遥测，
任何需要网络的组件都必须显式声明。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # ---- 存储 ----
    db_path: str = "localrag.db"          # 单文件 SQLite，无外部数据库
    audit_path: str = "audit.jsonl"       # 审计日志（哈希链，防篡改）

    # ---- 切分 ----
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- 嵌入 ----
    embedder: str = "hashing"             # hashing | sentence-transformers
    embedding_dim: int = 256
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- 检索 ----
    top_k: int = 5
    min_score: float = 0.0                # 低于阈值的结果直接丢弃（可审计）

    # ---- 离线与网络 ----
    offline_only: bool = True             # 强制离线：禁止一切出网调用
    allow_network: bool = False           # 仅在显式开启时才允许（如首次下载本地模型）

    # ---- 默认权限标签 ----
    default_acl: str = "public"

    def apply_offline_env(self) -> None:
        """把「离线」落到环境变量层面，阻断第三方库的隐式联网与遥测。"""
        if self.offline_only and not self.allow_network:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get(
                "SENTENCE_TRANSFORMERS_HOME", os.path.abspath("./.models")
            )
            os.environ["LOCALRAG_NO_NETWORK"] = "1"

    def fingerprint(self) -> str:
        """配置指纹：写进审计日志，保证「同一个回答由哪套配置产生」可复现。"""
        d = asdict(self)
        d.pop("db_path")
        d.pop("audit_path")
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def is_offline(self) -> bool:
        return bool(self.offline_only and not self.allow_network)
