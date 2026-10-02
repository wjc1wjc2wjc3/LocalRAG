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
    hybrid: bool = True                   # 混合检索：向量 + BM25 词法
    hybrid_mode: str = "rrf"              # rrf（推荐）| weighted
    bm25_weight: float = 0.5              # weighted 模式下的词法权重
    mmr_lambda: float = 0.3               # MMR 去重强度，0=关闭
    candidate_pool: int = 50              # 融合前的候选池大小

    # ---- 结构感知与推理检索（受 PageIndex 启发） ----
    structure_aware: bool = True          # 按章节树聚合得分，优先返回高相关章节
    structure_top_sections: int = 3       # 结构感知时保留的 top 章节数
    history_turns: int = 2                # 上下文感知：并入查询的近期历史轮数
    reasoning_rerank: bool = False        # LLM 推理式章节导航（需本地模型，可选外挂）
    reasoning_endpoint: str = "http://127.0.0.1:8080/v1"  # 本地 OpenAI 兼容端点
    reasoning_model: str = "auto"
    reasoning_api_key: str = ""           # 云端推理端点所需的 Key（本地可留空）
    reasoning_candidates: int = 10        # 提交给 LLM 裁决的候选章节数

    # ---- 生成器：离线 / 在线 LLM 由用户选择 ----
    # llm_base_url 为空 = 默认抽取式（零 LLM、全离线）；
    # 填入任意 OpenAI 兼容端点 = 启用 LLM 生成（本地服务或云端厂商均可）。
    llm_base_url: str = ""                # 例如 http://127.0.0.1:8080/v1 或 https://api.openai.com/v1
    llm_api_key: str = ""                 # 云端厂商（OpenAI / 兼容服务）所需；本地模型可留空
    llm_model: str = "auto"
    llm_timeout: float = 60.0

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
        """配置指纹：写进审计日志，保证「同一个回答由哪套配置产生」可复现。

        主动剔除密钥类字段，避免任何 secret 进入审计（仅保留其「是否启用」的布尔，
        不泄漏明文）。
        """
        d = asdict(self)
        d.pop("db_path")
        d.pop("audit_path")
        for secret in ("llm_api_key", "reasoning_api_key"):
            d.pop(secret, None)
        d["llm_configured"] = bool(self.llm_base_url)
        d["reasoning_configured"] = bool(self.reasoning_endpoint and self.reasoning_rerank)
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def is_offline(self) -> bool:
        return bool(self.offline_only and not self.allow_network)
