"""LocalRAG —— 隐私优先、全离线、可审计溯源的本地 RAG 知识库。

核心（索引 / 检索 / 存储 / 审计）**零第三方依赖**，仅用 Python 标准库即可运行。
"""
from .audit import AuditLog
from .config import Config
from .generator import ExtractiveGenerator, OpenAICompatGenerator
from .rag import Answer, Citation, LocalRAG
from .store import KnowledgeStore

__version__ = "0.1.0"

__all__ = [
    "LocalRAG",
    "Config",
    "KnowledgeStore",
    "AuditLog",
    "Citation",
    "Answer",
    "ExtractiveGenerator",
    "OpenAICompatGenerator",
]
