"""LocalRAG 主流程：索引 → 检索 → 生成 → 审计。

三大差异化能力的落地位置：
1. **全离线**：嵌入、向量计算、存储、生成全在本地；`offline_only` 时禁用一切联网。
2. **可审计溯源**：每个答案返回 chunk 级引用（文档 / 页码 / 字符区间 / 分数），
   并写入哈希链审计日志（查询原文只留哈希，不落库）。
3. **增量索引 + 文档级权限**：按内容哈希跳过未变更文档；检索按 acl 标签过滤。
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field

from .audit import AuditLog
from .chunking import chunk_text, file_hash, load_text, page_of
from .config import Config
from .embeddings import cosine, get_embedder, tokenize
from .generator import ExtractiveGenerator
from .store import KnowledgeStore


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    page: int
    start: int
    end: int
    path: str
    title: str
    acl: str

    def location(self) -> str:
        return f"{self.title}#p{self.page}:{self.start}-{self.end}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "page": self.page,
            "span": [self.start, self.end],
            "source": self.path,
            "title": self.title,
            "acl": self.acl,
            "text": self.text,
        }


@dataclass
class Answer:
    query: str
    text: str
    citations: list = field(default_factory=list)
    latency_ms: float = 0.0
    audit_hash: str = ""
    config_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "latency_ms": round(self.latency_ms, 2),
            "audit_hash": self.audit_hash,
            "config_fingerprint": self.config_fingerprint,
        }


def _doc_id(path: str) -> str:
    return hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:32]


class LocalRAG:
    def __init__(self, cfg: Config | None = None, embedder=None, store=None,
                 audit=None, generator=None):
        self.cfg = cfg or Config()
        self.cfg.apply_offline_env()
        self.store = store or KnowledgeStore(self.cfg.db_path)
        self.embedder = embedder or get_embedder(self.cfg)
        self.audit = audit or AuditLog(self.cfg.audit_path)
        self.generator = generator or ExtractiveGenerator()

    # ---------------- 索引 ----------------
    def index_document(self, path: str, acl: str | None = None) -> dict:
        """索引单个文档。内容未变化则跳过（增量索引）。"""
        if not os.path.exists(path):
            return {"path": path, "status": "missing", "chunks": 0}

        doc_id = _doc_id(path)
        acl = acl or self.cfg.default_acl
        src_hash = file_hash(path)
        stored = self.store.stored_hash(doc_id)

        if stored == src_hash:
            return {"path": path, "doc_id": doc_id, "status": "skipped", "chunks": 0}

        doc = load_text(path)
        pieces = chunk_text(doc.text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not pieces:
            return {"path": path, "doc_id": doc_id, "status": "empty", "chunks": 0}

        vecs = self.embedder.embed([p[0] for p in pieces])
        rows = []
        for seq, ((text, start, end), vec) in enumerate(zip(pieces, vecs)):
            rows.append({
                "chunk_id": f"{doc_id}:{seq}",
                "doc_id": doc_id,
                "seq": seq,
                "text": text,
                "start": start,
                "end": end,
                "page": page_of(doc.pages, start),
                "acl": acl,
                "embedder": self.embedder.name,
                "dim": len(vec),
                "vec": vec,
            })

        self.store.upsert_document(
            doc_id=doc_id, path=os.path.abspath(path), title=doc.title,
            source_hash=src_hash, acl=acl,
            meta={"embedder": self.embedder.name, "chunk_size": self.cfg.chunk_size},
        )
        n = self.store.replace_chunks(doc_id, rows)
        status = "updated" if stored else "indexed"
        return {"path": path, "doc_id": doc_id, "status": status, "chunks": n}

    def index_paths(self, paths, acl: str | None = None) -> list:
        out = []
        for p in paths:
            if os.path.isdir(p):
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            out.append(self.index_document(fp, acl))
                        except RuntimeError as exc:
                            out.append({"path": fp, "status": f"error: {exc}", "chunks": 0})
            else:
                try:
                    out.append(self.index_document(p, acl))
                except RuntimeError as exc:
                    out.append({"path": p, "status": f"error: {exc}", "chunks": 0})
        return out

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, acls: list | None = None, top_k: int | None = None):
        """返回带溯源信息的 Citation 列表（按相关度降序）。"""
        k = top_k or self.cfg.top_k
        qv = self.embedder.embed_one(query)
        scored = []
        for row in self.store.iter_chunks(acls=acls):
            s = cosine(qv, row["vec"])
            if s < self.cfg.min_score:
                continue
            scored.append(Citation(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"], score=s,
                text=row["text"], page=row["page"], start=row["start"],
                end=row["end"], path=row["path"], title=row["title"], acl=row["acl"],
            ))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]

    # ---------------- 问答（含审计） ----------------
    def ask(self, query: str, acls: list | None = None, top_k: int | None = None) -> Answer:
        t0 = time.time()
        hits = self.retrieve(query, acls=acls, top_k=top_k)
        text = self.generator.generate(query, hits)
        latency = (time.time() - t0) * 1000.0

        audit_hash = self.audit.append({
            "type": "query",
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_len": len(query),
            "query_terms": len(set(tokenize(query))),
            "acls": acls or [self.cfg.default_acl],
            "top_k": top_k or self.cfg.top_k,
            "hits": [{"chunk_id": c.chunk_id, "score": round(c.score, 6)} for c in hits],
            "latency_ms": round(latency, 2),
            "generator": self.generator.name,
            "embedder": self.embedder.name,
            "config_fingerprint": self.cfg.fingerprint(),
            "offline": self.cfg.is_offline,
        })

        return Answer(
            query=query, text=text, citations=hits,
            latency_ms=latency, audit_hash=audit_hash,
            config_fingerprint=self.cfg.fingerprint(),
        )

    # ---------------- 运维 ----------------
    def stats(self) -> dict:
        s = self.store.stats()
        s.update({
            "embedder": self.embedder.name,
            "audit_entries": len(self.audit),
            "offline": self.cfg.is_offline,
        })
        return s

    def audit_verify(self):
        return self.audit.verify()

    def forget_document(self, path: str) -> None:
        """删除文档及其向量（支持「被遗忘权」，删除后不可检索）。"""
        self.store.delete_document(_doc_id(path))

    def close(self) -> None:
        self.store.close()
