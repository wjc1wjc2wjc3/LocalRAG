"""LocalRAG 主流程：索引 → 检索 → 生成 → 审计。

差异化能力的落地位置：
1. **全离线**：嵌入、向量计算、存储、生成全在本地；`offline_only` 时禁用一切联网。
2. **可审计溯源**：每个答案返回 chunk 级引用（文档 / 页码 / 章节 / 字符区间 / 分数），
   并写入哈希链审计日志（查询原文只留哈希，不落库）。
3. **增量索引 + 文档级权限**：按内容哈希跳过未变更文档；检索按 acl 标签过滤。
4. **混合检索**：向量 + BM25 词法（RRF 融合）+ MMR 去重，纯标准库实现。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from .audit import AuditLog
from .bm25 import BM25, rrf_fuse
from .chunking import chunk_document, file_hash, is_markdown, load_text, page_of
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
    section: str = ""
    tags: list = field(default_factory=list)

    def location(self) -> str:
        base = f"{self.title}#p{self.page}:{self.start}-{self.end}"
        return f"{base} ({self.section})" if self.section else base

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "page": self.page,
            "span": [self.start, self.end],
            "section": self.section,
            "tags": self.tags,
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

    def to_markdown(self) -> str:
        """导出为带引用编号的 Markdown（便于存档与二次分发）。"""
        lines = [f"## 问题\n\n{self.query}\n", "## 回答\n", self.text, "", "## 引用"]
        for i, c in enumerate(self.citations, start=1):
            lines.append(f"{i}. `{c.location()}`  score={c.score:.4f}  acl={c.acl}")
        return "\n".join(lines)


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
    def index_document(self, path: str, acl: str | None = None,
                       tags: list | None = None) -> dict:
        """索引单个文档。内容未变化则跳过（增量索引）。"""
        if not os.path.exists(path):
            return {"path": path, "status": "missing", "chunks": 0}

        doc_id = _doc_id(path)
        acl = acl or self.cfg.default_acl
        tags = list(tags or [])
        src_hash = file_hash(path)
        stored = self.store.stored_hash(doc_id)

        if stored == src_hash:
            return {"path": path, "doc_id": doc_id, "status": "skipped", "chunks": 0}

        doc = load_text(path)
        pieces = chunk_document(doc.text, self.cfg.chunk_size,
                                self.cfg.chunk_overlap,
                                markdown=is_markdown(path))
        if not pieces:
            return {"path": path, "doc_id": doc_id, "status": "empty", "chunks": 0}

        vecs = self.embedder.embed([p[0] for p in pieces])
        rows = []
        for seq, ((text, start, end, section), vec) in enumerate(zip(pieces, vecs)):
            rows.append({
                "chunk_id": f"{doc_id}:{seq}",
                "seq": seq,
                "text": text,
                "start": start,
                "end": end,
                "page": page_of(doc.pages, start),
                "section": section,
                "acl": acl,
                "tags": tags,
                "embedder": self.embedder.name,
                "dim": len(vec),
                "vec": vec,
            })

        self.store.upsert_document(
            doc_id=doc_id, path=os.path.abspath(path), title=doc.title,
            source_hash=src_hash, acl=acl, tags=tags,
            meta={"embedder": self.embedder.name, "chunk_size": self.cfg.chunk_size,
                  "markdown": is_markdown(path)},
        )
        n = self.store.replace_chunks(doc_id, rows)
        status = "updated" if stored else "indexed"
        return {"path": path, "doc_id": doc_id, "status": status, "chunks": n}

    def index_paths(self, paths, acl: str | None = None,
                    tags: list | None = None) -> list:
        out = []
        for p in paths:
            if os.path.isdir(p):
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            out.append(self.index_document(fp, acl, tags))
                        except RuntimeError as exc:
                            out.append({"path": fp, "status": f"error: {exc}", "chunks": 0})
            else:
                try:
                    out.append(self.index_document(p, acl, tags))
                except RuntimeError as exc:
                    out.append({"path": p, "status": f"error: {exc}", "chunks": 0})
        return out

    # ---------------- 检索 ----------------
    def _mmr(self, ranking, score_by_idx, rows, k: int) -> list:
        """最大边际相关：在相关度与多样性之间取平衡，避免返回一堆近乎重复的 chunk。"""
        lam = self.cfg.mmr_lambda
        vals = [score_by_idx.get(i, 0.0) for i in ranking]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        rel = {i: (score_by_idx.get(i, 0.0) - lo) / rng for i in ranking}

        selected: list = []
        remaining = list(ranking)
        limit = min(k, len(ranking))
        while remaining and len(selected) < limit:
            best, best_val = None, None
            for i in remaining:
                redundancy = max(
                    (cosine(rows[i]["vec"], rows[j]["vec"]) for j in selected),
                    default=0.0)
                val = rel[i] - lam * redundancy
                if best_val is None or val > best_val:
                    best, best_val = i, val
            selected.append(best)
            remaining.remove(best)
        return selected

    def retrieve(self, query: str, acls: list | None = None,
                 top_k: int | None = None, tags: list | None = None):
        """检索：向量 + BM25 混合（RRF/加权），可选 MMR 去重。

        返回带溯源信息的 Citation 列表（按最终相关度降序）。
        """
        k = top_k or self.cfg.top_k
        rows = list(self.store.iter_chunks(acls=acls, tags=tags))
        if not rows:
            return []

        qv = self.embedder.embed_one(query)
        vec_scores = [cosine(qv, r["vec"]) for r in rows]
        pool_n = min(len(rows), max(self.cfg.candidate_pool, k * 4))
        order_vec = sorted(range(len(rows)), key=lambda i: vec_scores[i],
                           reverse=True)[:pool_n]

        if not self.cfg.hybrid:
            ranking = order_vec
            score_by_idx = {i: vec_scores[i] for i in order_vec}
        else:
            lex = BM25().build([r["text"] for r in rows]).scores(query)
            order_lex = sorted(range(len(rows)), key=lambda i: lex[i],
                               reverse=True)[:pool_n]
            if self.cfg.hybrid_mode == "weighted":
                ids = list(dict.fromkeys(list(order_vec) + list(order_lex)))

                def _norm(values, idxs):
                    picked = [values[i] for i in idxs]
                    lo, hi = min(picked), max(picked)
                    rng = (hi - lo) or 1.0
                    return {i: (values[i] - lo) / rng for i in idxs}

                nv, nl = _norm(vec_scores, ids), _norm(lex, ids)
                w = self.cfg.bm25_weight
                score_by_idx = {i: (1 - w) * nv[i] + w * nl[i] for i in ids}
            else:  # rrf
                fused = rrf_fuse([[rows[i]["chunk_id"] for i in order_vec],
                                  [rows[i]["chunk_id"] for i in order_lex]])
                id_to_idx = {rows[i]["chunk_id"]: i for i in range(len(rows))}
                pairs = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
                score_by_idx = {id_to_idx[cid]: s for cid, s in pairs}
            ranking = sorted(score_by_idx, key=lambda i: score_by_idx[i], reverse=True)

        # 低分过滤（混合时用融合分，纯向量时用余弦分）
        if self.cfg.min_score > 0:
            ranking = [i for i in ranking if score_by_idx.get(i, 0.0) >= self.cfg.min_score]
        # 结果去重
        if self.cfg.mmr_lambda > 0 and len(ranking) > 1:
            ranking = self._mmr(ranking, score_by_idx, rows, k)

        return [Citation(
            chunk_id=rows[i]["chunk_id"], doc_id=rows[i]["doc_id"],
            score=score_by_idx.get(i, vec_scores[i]), text=rows[i]["text"],
            page=rows[i]["page"], start=rows[i]["start"], end=rows[i]["end"],
            path=rows[i]["path"], title=rows[i]["title"], acl=rows[i]["acl"],
            section=rows[i].get("section", ""), tags=rows[i].get("tags", []),
        ) for i in ranking[:k]]

    # ---------------- 问答（含审计） ----------------
    def ask(self, query: str, acls: list | None = None, top_k: int | None = None,
            tags: list | None = None) -> Answer:
        t0 = time.time()
        hits = self.retrieve(query, acls=acls, top_k=top_k, tags=tags)
        text = self.generator.generate(query, hits)
        latency = (time.time() - t0) * 1000.0

        audit_hash = self.audit.append({
            "type": "query",
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_len": len(query),
            "query_terms": len(set(tokenize(query))),
            "acls": acls or [self.cfg.default_acl],
            "tags": tags or [],
            "top_k": top_k or self.cfg.top_k,
            "hits": [{"chunk_id": c.chunk_id, "score": round(c.score, 6)} for c in hits],
            "latency_ms": round(latency, 2),
            "generator": self.generator.name,
            "embedder": self.embedder.name,
            "retrieval": ("hybrid:" + self.cfg.hybrid_mode) if self.cfg.hybrid else "vector",
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
            "hybrid": self.cfg.hybrid,
            "hybrid_mode": self.cfg.hybrid_mode if self.cfg.hybrid else "off",
        })
        return s

    def documents(self) -> list:
        """列出已索引文档（含 acl 与 tags）。"""
        return self.store.list_documents()

    def audit_stats(self) -> dict:
        """审计日志统计：查询量、热点文档、权限分布、平均延迟。"""
        path = self.cfg.audit_path
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {"queries": 0}
        queries = 0
        total_hits = 0
        latencies: list = []
        hit_docs: dict = {}
        acls: dict = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line).get("event", {})
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "query":
                    continue
                queries += 1
                latencies.append(float(ev.get("latency_ms", 0.0)))
                for h in ev.get("hits", []):
                    total_hits += 1
                    doc = str(h.get("chunk_id", "")).split(":")[0]
                    hit_docs[doc] = hit_docs.get(doc, 0) + 1
                for a in ev.get("acls", []):
                    acls[a] = acls.get(a, 0) + 1

        titles = {d["doc_id"]: d["title"] for d in self.store.list_documents()}
        top = sorted(hit_docs.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {
            "queries": queries,
            "total_hits": total_hits,
            "avg_hits_per_query": round(total_hits / queries, 2) if queries else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "top_documents": [{"title": titles.get(d, d), "doc_id": d, "hits": n}
                              for d, n in top],
            "acl_distribution": acls,
        }

    def audit_verify(self):
        return self.audit.verify()

    def forget_document(self, path: str) -> None:
        """删除文档及其向量（支持「被遗忘权」，删除后不可检索）。"""
        self.store.delete_document(_doc_id(path))

    def close(self) -> None:
        self.store.close()
