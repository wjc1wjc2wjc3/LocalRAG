"""知识库存储：**单文件 SQLite**，无外部数据库、无服务进程。

设计要点（对标现有项目缺失的能力）：
- chunk 级保存**字符区间 + 页码 + 所属章节**，使每个答案都能溯源到原文位置；
- 文档带 `acl` 权限标签，**并支持 `tags` 业务标签过滤**（元数据检索，很多轻量
  RAG 项目只能全库搜）；
- `source_hash` 支撑**增量索引**：内容未变的文档直接跳过；
- 表结构带轻量迁移，老库打开时自动补齐新增列，不需要重建索引。
"""
from __future__ import annotations

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    title       TEXT,
    source_hash TEXT NOT NULL,
    acl         TEXT NOT NULL DEFAULT 'public',
    tags        TEXT NOT NULL DEFAULT '[]',
    meta        TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    start     INTEGER NOT NULL,
    end       INTEGER NOT NULL,
    page      INTEGER NOT NULL DEFAULT 1,
    section   TEXT NOT NULL DEFAULT '',
    acl       TEXT NOT NULL DEFAULT 'public',
    tags      TEXT NOT NULL DEFAULT '[]',
    embedder  TEXT NOT NULL DEFAULT 'hashing',
    dim       INTEGER NOT NULL DEFAULT 0,
    vec       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_acl ON chunks(acl);
"""


def _now() -> float:
    return time.time()


class KnowledgeStore:
    def __init__(self, path: str = "localrag.db"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    # ---------------- 轻量迁移 ----------------
    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        cols = [r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")]
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _migrate(self) -> None:
        self._ensure_column("documents", "tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("chunks", "tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("chunks", "section", "TEXT NOT NULL DEFAULT ''")

    # ---------------- 文档 ----------------
    def upsert_document(self, doc_id: str, path: str, title: str, source_hash: str,
                        acl: str = "public", meta: dict | None = None,
                        tags: list | None = None) -> bool:
        """写入/更新文档元数据。返回 True 表示新建。"""
        cur = self.conn.execute("SELECT source_hash FROM documents WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        now = _now()
        tags_json = json.dumps(sorted(set(tags or [])), ensure_ascii=False)
        if row is None:
            self.conn.execute(
                "INSERT INTO documents(doc_id,path,title,source_hash,acl,tags,meta,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (doc_id, path, title, source_hash, acl, tags_json,
                 json.dumps(meta or {}), now, now),
            )
            self.conn.commit()
            return True
        self.conn.execute(
            "UPDATE documents SET path=?,title=?,acl=?,tags=?,meta=?,updated_at=? WHERE doc_id=?",
            (path, title, acl, tags_json, json.dumps(meta or {}), now, doc_id),
        )
        self.conn.commit()
        return False

    def get_document(self, doc_id: str):
        cur = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def stored_hash(self, doc_id: str) -> str | None:
        row = self.get_document(doc_id)
        return row["source_hash"] if row else None

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self.conn.commit()

    def list_documents(self) -> list:
        cur = self.conn.execute(
            "SELECT doc_id,path,title,source_hash,acl,tags,updated_at FROM documents"
            " ORDER BY updated_at DESC")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            try:
                d["tags"] = json.loads(d.get("tags") or "[]")
            except json.JSONDecodeError:
                d["tags"] = []
            out.append(d)
        return out

    # ---------------- chunk ----------------
    def replace_chunks(self, doc_id: str, rows: list) -> int:
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.executemany(
            "INSERT INTO chunks(chunk_id,doc_id,seq,text,start,end,page,section,acl,tags,embedder,dim,vec)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r["chunk_id"], doc_id, r["seq"], r["text"], r["start"], r["end"], r["page"],
              r.get("section", ""), r["acl"],
              json.dumps(sorted(set(r.get("tags") or [])), ensure_ascii=False),
              r["embedder"], r["dim"], json.dumps(r["vec"])) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    def iter_chunks(self, acl: str | None = None, acls: list | None = None,
                    tags: list | None = None):
        """迭代 chunk，可按权限标签与业务标签过滤。

        tags 为「任一命中」语义：chunk 的标签与给定集合有交集即通过。
        """
        cur = self.conn.execute(
            "SELECT c.*, d.path, d.title FROM chunks c JOIN documents d ON d.doc_id=c.doc_id"
        )
        want = set(tags or [])
        for r in cur:
            row = dict(r)
            if acls is not None and row["acl"] not in acls:
                continue
            if acl is not None and row["acl"] != acl:
                continue
            if want:
                try:
                    ctags = set(json.loads(row.get("tags") or "[]"))
                except json.JSONDecodeError:
                    ctags = set()
                if not (ctags & want):
                    continue
            row["vec"] = json.loads(row["vec"])
            try:
                row["tags"] = json.loads(row.get("tags") or "[]")
            except json.JSONDecodeError:
                row["tags"] = []
            yield row

    # ---------------- 统计 ----------------
    def stats(self) -> dict:
        d = self.conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        c = self.conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        return {"documents": d, "chunks": c, "db_path": self.path}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
