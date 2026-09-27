"""LocalRAG 命令行入口（标准库 argparse，零依赖）。

    py -m localrag.cli index ./docs --acl team
    py -m localrag.cli query "报销流程是什么？" --acl team
    py -m localrag.cli stats
    py -m localrag.cli audit-verify
    py -m localrag.cli forget ./docs/secret.txt
    py -m localrag.cli serve --port 8070
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .rag import LocalRAG


def _split_acls(raw: str) -> list:
    return [a.strip() for a in raw.split(",") if a.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="localrag",
        description="LocalRAG：全离线、可审计溯源的本地 RAG 知识库",
    )
    p.add_argument("--db", default="localrag.db", help="SQLite 库路径（默认 localrag.db）")
    p.add_argument("--audit", default="audit.jsonl", help="审计日志路径")
    p.add_argument("--embedder", default="hashing",
                   choices=["hashing", "sentence-transformers"], help="嵌入后端")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=120)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--allow-network", action="store_true",
                   help="显式允许联网（默认全离线；仅首次拉取本地模型时需要）")

    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="索引文档/目录")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--acl", default="public", help="权限标签，逗号分隔，如 team,finance")

    pq = sub.add_parser("query", help="检索并作答（带引用与审计）")
    pq.add_argument("query")
    pq.add_argument("--acl", default="public")
    pq.add_argument("--top-k", type=int, default=None)
    pq.add_argument("--json", action="store_true", help="以 JSON 输出")

    sub.add_parser("stats", help="查看库统计")
    sub.add_parser("audit-verify", help="校验审计链完整性")

    pf = sub.add_parser("forget", help="删除文档及其向量（被遗忘权）")
    pf.add_argument("path")

    ps = sub.add_parser("serve", help="启动 HTTP API（需可选依赖 fastapi+uvicorn）")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8070)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        db_path=args.db, audit_path=args.audit, embedder=args.embedder,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        top_k=args.top_k, min_score=args.min_score,
        allow_network=args.allow_network,
    )
    rag = LocalRAG(cfg)

    try:
        if args.cmd == "index":
            res = rag.index_paths(args.paths, acl=args.acl)
            for r in res:
                print(f"{r['status']:>8}  {r.get('chunks', 0):>4} chunks  {r['path']}")
            total = sum(r.get("chunks", 0) for r in res)
            print(f"\n共写入 {total} 个 chunk（skipped 表示内容未变，已跳过）")

        elif args.cmd == "query":
            ans = rag.ask(args.query, acls=_split_acls(args.acl), top_k=args.top_k)
            if args.json:
                print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(ans.text)
                print("\n引用：")
                for i, c in enumerate(ans.citations, 1):
                    print(f"  [{i}] {c.location()}  score={c.score:.4f}  acl={c.acl}")
                print(f"\n审计记录 {ans.audit_hash[:16]}…  耗时 {ans.latency_ms:.1f}ms")

        elif args.cmd == "stats":
            print(json.dumps(rag.stats(), ensure_ascii=False, indent=2))

        elif args.cmd == "audit-verify":
            ok, msg = rag.audit_verify()
            print(("PASS  " if ok else "FAIL  ") + msg)
            return 0 if ok else 1

        elif args.cmd == "forget":
            rag.forget_document(args.path)
            print(f"已删除：{args.path}")

        elif args.cmd == "serve":
            from .server import run  # 需要可选依赖
            run(host=args.host, port=args.port, cfg=cfg)
            return 0
    finally:
        rag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
