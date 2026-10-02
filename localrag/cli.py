"""LocalRAG 命令行入口（标准库 argparse，零依赖）。

    py -m localrag.cli index ./docs --acl team --tags finance
    py -m localrag.cli query "报销流程是什么" --acl team
    py -m localrag.cli docs                      # 列出已索引文档
    py -m localrag.cli stats
    py -m localrag.cli audit-verify              # 校验审计链
    py -m localrag.cli audit-stats               # 审计统计（热点文档/权限分布）
    py -m localrag.cli eval --qrels qrels.json   # 检索效果评测 recall@k / MRR
    py -m localrag.cli forget ./docs/secret.txt
    py -m localrag.cli serve --port 8070
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .eval import evaluate, load_qrels
from .rag import LocalRAG


def _split_list(raw: str) -> list:
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
    p.add_argument("--retrieval", default="hybrid", choices=["hybrid", "vector"],
                   help="hybrid=向量+BM25（默认），vector=仅向量")
    p.add_argument("--hybrid-mode", default="rrf", choices=["rrf", "weighted"])
    p.add_argument("--mmr-lambda", type=float, default=0.3,
                   help="MMR 去重强度，0=关闭（默认 0.3）")
    p.add_argument("--allow-network", action="store_true",
                   help="显式允许联网（默认全离线；仅首次拉取本地模型时需要）")
    p.add_argument("--disable-structure", action="store_true",
                   help="关闭结构感知检索（默认开启，按章节树聚合得分）")
    p.add_argument("--reasoning", action="store_true",
                   help="开启 LLM 推理式章节导航（需本地 OpenAI 兼容模型，可选外挂）")
    p.add_argument("--reasoning-endpoint", default=None,
                   help="推理导航端点（默认 http://127.0.0.1:8080/v1）")
    p.add_argument("--history-turns", type=int, default=None,
                   help="上下文感知：并入查询的历史轮数（默认 2）")
    p.add_argument("--llm-base-url", default=None,
                   help="启用 LLM 生成：任意 OpenAI 兼容端点（本地如 http://127.0.0.1:8080/v1，"
                        "或云端如 https://api.openai.com/v1）。留空=默认抽取式（全离线）")
    p.add_argument("--llm-api-key", default=None,
                   help="云端 LLM 厂商所需的 Key（本地模型可留空）")
    p.add_argument("--llm-model", default="auto", help="LLM 模型名（默认 auto）")

    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="索引文档/目录")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--acl", default="public", help="权限标签，逗号分隔，如 team,finance")
    pi.add_argument("--tags", default="", help="业务标签，逗号分隔，如 财务,2026")

    pq = sub.add_parser("query", help="检索并作答（带引用与审计）")
    pq.add_argument("query")
    pq.add_argument("--acl", default="public")
    pq.add_argument("--tags", default="", help="只检索带这些标签的文档")
    pq.add_argument("--top-k", type=int, default=None)
    pq.add_argument("--history", default="",
                    help="历史对话 JSON 文件路径（列表，每项为 {role,content} 或 [问,答]）")
    pq.add_argument("--domain-terms", default="", help="额外领域词，逗号分隔，并入查询")
    pq.add_argument("--reasoning", action="store_true",
                    help="本次查询启用 LLM 推理式章节导航")
    pq.add_argument("--json", action="store_true", help="以 JSON 输出")
    pq.add_argument("--md", action="store_true", help="以 Markdown 输出（含引用清单）")
    pq.add_argument("--out", default="", help="把结果写入文件")

    sub.add_parser("docs", help="列出已索引文档")

    sub.add_parser("stats", help="查看库统计")
    sub.add_parser("audit-verify", help="校验审计链完整性")
    sub.add_parser("audit-stats", help="审计统计（查询量/热点文档/权限分布）")

    pe = sub.add_parser("eval", help="检索效果评测（recall@k / MRR）")
    pe.add_argument("--qrels", required=True, help="评测集 JSON 文件路径")
    pe.add_argument("--top-k", type=int, default=5)
    pe.add_argument("--json", action="store_true")

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
        hybrid=(args.retrieval == "hybrid"), hybrid_mode=args.hybrid_mode,
        mmr_lambda=args.mmr_lambda,
        allow_network=args.allow_network,
        structure_aware=not args.disable_structure,
        reasoning_rerank=args.reasoning,
    )
    if args.reasoning_endpoint:
        cfg.reasoning_endpoint = args.reasoning_endpoint
    if args.history_turns is not None:
        cfg.history_turns = args.history_turns
    if args.llm_base_url:
        cfg.llm_base_url = args.llm_base_url
    if args.llm_api_key:
        cfg.llm_api_key = args.llm_api_key
    if args.llm_model:
        cfg.llm_model = args.llm_model
    rag = LocalRAG(cfg)

    try:
        if args.cmd == "index":
            res = rag.index_paths(args.paths, acl=args.acl,
                                  tags=_split_list(args.tags))
            for r in res:
                print(f"{r['status']:>8}  {r.get('chunks', 0):>4} chunks  {r['path']}")
            total = sum(r.get("chunks", 0) for r in res)
            print(f"\n共写入 {total} 个 chunk（skipped 表示内容未变，已跳过）")

        elif args.cmd == "query":
            history = None
            if args.history:
                with open(args.history, "r", encoding="utf-8") as f:
                    history = json.load(f)
            domain_terms = _split_list(args.domain_terms) if args.domain_terms else None
            reasoning = args.reasoning if args.reasoning else None
            ans = rag.ask(args.query, acls=_split_list(args.acl),
                          top_k=args.top_k, tags=_split_list(args.tags),
                          history=history, domain_terms=domain_terms, reasoning=reasoning)
            if args.json:
                out = json.dumps(ans.to_dict(), ensure_ascii=False, indent=2)
            elif args.md:
                out = ans.to_markdown()
            else:
                refs = "\n".join(
                    f"  [{i}] {c.location()}  score={c.score:.4f}  acl={c.acl}"
                    for i, c in enumerate(ans.citations, 1))
                tr = ans.trace
                secs = ""
                if tr.get("top_sections"):
                    secs = "；".join(s["path"] for s in tr["top_sections"][:3])
                    secs = f"\n命中章节：{secs}"
                out = (f"{ans.text}\n\n引用：\n{refs}\n"
                       f"{secs}\n\n审计记录 {ans.audit_hash[:16]}…  耗时 {ans.latency_ms:.1f}ms")
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(out)
                print(f"已写入 {args.out}")
            else:
                print(out)

        elif args.cmd == "docs":
            for d in rag.documents():
                tags = ",".join(d.get("tags") or []) or "-"
                print(f"{d['title']:<40} acl={d['acl']:<12} tags={tags}")

        elif args.cmd == "stats":
            print(json.dumps(rag.stats(), ensure_ascii=False, indent=2))

        elif args.cmd == "audit-verify":
            ok, msg = rag.audit_verify()
            print(("PASS  " if ok else "FAIL  ") + msg)
            return 0 if ok else 1

        elif args.cmd == "audit-stats":
            print(json.dumps(rag.audit_stats(), ensure_ascii=False, indent=2))

        elif args.cmd == "eval":
            qrels = load_qrels(args.qrels)
            r = evaluate(rag, qrels, top_k=args.top_k)
            if args.json:
                print(json.dumps(r, ensure_ascii=False, indent=2))
            else:
                key = f"recall@{r['top_k']}"
                print(f"用例数 {r['cases']}  top_k={r['top_k']}")
                print(f"  {key}: {r[key]}")
                print(f"  MRR: {r['mrr']}")
                for case in r["details"]:
                    rank = case["first_rank"] if case["first_rank"] else "未命中"
                    print(f"    - {case['query']}  → 首次命中位置 {rank}")

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
