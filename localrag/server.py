"""可选的 HTTP API（需 fastapi + uvicorn，属**可选依赖**）。

保持核心零依赖：未安装时给出明确安装提示，而不是让整包 import 失败。
接口刻意做得「可审计」：每次 /query 都返回 citations 与 audit_hash。

    py -m localrag.cli serve --port 8070
"""
from __future__ import annotations

import os
from typing import Optional

from .config import Config


def create_app(cfg: Config | None = None):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HTTP API 需要可选依赖：pip install fastapi uvicorn"
        ) from exc

    cfg = cfg or Config()
    # 允许通过环境变量注入在线/本地 LLM（不写明文进代码或配置）：
    #   LOCALRAG_LLM_BASE_URL / LOCALRAG_LLM_API_KEY / LOCALRAG_LLM_MODEL
    if os.environ.get("LOCALRAG_LLM_BASE_URL"):
        cfg.llm_base_url = os.environ["LOCALRAG_LLM_BASE_URL"]
    if os.environ.get("LOCALRAG_LLM_API_KEY"):
        cfg.llm_api_key = os.environ["LOCALRAG_LLM_API_KEY"]
    if os.environ.get("LOCALRAG_LLM_MODEL"):
        cfg.llm_model = os.environ["LOCALRAG_LLM_MODEL"]
    from .rag import LocalRAG

    rag = LocalRAG(cfg)
    app = FastAPI(title="LocalRAG", version="0.1.0",
                  description="全离线、可审计溯源的本地 RAG 知识库")

    class IndexReq(BaseModel):
        paths: list
        acl: str = "public"

    class QueryReq(BaseModel):
        query: str
        acl: str = "public"
        top_k: Optional[int] = None
        history: list = []
        domain_terms: list = []
        reasoning: Optional[bool] = None

    @app.post("/index")
    def index(req: IndexReq):
        return {"results": rag.index_paths(req.paths, acl=req.acl)}

    @app.post("/query")
    def query(req: QueryReq):
        acls = [a.strip() for a in req.acl.split(",") if a.strip()]
        ans = rag.ask(req.query, acls=acls, top_k=req.top_k,
                      history=req.history or None,
                      domain_terms=req.domain_terms or None,
                      reasoning=req.reasoning)
        return ans.to_dict()

    @app.get("/stats")
    def stats():
        return rag.stats()

    @app.get("/audit/verify")
    def audit_verify():
        ok, msg = rag.audit_verify()
        return {"ok": ok, "message": msg}

    @app.get("/health")
    def health():
        return {"status": "ok", "offline": cfg.is_offline}

    return app


def run(host: str = "127.0.0.1", port: int = 8070, cfg: Config | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("启动服务需要可选依赖：pip install fastapi uvicorn") from exc
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
