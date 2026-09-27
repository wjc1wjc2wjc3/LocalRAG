"""零依赖快速上手：索引 → 检索 → 溯源 → 审计校验

    py examples/quickstart.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localrag.config import Config  # noqa: E402
from localrag.rag import LocalRAG  # noqa: E402

DOCS = {
    "finance_policy.txt": (
        "报销流程：员工需在每月五日前提交发票原件与主管审批单。\n"
        "差旅标准：一线城市住宿每晚不超过五百元，需提前在系统中申请。\n"
    ),
    "hr_policy.txt": (
        "假期政策：入职满一年享有年假十天，需主管审批后方可休假。\n"
        "考勤制度：弹性工作制，核心工作时间为十点至十六点。\n"
    ),
}


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="localrag-demo-")
    paths = []
    for name, body in DOCS.items():
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        paths.append(p)

    cfg = Config(db_path=os.path.join(tmp, "demo.db"),
                 audit_path=os.path.join(tmp, "audit.jsonl"),
                 chunk_size=200, chunk_overlap=40, top_k=3)
    rag = LocalRAG(cfg)

    print("== 索引（含权限标签 team）==")
    for r in rag.index_paths(paths, acl="team"):
        print(f"  {r['status']:>8}  {r.get('chunks', 0)} chunks  {os.path.basename(r['path'])}")

    print("\n== 增量索引（内容未变应跳过）==")
    for r in rag.index_paths(paths, acl="team"):
        print(f"  {r['status']:>8}  {os.path.basename(r['path'])}")

    print("\n== 提问：报销需要提交什么？ ==")
    ans = rag.ask("报销需要提交什么材料？", acls=["team"])
    print(ans.text)
    print("\n引用（可溯源到页码与字符区间）：")
    for i, c in enumerate(ans.citations, 1):
        print(f"  [{i}] {c.location()}  score={c.score:.4f}")

    print("\n== 权限隔离：用 hr 标签访问 finance 文档 ==")
    print(f"  命中数 = {len(rag.retrieve('报销流程', acls=['hr']))}（应为 0）")

    print("\n== 审计链校验 ==")
    ok, msg = rag.audit_verify()
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")

    print("\n== 统计 ==")
    print(" ", rag.stats())
    rag.close()


if __name__ == "__main__":
    main()
