"""LocalRAG 核心测试（标准库 unittest，无需 pytest）。

    py -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

from localrag.config import Config
from localrag.eval import evaluate
from localrag.rag import LocalRAG

DOC = ("报销流程：员工需在每月五日前提交发票与审批单。\n" * 6
       + "假期政策：年假十天，需主管审批后方可休假。\n" * 6)


class TestLocalRAG(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(
            db_path=os.path.join(self.tmp.name, "t.db"),
            audit_path=os.path.join(self.tmp.name, "a.jsonl"),
            chunk_size=120, chunk_overlap=20, top_k=3,
        )
        self.rag = LocalRAG(self.cfg)
        self.doc = os.path.join(self.tmp.name, "policy.txt")
        with open(self.doc, "w", encoding="utf-8") as f:
            f.write(DOC)

    def tearDown(self):
        self.rag.close()
        self.tmp.cleanup()

    def test_index_and_cited_query(self):
        r = self.rag.index_document(self.doc, acl="finance")
        self.assertEqual(r["status"], "indexed")
        self.assertGreater(r["chunks"], 0)

        ans = self.rag.ask("报销流程", acls=["finance"])
        self.assertGreater(len(ans.citations), 0)
        self.assertIn("报销", ans.citations[0].text)
        # 溯源信息完整：页码 + 字符区间
        c = ans.citations[0]
        self.assertGreaterEqual(c.page, 1)
        self.assertGreaterEqual(c.end, c.start)

    def test_incremental_index_skips_unchanged(self):
        self.rag.index_document(self.doc)
        again = self.rag.index_document(self.doc)
        self.assertEqual(again["status"], "skipped")

    def test_document_level_acl(self):
        self.rag.index_document(self.doc, acl="finance")
        self.assertEqual(len(self.rag.retrieve("报销", acls=["hr"])), 0)
        self.assertGreater(len(self.rag.retrieve("报销", acls=["finance"])), 0)

    def test_audit_chain_verifies(self):
        self.rag.index_document(self.doc)
        self.rag.ask("报销流程")
        self.rag.ask("假期政策")
        ok, msg = self.rag.audit_verify()
        self.assertTrue(ok, msg)

    def test_forget_removes_vectors(self):
        self.rag.index_document(self.doc)
        self.assertEqual(self.rag.store.stats()["documents"], 1)
        self.rag.forget_document(self.doc)
        self.assertEqual(self.rag.store.stats()["documents"], 0)
        self.assertEqual(self.rag.store.stats()["chunks"], 0)

    def test_offline_by_default(self):
        self.assertTrue(self.cfg.is_offline)


EXTRA_DOCS = {
    "a.txt": "报销流程：员工需在每月五日前提交发票原件与主管审批单。错误码 E1002 表示发票缺失。",
    "b.txt": "假期政策：年假十天，需主管审批后方可休假。错误码 E2007 表示审批超时。",
}


class TestHybridRetrieval(unittest.TestCase):
    """向量 + BM25 混合检索与 MMR 去重。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = []
        for name, body in EXTRA_DOCS.items():
            p = os.path.join(self.tmp.name, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body * 3)  # 重复内容，制造近似 chunk 以验证 MMR
            self.paths.append(p)
        self._n = 0
        self.rags: list = []

    def tearDown(self):
        # Windows 下 SQLite 文件必须先关闭才能被删除
        for rag in self.rags:
            rag.close()
        self.tmp.cleanup()

    def _mk(self, **kw):
        self._n += 1
        cfg = Config(db_path=os.path.join(self.tmp.name, f"t{self._n}.db"),
                     audit_path=os.path.join(self.tmp.name, f"a{self._n}.jsonl"),
                     chunk_size=100, chunk_overlap=10, top_k=5, **kw)
        rag = LocalRAG(cfg)
        self.rags.append(rag)
        return rag

    def test_exact_code_found(self):
        """专有编号这类低频精确词，词法检索应能命中。"""
        rag = self._mk(hybrid=True)
        rag.index_paths(self.paths)
        hits = rag.retrieve("E1002")
        self.assertTrue(hits)
        self.assertIn("E1002", hits[0].text)
        self.assertIn("a.txt", hits[0].path)

    def test_vector_only_mode(self):
        rag = self._mk(hybrid=False)
        rag.index_paths(self.paths)
        hits = rag.retrieve("报销")
        self.assertTrue(hits)
        self.assertIn("报销", hits[0].text)

    def test_weighted_mode(self):
        rag = self._mk(hybrid=True, hybrid_mode="weighted")
        rag.index_paths(self.paths)
        hits = rag.retrieve("年假")
        self.assertTrue(hits)
        self.assertIn("b.txt", hits[0].path)

    def test_mmr_avoids_identical_chunks(self):
        rag = self._mk(mmr_lambda=0.9)
        rag.index_paths(self.paths)
        hits = rag.retrieve("报销流程", top_k=2)
        if len(hits) >= 2:
            self.assertNotEqual(hits[0].text, hits[1].text)


class TestTagsFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=os.path.join(self.tmp.name, "t.db"),
                          audit_path=os.path.join(self.tmp.name, "a.jsonl"),
                          chunk_size=200, chunk_overlap=20, top_k=5)
        self.rag = LocalRAG(self.cfg)
        self.a = os.path.join(self.tmp.name, "a.txt")
        self.b = os.path.join(self.tmp.name, "b.txt")
        with open(self.a, "w", encoding="utf-8") as f:
            f.write(EXTRA_DOCS["a.txt"])
        with open(self.b, "w", encoding="utf-8") as f:
            f.write(EXTRA_DOCS["b.txt"])

    def tearDown(self):
        self.rag.close()
        self.tmp.cleanup()

    def test_tag_filter_narrows_results(self):
        self.rag.index_document(self.a, acl="team", tags=["finance"])
        self.rag.index_document(self.b, acl="team", tags=["hr"])
        hits = self.rag.retrieve("报销", acls=["team"], tags=["finance"])
        self.assertTrue(hits)
        self.assertTrue(all("finance" in c.tags for c in hits))

    def test_docs_listing_includes_tags(self):
        self.rag.index_document(self.a, acl="team", tags=["finance"])
        docs = self.rag.documents()
        self.assertEqual(len(docs), 1)
        self.assertIn("finance", docs[0]["tags"])


class TestMarkdownAndHTML(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=os.path.join(self.tmp.name, "t.db"),
                          audit_path=os.path.join(self.tmp.name, "a.jsonl"),
                          chunk_size=120, chunk_overlap=20, top_k=5)
        self.rag = LocalRAG(self.cfg)

    def tearDown(self):
        self.rag.close()
        self.tmp.cleanup()

    def test_markdown_section_recorded(self):
        p = os.path.join(self.tmp.name, "guide.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("# 总纲\n\n第一段说明文字。\n\n## 报销细则\n\n报销需提交发票与审批单。\n")
        self.rag.index_document(p)
        hits = self.rag.retrieve("报销需提交发票")
        self.assertTrue(hits)
        self.assertTrue(hits[0].section)  # 章节路径非空

    def test_html_loaded_without_deps(self):
        p = os.path.join(self.tmp.name, "page.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write("<html><head><title>手册</title></head>"
                    "<body><p>报销需要发票原件</p></body></html>")
        r = self.rag.index_document(p)
        self.assertEqual(r["status"], "indexed")
        hits = self.rag.retrieve("报销需要发票原件")
        self.assertTrue(hits)


class TestOpsAndEval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=os.path.join(self.tmp.name, "t.db"),
                          audit_path=os.path.join(self.tmp.name, "a.jsonl"),
                          chunk_size=200, chunk_overlap=20, top_k=3)
        self.rag = LocalRAG(self.cfg)
        self.a = os.path.join(self.tmp.name, "a.txt")
        with open(self.a, "w", encoding="utf-8") as f:
            f.write(EXTRA_DOCS["a.txt"])
        self.rag.index_document(self.a, acl="team", tags=["finance"])

    def tearDown(self):
        self.rag.close()
        self.tmp.cleanup()

    def test_audit_stats(self):
        self.rag.ask("报销流程", acls=["team"])
        self.rag.ask("发票缺失", acls=["team"])
        st = self.rag.audit_stats()
        self.assertEqual(st["queries"], 2)
        self.assertIn("top_documents", st)
        self.assertIn("team", st["acl_distribution"])
        self.assertGreater(st["total_hits"], 0)

    def test_evaluate_recall(self):
        qrels = [{"query": "报销流程", "relevant": ["a.txt"]},
                 {"query": "完全不相关的查询词", "relevant": ["不存在.txt"]}]
        res = evaluate(self.rag, qrels, top_k=3)
        self.assertEqual(res["cases"], 2)
        self.assertGreaterEqual(res["recall@3"], 0.5)

    def test_answer_to_markdown(self):
        ans = self.rag.ask("报销流程")
        md = ans.to_markdown()
        self.assertIn("## 引用", md)
        self.assertIn("a.txt", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
