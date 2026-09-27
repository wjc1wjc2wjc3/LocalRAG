"""LocalRAG 核心测试（标准库 unittest，无需 pytest）。

    py -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

from localrag.config import Config
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
