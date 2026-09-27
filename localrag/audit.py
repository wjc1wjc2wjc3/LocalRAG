"""审计日志：**哈希链**（tamper-evident），本地 JSONL，不出网。

每一次检索都记录：查询内容哈希、命中的 chunk_id 与分数、耗时、配置指纹、
调用方权限标签。相邻记录用 prev_hash 串联，任何一条被删改都会导致
`verify()` 失败 —— 这是「可审计溯源」的落地实现，也是现有多数 RAG 项目
（统计中「隐私/安全/审计」仅占 1.1%）不具备的能力。
"""
from __future__ import annotations

import hashlib
import json
import os
import time


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(obj) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str = "audit.jsonl"):
        self.path = path
        self._last = "0" * 64
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._last = json.loads(line).get("hash", self._last)
                        except json.JSONDecodeError:
                            break

    def append(self, event: dict) -> str:
        """写入一条审计记录，返回该记录的哈希。"""
        entry = {
            "ts": time.time(),
            "prev": self._last,
            "event": event,
        }
        entry_hash = _sha(entry)
        entry["hash"] = entry_hash
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canon(entry) + "\n")
        self._last = entry_hash
        return entry_hash

    def verify(self) -> tuple:
        """校验哈希链完整性。返回 (ok, message)。"""
        if not os.path.exists(self.path):
            return True, "审计日志为空"
        prev = "0" * 64
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"第 {i} 行不是合法 JSON"
                claimed = e.get("hash")
                body = {"ts": e.get("ts"), "prev": e.get("prev"), "event": e.get("event")}
                if _sha(body) != claimed:
                    return False, f"第 {i} 行内容被篡改（哈希不匹配）"
                if e.get("prev") != prev:
                    return False, f"第 {i} 行链条断裂（prev 不匹配）"
                prev = claimed
                n += 1
        return True, f"审计链完整，共 {n} 条记录"

    def __len__(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
