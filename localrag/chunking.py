"""文档加载与切分。

核心能力（.txt/.md/.rst/.log/.csv）零第三方依赖；
.pdf 需要可选依赖 `pypdf`，缺失时给出明确报错而不是静默失败。
每个 chunk 保留在原文中的字符区间与页码，用于**可审计溯源**。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".yaml", ".yml"}


@dataclass
class LoadedDoc:
    text: str
    title: str
    pages: list = field(default_factory=lambda: [(1, 0)])  # [(page_no, start_offset), ...]


def file_hash(path: str) -> str:
    """内容哈希：用于增量索引（内容未变则跳过重建）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_text(path: str) -> LoadedDoc:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return LoadedDoc(text=text, title=os.path.basename(path), pages=[(1, 0)])

    if ext == ".pdf":
        try:
            import pypdf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "读取 PDF 需要可选依赖：pip install pypdf"
            ) from exc
        reader = pypdf.PdfReader(path)
        parts: list[str] = []
        pages: list[tuple[int, int]] = []
        offset = 0
        for i, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            pages.append((i, offset))
            parts.append(t)
            offset += len(t) + 1
        return LoadedDoc(text="\n".join(parts), title=os.path.basename(path), pages=pages)

    raise RuntimeError(
        f"不支持的文件类型：{ext}（内置支持 {'/'.join(sorted(TEXT_EXT))}，PDF 需 pip install pypdf）"
    )


def page_of(pages: list, pos: int) -> int:
    """把字符偏移映射回页码（溯源用）。"""
    cur = pages[0][0] if pages else 1
    for no, start in pages:
        if pos >= start:
            cur = no
        else:
            break
    return cur


def chunk_text(text: str, size: int = 800, overlap: int = 120):
    """按字符窗口切分，尽量不在词中间断开；返回 [(text, start, end), ...]。"""
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")

    step = max(1, size - overlap)
    out = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + size)
        if j < n:  # 尝试回退到最近的空格，避免切断单词
            k = text.rfind(" ", i, j)
            if k > i + size // 2:
                j = k
        piece = text[i:j].strip()
        if piece:
            out.append((piece, i, j))
        if j >= n:
            break
        i += step
    return out
