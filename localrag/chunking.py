"""文档加载与切分。

- 内置 `.txt/.md/.rst/.log/.csv/.json/.yaml` 与 **`.html/.htm`**（标准库解析，零依赖）；
- `.pdf` 需要可选依赖 `pypdf`，缺失时给出明确报错而不是静默失败；
- Markdown 支持**章节感知切分**，让 chunk 带上所属标题，检索与引用都能定位到章节；
- 每个 chunk 保留在原文中的字符区间与页码，用于**可审计溯源**。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".yaml", ".yml"}
HTML_EXT = {".html", ".htm"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


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


class _TextExtractor(HTMLParser):
    """提取 HTML 正文与标题（标准库，无 BeautifulSoup 依赖）。"""

    _BREAK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        else:
            self.parts.append(data)


def load_html(path: str) -> LoadedDoc:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    ex = _TextExtractor()
    ex.feed(raw)
    text = "".join(ex.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return LoadedDoc(text=text,
                     title=ex.title.strip() or os.path.basename(path),
                     pages=[(1, 0)])


def load_text(path: str) -> LoadedDoc:
    ext = os.path.splitext(path)[1].lower()
    if ext in HTML_EXT:
        return load_html(path)
    if ext in TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return LoadedDoc(text=text, title=os.path.basename(path), pages=[(1, 0)])

    if ext == ".pdf":
        try:
            import pypdf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("读取 PDF 需要可选依赖：pip install pypdf") from exc
        reader = pypdf.PdfReader(path)
        parts: list = []
        pages: list = []
        offset = 0
        for i, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            pages.append((i, offset))
            parts.append(t)
            offset += len(t) + 1
        return LoadedDoc(text="\n".join(parts), title=os.path.basename(path), pages=pages)

    raise RuntimeError(
        f"不支持的文件类型：{ext}"
        f"（内置支持 {'/'.join(sorted(TEXT_EXT | HTML_EXT))}，PDF 需 pip install pypdf）"
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


def is_markdown(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".md", ".markdown"}


def split_markdown_sections(text: str):
    """按标题切出章节，返回 [(标题路径, start, end), ...]。

    章节区间从**标题行本身**开始，因此首个 chunk 会自然包含标题，
    引用时定位到的就是原文真实位置（不做文本改写）。
    """
    marks = [(m.start(), len(m.group(1)), m.group(2).strip())
             for m in _HEADING.finditer(text)]
    if not marks:
        return [("", 0, len(text))]

    sections: list = []
    stack: list = []
    if marks[0][0] > 0:
        sections.append(("", 0, marks[0][0]))
    for i, (start, level, title) in enumerate(marks):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.append((path, start, end))
    return sections


def chunk_text(text: str, size: int = 800, overlap: int = 120,
               start_at: int = 0, end_at: int | None = None):
    """按字符窗口切分，尽量不在词中间断开；返回 [(text, start, end), ...]。"""
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")

    lo = start_at
    hi = len(text) if end_at is None else min(len(text), end_at)
    step = max(1, size - overlap)
    out = []
    i = lo
    while i < hi:
        j = min(hi, i + size)
        if j < hi:  # 尝试回退到最近的空格，避免切断单词
            k = text.rfind(" ", i, j)
            if k > i + size // 2:
                j = k
        piece = text[i:j].strip()
        if piece:
            out.append((piece, i, j))
        if j >= hi:
            break
        i += step
    return out


def chunk_document(text: str, size: int = 800, overlap: int = 120,
                   markdown: bool = False):
    """统一入口，返回 [(text, start, end, section), ...]。

    markdown=True 时按章节切分，并回填所属标题路径。
    """
    if not markdown:
        return [(t, s, e, "") for (t, s, e) in chunk_text(text, size, overlap)]

    out = []
    for path, start, end in split_markdown_sections(text):
        for (t, s, e) in chunk_text(text, size, overlap, start_at=start, end_at=end):
            out.append((t, s, e, path))
    return out
