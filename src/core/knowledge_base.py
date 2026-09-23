"""把 `data/knowledge_base/` 下的文档读成可检索的文本块。

知识库是决策智能体（DA）唯一的证据来源。它检索得准不准，直接决定 DA 的
特征评分有没有意义——喂给它什么，它就依据什么下判断。

支持 `.md` / `.txt`；`.pdf` 需要额外装 `pypdf`，没装就跳过并打印提示，
不报错、不中断。切块按 Markdown 标题来做，因为 `notes/` 下的文档本来就是
按「一节讲一个机制」组织的，一个标题下的内容正好是一个完整的论证单元。
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_ROOT = _PROJECT_ROOT / "data" / "knowledge_base"

# 索引目录本身不该被当成语料读进来
_IGNORED_DIRS = {"index", ".git"}

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_PDF_SUFFIXES = {".pdf"}

# 一个文本块的目标长度。中文字符信息密度高，500 字已经能容纳一个完整的
# 机制论证；再长会让向量被稀释，检索时反而不准。
_DEFAULT_MAX_CHARS = 500


def chunk_markdown(
    text: str, *, max_chars: int = _DEFAULT_MAX_CHARS, title: Optional[str] = None
) -> List[Tuple[Optional[str], str]]:
    """按 Markdown 标题切块，返回 [(小节标题, 正文), ...]。

    每个块会把所在文档的标题和自身的小节标题附在正文开头，这样单独看一个块
    也知道它出自哪里——检索命中后 DA 能看到上下文，不会拿着半句话当证据。

    超长的小节会按空行再切一次，避免一个块里塞进好几个不相干的论点。
    """
    lines = text.splitlines()
    preamble: List[str] = []
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if current_heading is None and not current_body:
                # 文档开头的 `# 标题` 是文档标题，不当成一个小节
                if title is None:
                    title = heading
                    continue
            if current_heading is not None or current_body:
                sections.append((current_heading, current_body))
            current_heading = heading
            current_body = []
        elif current_heading is None and not sections:
            if stripped:
                preamble.append(line)
        else:
            current_body.append(line)

    if current_heading is not None or current_body:
        sections.append((current_heading, current_body))

    if preamble:
        sections.insert(0, (None, preamble))

    chunks: List[Tuple[Optional[str], str]] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        pieces = _split_long_body(body, max_chars)
        for piece in pieces:
            header_parts = [part for part in (title, heading) if part]
            prefix = " > ".join(header_parts)
            content = f"{prefix}\n{piece}" if prefix else piece
            chunks.append((heading, content))
    return chunks


def _split_long_body(body: str, max_chars: int) -> List[str]:
    """把小节正文按空行边界切成不超过 max_chars 的片段。"""
    if len(body) <= max_chars:
        return [body]

    pieces: List[str] = []
    buffer = ""
    for block in body.split("\n\n"):
        candidate = f"{buffer}\n\n{block}" if buffer else block
        if len(candidate) <= max_chars or not buffer:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = block
    if buffer:
        pieces.append(buffer)
    return pieces


def _read_pdf(path: Path) -> Optional[str]:
    """尽力读取 PDF。没装 pypdf 就返回 None（调用方负责提示）。"""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(
    root: Optional[Path] = None, *, verbose: bool = False
) -> List[Dict[str, Any]]:
    """读取知识库目录，返回文本块列表。

    每块形如::

        {"doc_id": str, "source": "notes/01_time_series_features.md",
         "heading": "中国外汇储备与中国央行干预能力", "content": str}
    """
    base = Path(root) if root else DEFAULT_KB_ROOT
    if not base.exists():
        if verbose:
            print(f"知识库目录不存在：{base}")
        return []

    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and not any(part in _IGNORED_DIRS for part in path.relative_to(base).parts)
    )

    chunks: List[Dict[str, Any]] = []
    skipped_pdfs: List[str] = []

    for path in files:
        suffix = path.suffix.lower()
        relative = path.relative_to(base).as_posix()

        if suffix in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix in _PDF_SUFFIXES:
            text = _read_pdf(path)
            if text is None:
                skipped_pdfs.append(relative)
                continue
        else:
            continue

        for index, (heading, content) in enumerate(chunk_markdown(text)):
            chunks.append({
                "doc_id": f"{relative}#{index}",
                "source": relative,
                "heading": heading,
                "content": content,
            })

    if verbose and skipped_pdfs:
        print(
            f"跳过了 {len(skipped_pdfs)} 个 PDF（未安装 pypdf，"
            f"需要的话运行：.\\venv\\Scripts\\pip.exe install pypdf）：{skipped_pdfs}"
        )
    return chunks


def corpus_fingerprint(chunks: List[Dict[str, Any]]) -> str:
    """语料指纹：内容没变就不必重建向量索引。"""
    digest = hashlib.sha1()
    for chunk in sorted(chunks, key=lambda item: item["doc_id"]):
        digest.update(chunk["doc_id"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk["content"].encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()
