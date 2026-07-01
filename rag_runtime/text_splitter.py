from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rag_runtime.document_loader import Document, load_documents

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
MIN_CHUNK_SIZE = 150

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    content: str
    tenant_id: str
    category: str


def split_long_text_by_chars(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    当某个段落太长时，用字符窗口切分。

    chunk_size 表示每个 chunk 的最大字符数。
    overlap 表示相邻 chunk 之间重叠的字符数。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # overlap 让相邻 chunk 保留部分上下文，降低边界语义被切断的风险。
        start = end - overlap

    return chunks

def merge_small_chunks(
    chunks: list[str],
    min_chunk_size: int,
    chunk_size: int,
) -> list[str]:
    """
    合并过短的 chunk，避免标题或残句单独入库。

    规则：
    1. 标题类短 chunk 优先和后面的 chunk 合并。
    2. 普通短 chunk 优先合并到前一个 chunk。
    3. 允许合并后略微超过 chunk_size，但不超过 chunk_size + min_chunk_size。
    """
    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be greater than 0")

    if min_chunk_size >= chunk_size:
        raise ValueError("min_chunk_size must be smaller than chunk_size")

    merged: list[str] = []  # 最终结果列表；每个元素都是一个合并后的 chunk。
    pending: str | None = None  # pending 是等待合并的短 chunk，可能是标题或残句。
    soft_limit = chunk_size + min_chunk_size

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if pending:
            candidate = f"{pending}\n\n{chunk}".strip()

            if len(candidate) <= soft_limit:
                chunk = candidate
                pending = None
            else:
                merged.append(pending)
                pending = None

        is_small = len(chunk) < min_chunk_size
        is_heading = chunk.lstrip().startswith("#")

        if is_small and is_heading: # 标题类短 chunk 优先和后面 chunk 合并
            pending = chunk
            continue

        if is_small and merged: # 普通短 chunk 优先合并到前一个 chunk
            candidate = f"{merged[-1]}\n\n{chunk}".strip()

            if len(candidate) <= soft_limit:
                merged[-1] = candidate
            else:
                pending = chunk

            continue

        # if is_small:    
        #     continue
        
        if is_small:
            # 如果是第一个片段，且没有前文可合并，也没有下文可等待（因为不知道下文多长）
            # 最安全的做法是先放入 pending，或者如果它极短，直接保留
            if not merged:
                 pending = chunk # 放入 pending，让下一次循环尝试与第二个片段合并
                 continue
            else:
                 # 这里的逻辑其实已经被上面的 'is_small and merged' 覆盖了
                 pass 

        merged.append(chunk)

    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}".strip()
        else:
            merged.append(pending)

    return merged

def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """
    把一篇文档切成多个文本 chunk。

    当前策略：
    1. 优先按空行分段。
    2. 能放进同一个 chunk 的段落就合并。
    3. 如果某段太长，再按字符窗口切。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be greater than 0")
    
    if min_chunk_size >= chunk_size:
        raise ValueError("min_chunk_size must be smaller than chunk_size")

    # 统一 Windows / Linux / macOS 换行符，避免分段逻辑在不同系统上表现不一致。
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalized_text:
        return []

    # 优先按空行切自然段；相比按单个换行切，更能保留段落级语义完整性。
    paragraphs = [
        paragraph.strip()
        for paragraph in normalized_text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []
    # current_parts 是当前正在装箱的段落集合；装满后会合并成一个 chunk。
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        # 段落本身已经超过 chunk_size 时，不能再靠段落合并解决，只能用字符窗口兜底切分。
        if len(paragraph) > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_len = 0

            chunks.extend(
                split_long_text_by_chars(
                    text=paragraph,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
            continue

        # +2 是因为后续用 "\n\n" 拼接段落，两个换行符也会计入 chunk 长度。
        candidate_len = current_len + len(paragraph) + 2

        if current_parts and candidate_len > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [paragraph]
            current_len = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_len = candidate_len

    # 循环结束后，最后一个未装满的 chunk 也需要写入结果。
    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return merge_small_chunks(
        chunks=chunks,
        min_chunk_size=min_chunk_size,
        chunk_size=chunk_size,
    )


def build_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index}"


def split_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[Chunk]:
    texts = split_text(
        text=document.text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )

    chunks: list[Chunk] = []

    for index, content in enumerate(texts):
        chunk = Chunk(
            chunk_id=build_chunk_id(document.document_id, index),
            document_id=document.document_id,
            title=document.title,
            source_path=document.source_path,
            chunk_index=index,
            content=content,
            tenant_id=document.tenant_id,
            category=document.category,
        )
        chunks.append(chunk)

    return chunks


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for document in documents:
        document_chunks = split_document(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        all_chunks.extend(document_chunks)

    return all_chunks


def print_chunks_summary(chunks: Iterable[Chunk]) -> None:
    for chunk in chunks:
        print("=" * 80)
        print(f"chunk_id:     {chunk.chunk_id}")
        print(f"document_id:  {chunk.document_id}")
        print(f"title:        {chunk.title}")
        print(f"source_path:  {chunk.source_path}")
        print(f"chunk_index:  {chunk.chunk_index}")
        print(f"content_len:  {len(chunk.content)}")
        print(f"preview:      {chunk.content[:100].replace(chr(10), ' ')}...")
        print(f"tenant_id:    {chunk.tenant_id}")
        print(f"category:     {chunk.category}")


def main() -> None:
    docs_dir = Path("experiments/docs")
    documents = load_documents(docs_dir)

    # 这里故意用比较小的 chunk_size，方便你看到切块效果。
    # 真实 RAG 里可以调大，例如 500、800、1000。
    chunks = split_documents(
        documents=documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print_chunks_summary(chunks)

    print("\nFirst chunk as dict:")
    if chunks:
        print(asdict(chunks[0]))


if __name__ == "__main__":
    main()