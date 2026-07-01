from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException
from sqlmodel import Session, select

from database import engine
from rag_runtime.build_chroma_index import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    embed_texts,
    get_chroma_client,
)
from rag_runtime.text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    split_text,
)
from models.document import (
    Document,
    DocumentChunk,
    generate_document_id,
    utc_now,
)


ALLOWED_FILE_TYPES = {"md", "txt"}
ALLOWED_CATEGORIES = {"it", "hr", "finance", "admin", "security", "other"}

DEFAULT_STORAGE_ROOT = Path("storage/documents")
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024


def calculate_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")

    if suffix not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only .md and .txt files are supported",
        )

    return suffix


def build_safe_filename(filename: str) -> str:
    original_name = Path(filename or "").name

    if not original_name:
        raise HTTPException(status_code=400, detail="Filename is required")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name).strip("._")

    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename is invalid")

    return safe_name


def validate_document_content(content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File is too large",
        )


def validate_category(category: str) -> str:
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid document category")

    return category


def build_document_storage_path(
    storage_root: Path,
    tenant_id: str,
    document_id: str,
    version: int,
    safe_filename: str,
) -> Path:
    return storage_root / tenant_id / document_id / f"v{version}" / safe_filename


def create_document_from_bytes(
    *,
    filename: str,
    content: bytes,
    tenant_id: str,
    uploaded_by: str,
    category: str = "other",
    storage_root: Path = DEFAULT_STORAGE_ROOT,
) -> Document:
    safe_filename = build_safe_filename(filename)
    file_type = get_file_type(safe_filename)
    validate_document_content(content)
    category = validate_category(category)

    document_id = generate_document_id()
    version = 1
    checksum = calculate_checksum(content)

    save_path = build_document_storage_path(
        storage_root=storage_root,
        tenant_id=tenant_id,
        document_id=document_id,
        version=version,
        safe_filename=safe_filename,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        uploaded_by=uploaded_by,
        filename=filename,
        file_type=file_type,
        category=category,
        source_path=save_path.as_posix(),
        status="uploaded",
        version=version,
        checksum=checksum,
    )

    with Session(engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)
        return document


def list_documents(
    *,
    tenant_id: str,
    category: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Document], int]:
    with Session(engine) as session:
        statement = select(Document).where(Document.tenant_id == tenant_id)

        if category is not None:
            statement = statement.where(Document.category == category)

        if status is not None:
            statement = statement.where(Document.status == status)
        else:
            statement = statement.where(Document.status != "deleted")

        documents = list(session.exec(statement).all())
        total = len(documents)

        return documents[offset : offset + limit], total


def get_document(
    *,
    document_id: str,
    tenant_id: str,
) -> Document:
    with Session(engine) as session:
        document = session.get(Document, document_id)

        if (
            document is None
            or document.tenant_id != tenant_id
            or document.status == "deleted"
        ):
            raise HTTPException(status_code=404, detail="Document not found")

        return document
    

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def build_embedding_id(
    *,
    document_id: str,
    version: int,
    chunk_index: int,
) -> str:
    return f"doc:{document_id}:v{version}:chunk:{chunk_index}"


def build_chunk_metadata(
    *,
    document: Document,
    chunk_index: int,
    embedding_id: str,
) -> dict[str, Any]:
    return {
        "chunk_id": embedding_id,
        "document_id": document.id,
        "document_db_id": document.id,
        "title": Path(document.filename).stem,
        "filename": document.filename,
        "source_path": document.source_path,
        "chunk_index": chunk_index,
        "tenant_id": document.tenant_id,
        "category": document.category,
        "version": document.version,
        "checksum": document.checksum,
        "source_type": "uploaded_document",
    }


def build_embedding_text(
    *,
    document: Document,
    content: str,
) -> str:
    title = Path(document.filename).stem
    return f"{title}\n\n{content}"


def get_document_chroma_collection(
    *,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    client = get_chroma_client(chroma_dir)

    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception:
        return client.create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "description": "Local document RAG collection with uploaded documents",
            },
        )


def delete_existing_document_chunks(
    *,
    session: Session,
    document_id: str,
    collection: Any,
) -> int:
    existing_chunks = list(
        session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ).all()
    )

    embedding_ids = [
        chunk.embedding_id
        for chunk in existing_chunks
        if chunk.embedding_id
    ]

    if embedding_ids:
        collection.delete(ids=embedding_ids)

    for chunk in existing_chunks:
        session.delete(chunk)

    session.commit()

    return len(embedding_ids)


def index_document(
    *,
    document_id: str,
    tenant_id: str,
    embedding_function: EmbeddingFunction = embed_texts,
    chroma_collection: Any | None = None,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Document:
    with Session(engine) as session:
        document = session.get(Document, document_id)

        if (
            document is None
            or document.tenant_id != tenant_id
            or document.status == "deleted"
        ):
            raise HTTPException(status_code=404, detail="Document not found")

        document.status = "indexing"
        document.error_message = None
        document.updated_at = utc_now()
        session.add(document)
        session.commit()
        session.refresh(document)

        try:
            source_path = Path(document.source_path)

            if not source_path.exists():
                raise RuntimeError(f"Document source file not found: {source_path}")

            text = source_path.read_text(encoding="utf-8").strip()

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Document content is empty",
                )

            chunk_texts = split_text(
                text=text,
                chunk_size=DEFAULT_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
                min_chunk_size=MIN_CHUNK_SIZE,
            )

            if not chunk_texts:
                raise HTTPException(
                    status_code=400,
                    detail="No chunks generated from document",
                )

            collection = chroma_collection or get_document_chroma_collection(
                chroma_dir=chroma_dir,
                collection_name=collection_name,
            )

            delete_existing_document_chunks(
                session=session,
                document_id=document.id,
                collection=collection,
            )
            session.refresh(document)

            embedding_ids: list[str] = []
            chroma_documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embedding_texts: list[str] = []

            for chunk_index, content in enumerate(chunk_texts):
                embedding_id = build_embedding_id(
                    document_id=document.id,
                    version=document.version,
                    chunk_index=chunk_index,
                )
                metadata = build_chunk_metadata(
                    document=document,
                    chunk_index=chunk_index,
                    embedding_id=embedding_id,
                )

                embedding_ids.append(embedding_id)
                chroma_documents.append(content)
                metadatas.append(metadata)
                embedding_texts.append(
                    build_embedding_text(
                        document=document,
                        content=content,
                    )
                )

            embeddings = embedding_function(embedding_texts)

            if len(embeddings) != len(chunk_texts):
                raise RuntimeError(
                    f"Embedding count mismatch: chunks={len(chunk_texts)}, "
                    f"embeddings={len(embeddings)}"
                )

            collection.add(
                ids=embedding_ids,
                documents=chroma_documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            for chunk_index, content in enumerate(chunk_texts):
                metadata = metadatas[chunk_index]
                chunk = DocumentChunk(
                    tenant_id=document.tenant_id,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    content=content,
                    category=document.category,
                    metadata_json=json.dumps(metadata, ensure_ascii=False),
                    embedding_id=embedding_ids[chunk_index],
                )
                session.add(chunk)

            document.status = "indexed"
            document.chunk_count = len(chunk_texts)
            document.error_message = None
            document.updated_at = utc_now()
            session.add(document)
            session.commit()
            session.refresh(document)

            return document

        except Exception as exc:
            document.status = "failed"
            document.error_message = (
                str(exc.detail)
                if isinstance(exc, HTTPException)
                else str(exc)
            )
            document.updated_at = utc_now()
            session.add(document)
            session.commit()

            if isinstance(exc, HTTPException):
                raise exc

            raise HTTPException(
                status_code=500,
                detail=f"Document indexing failed: {exc}",
            )
        

def delete_document(
    *,
    document_id: str,
    tenant_id: str,
    chroma_collection: Any | None = None,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[Document, int]:
    with Session(engine) as session:
        document = session.get(Document, document_id)

        if (
            document is None
            or document.tenant_id != tenant_id
            or document.status == "deleted"
        ):
            raise HTTPException(status_code=404, detail="Document not found")

        collection = chroma_collection or get_document_chroma_collection(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
        )

        deleted_embeddings = delete_existing_document_chunks(
            session=session,
            document_id=document.id,
            collection=collection,
        )
        session.refresh(document)

        document.status = "deleted"
        document.error_message = None
        document.updated_at = utc_now()
        session.add(document)
        session.commit()
        session.refresh(document)

        return document, deleted_embeddings