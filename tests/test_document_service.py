from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from sqlmodel import Session, select

from database import create_db_and_tables, engine
from models.document import DocumentChunk
from services.document_service import (
    calculate_checksum,
    create_document_from_bytes,
    get_document,
    index_document,
    list_documents,
)


def unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def test_create_document_from_bytes_saves_file_and_record(tmp_path: Path):
    create_db_and_tables()

    content = b"# VPN Guide\n\nCheck network, account status, and VPN client config."
    tenant_id = unique_name("tenant_demo")

    document = create_document_from_bytes(
        filename="vpn_guide.md",
        content=content,
        tenant_id=tenant_id,
        uploaded_by="user_demo",
        category="it",
        storage_root=tmp_path,
    )

    assert document.id.startswith("doc_")
    assert document.tenant_id == tenant_id
    assert document.uploaded_by == "user_demo"
    assert document.filename == "vpn_guide.md"
    assert document.file_type == "md"
    assert document.category == "it"
    assert document.status == "uploaded"
    assert document.version == 1
    assert document.checksum == calculate_checksum(content)

    saved_path = Path(document.source_path)
    assert saved_path.exists()
    assert saved_path.read_bytes() == content


def test_create_document_rejects_unsupported_file_type(tmp_path: Path):
    create_db_and_tables()

    with pytest.raises(HTTPException) as exc_info:
        create_document_from_bytes(
            filename="policy.pdf",
            content=b"fake pdf content",
            tenant_id=unique_name("tenant_demo"),
            uploaded_by="user_demo",
            category="hr",
            storage_root=tmp_path,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Only .md and .txt files are supported"


def test_list_documents_filters_by_tenant_and_category(tmp_path: Path):
    create_db_and_tables()

    tenant_a = unique_name("tenant_a")
    tenant_b = unique_name("tenant_b")

    hr_document = create_document_from_bytes(
        filename="leave_policy.txt",
        content=b"Annual leave must be submitted in the HR system.",
        tenant_id=tenant_a,
        uploaded_by="user_demo",
        category="hr",
        storage_root=tmp_path,
    )

    create_document_from_bytes(
        filename="vpn_guide.md",
        content=b"VPN troubleshooting guide.",
        tenant_id=tenant_a,
        uploaded_by="user_demo",
        category="it",
        storage_root=tmp_path,
    )

    create_document_from_bytes(
        filename="other_tenant_hr.txt",
        content=b"Other tenant HR policy.",
        tenant_id=tenant_b,
        uploaded_by="user_demo",
        category="hr",
        storage_root=tmp_path,
    )

    documents, total = list_documents(
        tenant_id=tenant_a,
        category="hr",
    )

    assert total == 1
    assert len(documents) == 1
    assert documents[0].id == hr_document.id
    assert documents[0].tenant_id == tenant_a
    assert documents[0].category == "hr"


def test_get_document_requires_matching_tenant(tmp_path: Path):
    create_db_and_tables()

    tenant_id = unique_name("tenant_demo")
    document = create_document_from_bytes(
        filename="invoice_rules.md",
        content=b"Invoice rules for reimbursement.",
        tenant_id=tenant_id,
        uploaded_by="user_demo",
        category="finance",
        storage_root=tmp_path,
    )

    fetched = get_document(
        document_id=document.id,
        tenant_id=tenant_id,
    )

    assert fetched.id == document.id
    assert fetched.tenant_id == tenant_id

    with pytest.raises(HTTPException) as exc_info:
        get_document(
            document_id=document.id,
            tenant_id=unique_name("tenant_other"),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Document not found"


class FakeChromaCollection:
    def __init__(self) -> None:
        self.added_ids: list[str] = []
        self.added_documents: list[str] = []
        self.added_metadatas: list[dict] = []
        self.deleted_ids: list[str] = []

    def add(self, *, ids, documents, embeddings, metadatas):
        self.added_ids.extend(ids)
        self.added_documents.extend(documents)
        self.added_metadatas.extend(metadatas)

    def delete(self, *, ids):
        self.deleted_ids.extend(ids)


def fake_embed_texts(texts: list[str]) -> list[list[float]]:
    return [
        [float(index), 0.0, 1.0]
        for index, _ in enumerate(texts)
    ]


def test_index_document_creates_chunks_and_updates_status(tmp_path: Path):
    create_db_and_tables()

    content = (
        "# VPN Guide\n\n"
        "When VPN connection fails, users should first check local network, "
        "account status, MFA status, and VPN client configuration.\n\n"
        "If the problem still exists, users should collect error screenshots "
        "and submit an IT support ticket."
    ).encode("utf-8")

    tenant_id = unique_name("tenant_demo")
    document = create_document_from_bytes(
        filename="vpn_guide.md",
        content=content,
        tenant_id=tenant_id,
        uploaded_by="user_demo",
        category="it",
        storage_root=tmp_path,
    )

    fake_collection = FakeChromaCollection()

    indexed_document = index_document(
        document_id=document.id,
        tenant_id=tenant_id,
        embedding_function=fake_embed_texts,
        chroma_collection=fake_collection,
    )

    assert indexed_document.id == document.id
    assert indexed_document.status == "indexed"
    assert indexed_document.chunk_count >= 1
    assert indexed_document.error_message is None

    with Session(engine) as session:
        chunks = list(
            session.exec(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == document.id
                )
            ).all()
        )

    assert len(chunks) == indexed_document.chunk_count
    assert len(fake_collection.added_ids) == indexed_document.chunk_count
    assert fake_collection.added_ids[0].startswith(f"doc:{document.id}:v1:chunk:")
    assert fake_collection.added_metadatas[0]["document_id"] == document.id
    assert fake_collection.added_metadatas[0]["tenant_id"] == tenant_id
    assert fake_collection.added_metadatas[0]["category"] == "it"
    assert fake_collection.added_metadatas[0]["source_type"] == "uploaded_document"


def test_reindex_document_replaces_existing_chunks(tmp_path: Path):
    create_db_and_tables()

    content = (
        "# Leave Policy\n\n"
        "Annual leave should be submitted in the HR system before approval. "
        "Managers should review the leave request according to team workload."
    ).encode("utf-8")

    tenant_id = unique_name("tenant_demo")
    document = create_document_from_bytes(
        filename="leave_policy.md",
        content=content,
        tenant_id=tenant_id,
        uploaded_by="user_demo",
        category="hr",
        storage_root=tmp_path,
    )

    fake_collection = FakeChromaCollection()

    first_indexed = index_document(
        document_id=document.id,
        tenant_id=tenant_id,
        embedding_function=fake_embed_texts,
        chroma_collection=fake_collection,
    )

    first_ids = list(fake_collection.added_ids)

    second_indexed = index_document(
        document_id=document.id,
        tenant_id=tenant_id,
        embedding_function=fake_embed_texts,
        chroma_collection=fake_collection,
    )

    with Session(engine) as session:
        chunks = list(
            session.exec(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == document.id
                )
            ).all()
        )

    assert second_indexed.status == "indexed"
    assert second_indexed.chunk_count == first_indexed.chunk_count
    assert len(chunks) == second_indexed.chunk_count
    assert fake_collection.deleted_ids == first_ids