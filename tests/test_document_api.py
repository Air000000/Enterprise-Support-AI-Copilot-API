from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

from auth import CurrentUser, get_current_user
from main import app
from models.document import Document
from routers import documents as documents_router
import services.document_service as document_service


@pytest.fixture(autouse=True)
def default_auth_user():
    user = CurrentUser(
        sub="support",
        user_id="user_demo",
        tenant_id="tenant_demo",
        role="support",
    )
    app.dependency_overrides[get_current_user] = lambda: user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "documents.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(document_service, "engine", test_engine)
    monkeypatch.setenv("DOCUMENT_STORAGE_ROOT", str(tmp_path / "storage"))

    with TestClient(app) as test_client:
        yield test_client

    SQLModel.metadata.drop_all(test_engine)


def test_upload_document_api_success(client):
    response = client.post(
        "/documents/upload",
        data={"category": "it"},
        files={
            "file": (
                "api_vpn_guide.md",
                b"# VPN Guide\n\nCheck network, account status, and VPN client config.",
                "text/markdown",
            ),
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["id"].startswith("doc_")
    assert data["tenant_id"] == "tenant_demo"
    assert data["uploaded_by"] == "user_demo"
    assert data["filename"] == "api_vpn_guide.md"
    assert data["file_type"] == "md"
    assert data["category"] == "it"
    assert data["status"] == "uploaded"
    assert data["version"] == 1
    assert data["chunk_count"] == 0
    assert data["checksum"]

    saved_path = Path(data["source_path"])
    assert saved_path.exists()
    assert saved_path.read_bytes().startswith(b"# VPN Guide")


def test_upload_document_api_rejects_unsupported_type(client):
    response = client.post(
        "/documents/upload",
        data={"category": "hr"},
        files={
            "file": (
                "policy.pdf",
                b"fake pdf content",
                "application/pdf",
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .md and .txt files are supported"


def test_list_documents_api_with_category_filter(client):
    upload_response = client.post(
        "/documents/upload",
        data={"category": "finance"},
        files={
            "file": (
                "api_invoice_rules.txt",
                b"Invoice reimbursement requires a valid invoice and approval record.",
                "text/plain",
            ),
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()

    list_response = client.get("/documents?category=finance")

    assert list_response.status_code == 200

    data = list_response.json()
    assert data["total"] >= 1
    assert data["limit"] == 20
    assert data["offset"] == 0
    assert any(item["id"] == uploaded["id"] for item in data["items"])
    assert all(item["category"] == "finance" for item in data["items"])


def test_get_document_api_success(client):
    upload_response = client.post(
        "/documents/upload",
        data={"category": "hr"},
        files={
            "file": (
                "api_leave_policy.md",
                b"# Leave Policy\n\nAnnual leave should be submitted in the HR system.",
                "text/markdown",
            ),
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()

    get_response = client.get(f"/documents/{uploaded['id']}")

    assert get_response.status_code == 200

    data = get_response.json()
    assert data["id"] == uploaded["id"]
    assert data["filename"] == "api_leave_policy.md"
    assert data["file_type"] == "md"
    assert data["category"] == "hr"
    assert data["status"] == "uploaded"


def test_get_missing_document_api_returns_404(client):
    response = client.get("/documents/doc_missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_index_document_api_success(client, monkeypatch):
    upload_response = client.post(
        "/documents/upload",
        data={"category": "it"},
        files={
            "file": (
                "api_index_vpn.md",
                b"# VPN Guide\n\nCheck network and VPN client configuration.",
                "text/markdown",
            ),
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()

    def fake_index_document(*, document_id: str, tenant_id: str) -> Document:
        assert document_id == uploaded["id"]
        assert tenant_id == "tenant_demo"

        return Document(
            id=document_id,
            tenant_id=tenant_id,
            uploaded_by=uploaded["uploaded_by"],
            filename=uploaded["filename"],
            file_type=uploaded["file_type"],
            category=uploaded["category"],
            source_path=uploaded["source_path"],
            status="indexed",
            version=uploaded["version"],
            checksum=uploaded["checksum"],
            chunk_count=2,
        )

    monkeypatch.setattr(
        documents_router,
        "index_document_service",
        fake_index_document,
    )

    index_response = client.post(f"/documents/{uploaded['id']}/index")

    assert index_response.status_code == 200

    data = index_response.json()
    assert data["document_id"] == uploaded["id"]
    assert data["status"] == "indexed"
    assert data["chunk_count"] == 2


def test_delete_document_api_success(client, monkeypatch):
    upload_response = client.post(
        "/documents/upload",
        data={"category": "admin"},
        files={
            "file": (
                "api_delete_access_card.md",
                b"# Access Card\n\nReport lost cards to admin.",
                "text/markdown",
            ),
        },
    )

    assert upload_response.status_code == 201
    uploaded = upload_response.json()

    def fake_delete_document(*, document_id: str, tenant_id: str):
        assert document_id == uploaded["id"]
        assert tenant_id == "tenant_demo"

        return (
            Document(
                id=document_id,
                tenant_id=tenant_id,
                uploaded_by=uploaded["uploaded_by"],
                filename=uploaded["filename"],
                file_type=uploaded["file_type"],
                category=uploaded["category"],
                source_path=uploaded["source_path"],
                status="deleted",
                version=uploaded["version"],
                checksum=uploaded["checksum"],
                chunk_count=0,
            ),
            2,
        )

    monkeypatch.setattr(
        documents_router,
        "delete_document_service",
        fake_delete_document,
    )

    delete_response = client.delete(f"/documents/{uploaded['id']}")

    assert delete_response.status_code == 200

    data = delete_response.json()
    assert data["document_id"] == uploaded["id"]
    assert data["status"] == "deleted"
    assert data["deleted_embeddings"] == 2