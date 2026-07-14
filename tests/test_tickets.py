import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

from auth import CurrentUser, get_current_user
from main import app
import services.ticket_service as ticket_service


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
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "tickets.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(ticket_service, "engine", test_engine)

    with TestClient(app) as test_client:
        yield test_client

    SQLModel.metadata.drop_all(test_engine)


def test_create_ticket(client):
    response = client.post(
        "/tickets",
        json={
            "title": "VPN cannot connect",
            "description": "Employee cannot connect to company VPN while working remotely.",
            "category": "it",
            "priority": "high",
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["title"] == "VPN cannot connect"
    assert data["description"] == "Employee cannot connect to company VPN while working remotely."
    assert data["category"] == "it"
    assert data["priority"] == "high"
    assert data["status"] == "open"
    assert data["tenant_id"] == "tenant_demo"
    assert data["created_by"] == "user_demo"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_ticket_crud_flow(client):
    create_response = client.post(
        "/tickets",
        json={
            "title": "Invoice rejected",
            "description": "Finance rejected a supplier invoice because the amount does not match the contract.",
            "category": "finance",
            "priority": "medium",
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    ticket_id = created["id"]

    get_response = client.get(f"/tickets/{ticket_id}")
    assert get_response.status_code == 200

    fetched = get_response.json()
    assert fetched["id"] == ticket_id
    assert fetched["title"] == "Invoice rejected"
    assert fetched["category"] == "finance"
    assert fetched["status"] == "open"

    patch_response = client.patch(
        f"/tickets/{ticket_id}",
        json={
            "status": "in_progress",
            "priority": "high",
        },
    )

    assert patch_response.status_code == 200

    updated = patch_response.json()
    assert updated["id"] == ticket_id
    assert updated["status"] == "in_progress"
    assert updated["priority"] == "high"
    assert updated["title"] == "Invoice rejected"

    list_response = client.get("/tickets")
    assert list_response.status_code == 200

    tickets = list_response.json()
    assert any(ticket["id"] == ticket_id for ticket in tickets)


def test_list_tickets_with_category_filter(client):
    client.post(
        "/tickets",
        json={
            "title": "Meeting room issue",
            "description": "The meeting room was booked but the room is still occupied.",
            "category": "admin",
            "priority": "medium",
        },
    )

    client.post(
        "/tickets",
        json={
            "title": "Data access request",
            "description": "Employee needs access to internal reporting data.",
            "category": "security",
            "priority": "high",
        },
    )

    response = client.get("/tickets?category=admin")

    assert response.status_code == 200

    data = response.json()
    assert len(data) >= 1
    assert all(ticket["category"] == "admin" for ticket in data)


def test_list_tickets_with_status_filter(client):
    create_response = client.post(
        "/tickets",
        json={
            "title": "Leave policy question",
            "description": "Employee asks whether sick leave needs to be submitted in the HR system.",
            "category": "hr",
            "priority": "low",
        },
    )

    ticket_id = create_response.json()["id"]

    client.patch(
        f"/tickets/{ticket_id}",
        json={
            "status": "resolved",
        },
    )

    response = client.get("/tickets?status=resolved")

    assert response.status_code == 200

    data = response.json()
    assert any(ticket["id"] == ticket_id for ticket in data)
    assert all(ticket["status"] == "resolved" for ticket in data)


def test_get_missing_ticket_should_return_404(client):
    response = client.get("/tickets/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"


def test_update_missing_ticket_should_return_404(client):
    response = client.patch(
        "/tickets/999999",
        json={
            "status": "closed",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"


def test_create_ticket_with_empty_title_should_fail(client):
    response = client.post(
        "/tickets",
        json={
            "title": "",
            "description": "Description is valid.",
            "category": "it",
            "priority": "medium",
        },
    )

    assert response.status_code == 422


def test_create_ticket_with_invalid_category_should_fail(client):
    response = client.post(
        "/tickets",
        json={
            "title": "Invalid category ticket",
            "description": "This should fail because category is invalid.",
            "category": "invalid",
            "priority": "medium",
        },
    )

    assert response.status_code == 422


def test_update_ticket_with_invalid_status_should_fail(client):
    create_response = client.post(
        "/tickets",
        json={
            "title": "Status validation ticket",
            "description": "This ticket is used to test invalid status validation.",
            "category": "it",
            "priority": "medium",
        },
    )

    ticket_id = create_response.json()["id"]

    response = client.patch(
        f"/tickets/{ticket_id}",
        json={
            "status": "invalid_status",
        },
    )

    assert response.status_code == 422