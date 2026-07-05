from fastapi.testclient import TestClient
import pytest

from auth import get_current_user
from main import app


@pytest.fixture(autouse=True)
def use_real_auth(default_auth_user):
    app.dependency_overrides.pop(get_current_user, None)
    yield
    app.dependency_overrides.pop(get_current_user, None)


def auth_headers(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/auth/token",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_demo_token_and_me_flow():
    with TestClient(app) as client:
        me_response = client.get(
            "/auth/me",
            headers=auth_headers(client, "support", "support"),
        )

    assert me_response.status_code == 200
    assert me_response.json() == {
        "sub": "support",
        "user_id": "support_demo",
        "tenant_id": "tenant_demo",
        "role": "support",
    }


def test_demo_token_rejects_bad_password():
    with TestClient(app) as client:
        response = client.post(
            "/auth/token",
            json={"username": "support", "password": "wrong"},
        )

    assert response.status_code == 401


def test_me_requires_bearer_token():
    with TestClient(app) as client:
        response = client.get("/auth/me")

    assert response.status_code == 401


def test_me_rejects_malformed_token():
    with TestClient(app) as client:
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not-a-jwt"},
        )

    assert response.status_code == 401


def test_me_rejects_tampered_token():
    with TestClient(app) as client:
        headers = auth_headers(client, "support", "support")
        token = headers["Authorization"].removeprefix("Bearer ")
        header, payload, signature = token.split(".")
        tampered_payload = payload[:-1] + ("a" if payload[-1] != "a" else "b")
        tampered = f"{header}.{tampered_payload}.{signature}"
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {tampered}"},
        )

    assert response.status_code == 401


def test_me_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "-1")
    with TestClient(app) as client:
        headers = auth_headers(client, "support", "support")
        response = client.get("/auth/me", headers=headers)

    assert response.status_code == 401


def test_business_endpoint_requires_bearer_token():
    with TestClient(app) as client:
        response = client.get("/todos")

    assert response.status_code == 401


def test_employee_cannot_access_documents():
    with TestClient(app) as client:
        response = client.get(
            "/documents",
            headers=auth_headers(client, "employee", "employee"),
        )

    assert response.status_code == 403


def test_support_can_access_documents():
    with TestClient(app) as client:
        response = client.get(
            "/documents",
            headers=auth_headers(client, "support", "support"),
        )

    assert response.status_code == 200


def test_employee_cannot_access_agent_ops():
    with TestClient(app) as client:
        response = client.get(
            "/agent-ops/metrics/summary",
            headers=auth_headers(client, "employee", "employee"),
        )

    assert response.status_code == 403


def test_support_can_access_agent_ops():
    with TestClient(app) as client:
        response = client.get(
            "/agent-ops/metrics/summary",
            headers=auth_headers(client, "support", "support"),
        )

    assert response.status_code == 200


def test_ticket_is_hidden_from_other_tenant():
    with TestClient(app) as client:
        create_response = client.post(
            "/tickets",
            json={
                "title": "Tenant scoped ticket",
                "description": "Only tenant_demo should see this ticket.",
                "category": "it",
                "priority": "medium",
            },
            headers=auth_headers(client, "support", "support"),
        )
        assert create_response.status_code == 201
        ticket_id = create_response.json()["id"]

        other_tenant_response = client.get(
            f"/tickets/{ticket_id}",
            headers=auth_headers(client, "support_other", "support"),
        )

    assert other_tenant_response.status_code == 404
