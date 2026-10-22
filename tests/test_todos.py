import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine, select

import main
from auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def default_auth_user():
    user = CurrentUser(
        sub="support",
        user_id="user_demo",
        tenant_id="tenant_demo",
        role="support",
    )
    main.app.dependency_overrides[get_current_user] = lambda: user
    yield
    main.app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "todos.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(main, "engine", test_engine)

    with TestClient(main.app) as test_client:
        yield test_client

    SQLModel.metadata.drop_all(test_engine)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "time" in data


def test_create_todo(client):
    response = client.post(
        "/todos",
        json={
            "title": "Test create todo",
            "due_time": "today",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test create todo"
    assert data["completed"] is False
    assert data["due_time"] == "today"
    assert "id" in data


def test_todo_crud_flow(client):
    create_response = client.post(
        "/todos",
        json={
            "title": "Test CRUD todo",
            "due_time": "tomorrow",
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    todo_id = created["id"]

    get_response = client.get(f"/todos/{todo_id}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Test CRUD todo"

    patch_response = client.patch(
        f"/todos/{todo_id}",
        json={
            "title": "Updated CRUD todo",
            "completed": True,
            "due_time": "tonight",
        },
    )

    assert patch_response.status_code == 200
    updated = patch_response.json()
    assert updated["title"] == "Updated CRUD todo"
    assert updated["completed"] is True
    assert updated["due_time"] == "tonight"

    delete_response = client.delete(f"/todos/{todo_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "Todo deleted"

    missing_response = client.get(f"/todos/{todo_id}")
    assert missing_response.status_code == 404


def test_create_todo_with_empty_title_should_fail(client):
    response = client.post(
        "/todos",
        json={
            "title": "",
        },
    )

    assert response.status_code == 422


def test_real_jwt_tenant_isolation_includes_ai_creation_and_hides_unowned_rows(client, monkeypatch):
    main.app.dependency_overrides.pop(get_current_user, None)
    def headers(username):
        response = client.post("/auth/token", json={"username": username, "password": "support"})
        assert response.status_code == 200
        return {"Authorization": "Bearer " + response.json()["access_token"]}
    owner, other = headers("support"), headers("support_other")
    with Session(main.engine) as session:
        legacy = main.Todo(title="unassigned legacy row")
        session.add(legacy)
        session.commit()
        session.refresh(legacy)
        legacy_id = legacy.id
    first = client.post("/todos", headers=owner, json={"title": "owner task"}).json()
    for method, payload in (("GET", {}), ("PATCH", {"json": {"completed": True}}), ("DELETE", {})):
        assert client.request(method, f"/todos/{first['id']}", headers=other, **payload).status_code == 404
        assert client.request(method, f"/todos/{legacy_id}", headers=owner, **payload).status_code == 404
    assert client.get("/todos", headers=other).json() == []
    assert client.get(f"/todos/{first['id']}", headers=owner).json()["completed"] is False
    response = client.post("/todos", headers=other, json={"title": "other task", "tenant_id": "tenant_demo"})
    assert response.status_code == 201
    def complete(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "tasks": [{"title": "AI other task", "time": None, "tenant_id": "tenant_demo"}]})))])
    monkeypatch.setattr(main.llm_client.chat.completions, "create", complete)
    ai = client.post("/ai/create-todos", headers=other, json={"text": "synthetic", "tenant_id": "tenant_demo"})
    assert ai.status_code == 201
    assert [t["title"] for t in client.get("/todos", headers=owner).json()] == ["owner task"]
    assert {t["title"] for t in client.get("/todos", headers=other).json()} == {"other task", "AI other task"}
    with Session(main.engine) as session:
        assert session.get(main.Todo, legacy_id).tenant_id is None  # Preserved, not deleted or claimed.
        assert {t.title for t in session.exec(select(main.Todo).where(
            main.Todo.tenant_id == "tenant_other")).all()} == {"other task", "AI other task"}


@pytest.mark.parametrize("path,payload", [
    ("/ai/chat", {"message": "synthetic"}),
    ("/ai/extract-tasks", {"text": "synthetic"}),
    ("/ai/create-todos", {"text": "synthetic"}),
])
def test_legacy_ai_provider_errors_never_expose_body_in_response_or_logs(client, monkeypatch, caplog, path, payload):
    # Alembic's fileConfig can disable existing loggers in earlier migration tests.
    monkeypatch.setattr(main.logger, "disabled", False)
    caplog.set_level("ERROR", logger=main.logger.name)
    marker = "PRIVATE_PROVIDER_ERROR_BODY_AND_KEY_MARKER"
    def fail(**kwargs):
        error = RuntimeError(marker)
        error.add_note(marker)
        raise error from ValueError(marker)
    monkeypatch.setattr(main.llm_client.chat.completions, "create", fail)
    response = client.post(path, json=payload)
    assert response.status_code == 500 and response.json() == {"detail": "LLM service unavailable"}
    assert marker not in response.text and marker not in caplog.text
    assert "RuntimeError" in caplog.text
    with Session(main.engine) as session:
        assert session.exec(select(main.Todo)).all() == []
