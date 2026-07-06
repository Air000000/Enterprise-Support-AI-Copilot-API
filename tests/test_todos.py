import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

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
