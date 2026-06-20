import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth-test.db"))
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-more-than-32-characters")
    monkeypatch.setenv("ENVIRONMENT", "test")

    import api.auth as auth

    importlib.reload(auth)
    auth.init_auth_db()
    app = FastAPI()
    app.include_router(auth.auth_router)
    app.include_router(auth.admin_router)
    with TestClient(app) as test_client:
        yield test_client


ADMIN = {
    "username": "admin.test",
    "email": "admin@example.com",
    "password": "MotDePasse123",
    "role": "admin",
}


def bootstrap_and_login(client: TestClient) -> dict:
    assert client.post("/auth/bootstrap", json=ADMIN).status_code == 201
    response = client.post(
        "/auth/login",
        json={"username": ADMIN["username"], "password": ADMIN["password"]},
    )
    assert response.status_code == 200
    return response.json()


def test_bootstrap_is_available_only_once(client: TestClient):
    assert client.get("/auth/status").json() == {"initialized": False}
    assert client.post("/auth/bootstrap", json=ADMIN).status_code == 201
    assert client.get("/auth/status").json() == {"initialized": True}
    assert client.post("/auth/bootstrap", json=ADMIN).status_code == 409


def test_login_returns_a_valid_bearer_token(client: TestClient):
    login = bootstrap_and_login(client)
    assert login["token_type"] == "bearer"
    assert login["user"]["role"] == "admin"

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == ADMIN["username"]


def test_login_rejects_an_invalid_password(client: TestClient):
    client.post("/auth/bootstrap", json=ADMIN)
    response = client.post(
        "/auth/login",
        json={"username": ADMIN["username"], "password": "MauvaisMotDePasse1"},
    )
    assert response.status_code == 401


def test_admin_can_create_user_and_change_role(client: TestClient):
    bootstrap_and_login(client)
    created = client.post(
        "/admin/users",
        json={
            "username": "agent.credit",
            "email": "agent@example.com",
            "password": "Temporaire123",
            "role": "conseiller",
        },
    )
    assert created.status_code == 201
    user_id = created.json()["id"]

    updated = client.patch(f"/admin/users/{user_id}/role", json={"role": "analyste"})
    assert updated.status_code == 200
    assert updated.json()["role"] == "analyste"


def test_non_admin_cannot_manage_users(client: TestClient):
    bootstrap_and_login(client)
    client.post(
        "/admin/users",
        json={
            "username": "conseiller.test",
            "email": "conseiller@example.com",
            "password": "Temporaire123",
            "role": "conseiller",
        },
    )
    client.post("/auth/logout")
    login = client.post(
        "/auth/login",
        json={"username": "conseiller.test", "password": "Temporaire123"},
    )
    token = login.json()["access_token"]

    response = client.get("/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_disabled_user_loses_access_immediately(client: TestClient):
    bootstrap_and_login(client)
    created = client.post(
        "/admin/users",
        json={
            "username": "disabled.user",
            "email": "disabled@example.com",
            "password": "Temporaire123",
            "role": "analyste",
        },
    ).json()
    client.post("/auth/logout")
    login = client.post(
        "/auth/login",
        json={"username": "disabled.user", "password": "Temporaire123"},
    ).json()

    client.post(
        "/auth/login",
        json={"username": ADMIN["username"], "password": ADMIN["password"]},
    )
    assert client.patch(f"/admin/users/{created['id']}/active", json={"is_active": False}).status_code == 200

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {login['access_token']}"},
    )
    assert response.status_code == 401
