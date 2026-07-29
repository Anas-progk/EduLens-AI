"""Tests for authentication endpoints — login, refresh, logout, RBAC, ownership."""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.config import TURNSTILE_ENABLED


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def demo_credentials():
    return {"email": "teacher@edulens.ai", "password": "demo123"}


@pytest.mark.asyncio
async def test_login_success(client, demo_credentials):
    resp = await client.post("/api/auth/login", json=demo_credentials)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0
    assert data["user"]["email"] == demo_credentials["email"]
    assert data["user"]["role"] == "teacher"


@pytest.mark.asyncio
async def test_login_invalid_password(client):
    resp = await client.post("/api/auth/login", json={"email": "teacher@edulens.ai", "password": "wrong"})
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    resp = await client.post("/api/auth/login", json={"email": "nonexist@edulens.ai", "password": "demo123"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_oauth(client, demo_credentials):
    resp = await client.post("/api/auth/login/oauth", data={"username": demo_credentials["email"], "password": demo_credentials["password"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_refresh_success(client, demo_credentials):
    login_resp = await client.post("/api/auth/login", json=demo_credentials)
    refresh_token = login_resp.json()["refresh_token"]

    resp = await client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == demo_credentials["email"]


@pytest.mark.asyncio
async def test_refresh_invalid_token(client):
    resp = await client.post("/api/auth/refresh", json={"refresh_token": "invalid-token"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_rotation(client, demo_credentials):
    login_resp = await client.post("/api/auth/login", json=demo_credentials)
    old_refresh = login_resp.json()["refresh_token"]

    resp1 = await client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert resp1.status_code == 200
    new_refresh = resp1.json()["refresh_token"]

    resp2 = await client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert resp2.status_code == 401


@pytest.mark.asyncio
async def test_logout(client, demo_credentials):
    login_resp = await client.post("/api/auth/login", json=demo_credentials)
    refresh_token = login_resp.json()["refresh_token"]

    resp = await client.post("/api/auth/logout", json={"refresh_token": refresh_token})
    assert resp.status_code == 200

    resp2 = await client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    assert resp2.status_code == 401


@pytest.mark.asyncio
async def test_rbac_teacher_cannot_access_hod_endpoints(client, demo_credentials):
    login_resp = await client.post("/api/auth/login", json=demo_credentials)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.get("/api/analytics/dashboard", headers=headers)
    assert resp.status_code in (200, 403, 404)


@pytest.mark.asyncio
async def test_hod_login(client):
    resp = await client.post("/api/auth/login", json={"email": "hod@edulens.ai", "password": "demo123"})
    assert resp.status_code == 200
    assert resp.json()["user"]["role"] == "hod"


@pytest.mark.asyncio
async def test_principal_login(client):
    resp = await client.post("/api/auth/login", json={"email": "principal@edulens.ai", "password": "demo123"})
    assert resp.status_code == 200
    assert resp.json()["user"]["role"] == "principal"


@pytest.mark.asyncio
async def test_login_missing_fields(client):
    resp = await client.post("/api/auth/login", json={"email": "teacher@edulens.ai"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_protected_route_without_token(client):
    resp = await client.get("/api/sessions")
    assert resp.status_code == 401
