from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


DEFAULT_DEMO_USERS: dict[str, dict[str, str]] = {
    "employee": {
        "password": "employee",
        "user_id": "user_demo",
        "tenant_id": "tenant_demo",
        "role": "employee",
    },
    "support": {
        "password": "support",
        "user_id": "support_demo",
        "tenant_id": "tenant_demo",
        "role": "support",
    },
    "admin": {
        "password": "admin",
        "user_id": "admin_demo",
        "tenant_id": "tenant_demo",
        "role": "admin",
    },
    "support_other": {
        "password": "support",
        "user_id": "support_other",
        "tenant_id": "tenant_other",
        "role": "support",
    },
}


@dataclass(frozen=True)
class CurrentUser:
    sub: str
    user_id: str
    tenant_id: str
    role: str


bearer_scheme = HTTPBearer(auto_error=False) 


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def get_jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "demo-jwt-secret")


def get_jwt_expire_minutes() -> int:
    return int(os.getenv("JWT_EXPIRE_MINUTES", "60"))


def get_demo_users() -> dict[str, dict[str, str]]:
    raw_users = os.getenv("AUTH_DEMO_USERS_JSON")
    if not raw_users:
        return DEFAULT_DEMO_USERS
    return json.loads(raw_users)


def create_access_token(user: CurrentUser) -> str:
    now = int(time.time())
    payload = {
        "sub": user.sub,
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "role": user.role,
        "exp": now + get_jwt_expire_minutes() * 60,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
    )
    signature = hmac.new(
        get_jwt_secret().encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def decode_access_token(token: str) -> CurrentUser:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    signing_input = f"{header_b64}.{payload_b64}"
    expected_signature = hmac.new(
        get_jwt_secret().encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()

    try:
        if not hmac.compare_digest(_b64decode(signature_b64), expected_signature):
            raise HTTPException(status_code=401, detail="Invalid token")

        payload: dict[str, Any] = json.loads(_b64decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise HTTPException(status_code=401, detail="Token expired")

        return CurrentUser(
            sub=str(payload["sub"]),
            user_id=str(payload["user_id"]),
            tenant_id=str(payload["tenant_id"]),
            role=str(payload["role"]),
        )
    except HTTPException:
        raise
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def authenticate_demo_user(username: str, password: str) -> CurrentUser | None:
    user = get_demo_users().get(username)
    if not user or user.get("password") != password:
        return None
    return CurrentUser(
        sub=username,
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role=user["role"],
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_access_token(credentials.credentials)


def require_roles(*roles: str):
    def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
