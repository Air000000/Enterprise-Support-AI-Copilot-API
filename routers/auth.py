from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import CurrentUser, authenticate_demo_user, create_access_token, get_current_user


router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    sub: str
    user_id: str
    tenant_id: str
    role: str


@router.post("/token", response_model=TokenResponse)
def create_token(request: TokenRequest) -> TokenResponse:
    user = authenticate_demo_user(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return TokenResponse(access_token=create_access_token(user))


@router.get("/me", response_model=MeResponse)
def read_me(user: CurrentUser = Depends(get_current_user)) -> MeResponse:
    return MeResponse(
        sub=user.sub,
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        role=user.role,
    )
