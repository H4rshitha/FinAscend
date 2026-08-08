"""JWT auth and RBAC.

Three roles, least-privilege by default:
  owner       — full access, the only role that may approve payments
  accountant  — may read everything and prepare plans, but NOT approve
  viewer      — read-only

The accountant/owner split is the point. The architecture plan flags it as a
gap beyond the PPT: small businesses commonly have a bookkeeper who prepares
payments but must not authorize them, and collapsing that into one role
removes the separation of duties that makes the review queue meaningful.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.schemas.base import UserRole

# Dev default only. A missing secret in production must fail loudly rather
# than silently signing tokens with a known key, so this is checked at startup.
JWT_SECRET = os.environ.get("FINASCEND_JWT_SECRET", "dev-only-insecure-secret")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 60

_bearer = HTTPBearer(auto_error=False)


class TokenPayload(BaseModel):
    sub: str
    role: UserRole
    business_id: str
    exp: Optional[int] = None


def create_access_token(
    *, user_id: str, role: UserRole, business_id: str, ttl_minutes: int = TOKEN_TTL_MINUTES
) -> str:
    """Issue a signed JWT."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    claims = {
        "sub": user_id,
        "role": role.value,
        "business_id": business_id,
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        raw = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**raw)
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "invalid_token",
                "message": "Authentication token is missing, malformed or expired.",
                "details": {"reason": str(exc)},
            },
        ) from exc


async def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> TokenPayload:
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "not_authenticated",
                "message": "This endpoint requires a bearer token.",
                "details": {},
            },
        )
    return decode_token(creds.credentials)


def require_role(*allowed: UserRole):
    """Dependency factory enforcing role membership.

    Written as a factory so the allowed set is visible in the route
    definition itself — `Depends(require_role(UserRole.OWNER))` states the
    access rule at the point of use rather than hiding it in a table
    somewhere else.
    """

    async def _check(user: TokenPayload = Depends(current_user)) -> TokenPayload:
        if user.role not in {r.value for r in allowed} and user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "insufficient_role",
                    "message": (
                        f"Role '{user.role}' cannot perform this action. "
                        f"Required: {', '.join(r.value for r in allowed)}."
                    ),
                    "details": {"your_role": user.role},
                },
            )
        return user

    return _check
