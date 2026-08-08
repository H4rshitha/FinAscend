"""JWT auth, RBAC and plan entitlements.

Three roles, least-privilege by default:
  owner       — full access, the only role that may approve payments
  accountant  — may read everything and prepare plans, but NOT approve
  viewer      — read-only

The accountant/owner split is the point. The architecture plan flags it as a
gap beyond the PPT: small businesses commonly have a bookkeeper who prepares
payments but must not authorize them, and collapsing that into one role
removes the separation of duties that makes the review queue meaningful.

ROLE vs PLAN — two different questions
--------------------------------------
`require_role` asks "is this person allowed to do this?" (authorization).
`require_capability` asks "did this organisation pay for this?" (entitlement).
They are enforced separately because they fail differently: a role failure is
403 and permanent for that user, while an entitlement failure is 402 and is
fixed by upgrading. Collapsing them into one check would make the UI unable to
tell "ask your boss" apart from "upgrade your plan".

The plan travels **inside the token** so the common path needs no database
read. That is a deliberate freshness trade: a plan change does not take effect
until the token is re-issued, bounded by `TOKEN_TTL_MINUTES`. For an upgrade
that is fine (worst case an hour of the old plan); a *downgrade* that must bite
immediately would need a DB check per request, and this module would be the
place to add it.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.entitlements import Capability, Plan, capabilities_for
from app.schemas.base import UserRole

# Dev default only. `assert_secret_is_safe()` is called at startup so an
# unconfigured secret fails loudly in a non-dev environment rather than
# silently signing tokens with a key that is published in this repository.
DEV_SECRET = "dev-only-insecure-secret"
JWT_SECRET = os.environ.get("FINASCEND_JWT_SECRET", DEV_SECRET)
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 60 * 12  # a working day; sessions survive a lunch break

_bearer = HTTPBearer(auto_error=False)


def assert_secret_is_safe() -> None:
    """Refuse to boot with the published dev secret outside development.

    Anyone holding the signing key can mint a token for any user in any
    organisation. Since this key is committed in plain sight, shipping it is
    equivalent to shipping no authentication at all — so this is a hard failure
    rather than a warning that scrolls past in a log.
    """
    env = os.environ.get("FINASCEND_ENV", "development").lower()
    if env != "development" and JWT_SECRET == DEV_SECRET:
        raise RuntimeError(
            "FINASCEND_JWT_SECRET is unset outside development. The fallback "
            "value is published in this repository, so any token could be "
            "forged. Set a long random secret before starting."
        )


class TokenPayload(BaseModel):
    sub: str
    role: UserRole
    business_id: str
    # Same value as business_id. Both are carried because `business_id` is what
    # the existing quant routes read, and renaming it across them would be an
    # unrelated change riding along with authentication.
    org_id: Optional[str] = None
    plan: Optional[Plan] = None
    email: Optional[str] = None
    name: Optional[str] = None
    exp: Optional[int] = None

    @property
    def capabilities(self) -> list[str]:
        return capabilities_for(self.plan) if self.plan else []

    def can(self, capability: Capability) -> bool:
        return self.plan is not None and capability.value in self.capabilities


def create_access_token(
    *,
    user_id: str,
    role: UserRole,
    business_id: str,
    plan: Optional[Plan] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    ttl_minutes: int = TOKEN_TTL_MINUTES,
) -> str:
    """Issue a signed JWT."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    claims: dict[str, object] = {
        "sub": user_id,
        "role": role.value if hasattr(role, "value") else str(role),
        "business_id": business_id,
        "org_id": business_id,
        "exp": int(expire.timestamp()),
    }
    if plan is not None:
        claims["plan"] = plan.value if hasattr(plan, "value") else str(plan)
    if email:
        claims["email"] = email
    if name:
        claims["name"] = name
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


def require_capability(capability: Capability):
    """Dependency factory enforcing a PLAN entitlement.

    Returns 402 Payment Required rather than 403 Forbidden. The distinction is
    load-bearing for the UI: 403 means this person may not do it, and the fix
    is a conversation with an administrator; 402 means the organisation has not
    bought it, and the fix is an upgrade. A single status for both would leave
    the frontend guessing which message to show.
    """

    async def _check(user: TokenPayload = Depends(current_user)) -> TokenPayload:
        if not user.can(capability):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error_code": "upgrade_required",
                    "message": (
                        "Your plan does not include this. It is available on a "
                        "higher plan."
                    ),
                    "details": {
                        "required_capability": capability.value,
                        "your_plan": user.plan.value if user.plan else None,
                    },
                },
            )
        return user

    return _check
