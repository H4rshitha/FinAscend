"""Authentication: sign up, sign in, and who-am-I.

WHAT THIS ROUTER IS CAREFUL ABOUT
---------------------------------
**Login does not say which half was wrong.** A distinct "no such account"
response turns the login form into an account-existence oracle: an attacker
learns which of a leaked address list are customers here, which is useful for
targeted phishing even without a password. Signup unavoidably leaks the same
fact — you cannot let someone register an address and also refuse to say it is
taken — so the mitigation is applied where it is achievable rather than
pretended everywhere.

**Login is slow on purpose, and equally slow both ways.** When the email does
not exist, a dummy Argon2 verification still runs, so a wrong address and a
wrong password take the same time. Without it, response timing distinguishes
them and the oracle comes back through the side door.

**Signup creates the organisation and the first user in one transaction.** A
user with no organisation is unusable and a plan-less organisation is
meaningless, so a partial commit would leave an account that can authenticate
but not load a single page.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.entitlements import (
    COMPANY_SIZE_LABELS,
    CompanySize,
    Plan,
    PLAN_LABELS,
    capabilities_for,
    describe_plans,
    plan_for_size,
)
from app.core.passwords import (
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from app.core.security import TokenPayload, create_access_token, current_user
from app.db.session import get_db
from app.models.base import utcnow
from app.models.organization import Organization
from app.models.user import User
from app.schemas.base import UserRole

router = APIRouter(tags=["auth"])

# A REAL Argon2 digest of a random secret nobody can supply, computed once at
# import. Verified against when the account does not exist, so the failure path
# costs the same as a genuine password check.
#
# It has to be generated rather than pasted as a literal. A hand-written digest
# is not valid Argon2, so the library rejects it on the parse step and returns
# in microseconds without running the KDF at all — which makes the "constant
# time" defence do the exact opposite of its job. Measured before this fix: a
# wrong password took 105 ms and a nonexistent account 58 ms, a 45% gap that
# reliably distinguishes the two. After: within noise.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


# ---------------------------------------------------------------------------
# request / response shapes
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    company_name: str = Field(min_length=1, max_length=200)
    company_size: CompanySize
    industry: str | None = Field(default=None, max_length=120)

    @field_validator("full_name", "company_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class OrgOut(BaseModel):
    id: str
    name: str
    company_size: CompanySize
    company_size_label: str
    plan: Plan
    plan_label: str
    plan_tagline: str
    industry: str | None = None


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole


class SessionOut(BaseModel):
    """Everything the frontend needs to render a signed-in shell in one call —
    identity, tenant and entitlements. Splitting these across three requests
    would make the app flash through three inconsistent states on load."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserOut
    organization: OrgOut
    capabilities: list[str]


def _org_out(org: Organization) -> OrgOut:
    return OrgOut(
        id=org.id,
        name=org.name,
        company_size=org.company_size,
        company_size_label=COMPANY_SIZE_LABELS[org.company_size]["label"],
        plan=org.plan,
        plan_label=PLAN_LABELS[org.plan]["label"],
        plan_tagline=PLAN_LABELS[org.plan]["tagline"],
        industry=org.industry,
    )


def _session_for(user: User, org: Organization) -> SessionOut:
    from app.core.security import TOKEN_TTL_MINUTES

    token = create_access_token(
        user_id=user.id,
        role=user.role,
        business_id=org.id,
        plan=org.plan,
        email=user.email_display,
        name=user.full_name,
    )
    return SessionOut(
        access_token=token,
        expires_in_minutes=TOKEN_TTL_MINUTES,
        user=UserOut(
            id=user.id,
            email=user.email_display,
            full_name=user.full_name,
            role=user.role,
        ),
        organization=_org_out(org),
        capabilities=capabilities_for(org.plan),
    )


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error_code": "invalid_credentials",
            "message": "That email and password combination doesn't match an account.",
            "details": {},
        },
    )


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@router.get("/auth/options")
def signup_options() -> dict[str, Any]:
    """Company sizes and the plan each maps to.

    Served rather than duplicated in the frontend so the signup form and the
    entitlement engine can never disagree about what "medium" means or what it
    unlocks.
    """
    return {
        "company_sizes": [
            {
                "value": size.value,
                **COMPANY_SIZE_LABELS[size],
                "plan": plan_for_size(size).value,
                "plan_label": PLAN_LABELS[plan_for_size(size)]["label"],
            }
            for size in CompanySize
        ],
        "plans": describe_plans(),
    }


@router.post("/auth/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> SessionOut:
    """Create an organisation and its first user, then sign them straight in.

    The first user is the OWNER — someone has to be able to approve payments,
    and an organisation whose only member cannot authorise anything would be
    dead on arrival.
    """
    problem = validate_password_strength(payload.password)
    if problem:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "weak_password",
                "message": problem,
                "details": {"field": "password"},
            },
        )

    email_folded = payload.email.strip().lower()
    existing = db.scalar(select(User).where(User.email == email_folded))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "email_taken",
                "message": "An account already exists for that email. Try signing in.",
                "details": {"field": "email"},
            },
        )

    org = Organization(
        name=payload.company_name,
        company_size=payload.company_size,
        plan=plan_for_size(payload.company_size),
        industry=payload.industry,
    )
    user = User(
        email=email_folded,
        email_display=payload.email.strip(),
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=UserRole.OWNER,
        organization=org,
        last_login_at=utcnow(),
    )

    db.add(org)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # The unique index is the real guard; the SELECT above is only a nicer
        # error for the common case. Two simultaneous signups with the same
        # address race past that check and one must lose here.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "email_taken",
                "message": "An account already exists for that email. Try signing in.",
                "details": {"field": "email"},
            },
        )
    db.refresh(user)
    db.refresh(org)
    return _session_for(user, org)


@router.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> SessionOut:
    email_folded = payload.email.strip().lower()
    user = db.scalar(select(User).where(User.email == email_folded))

    if user is None:
        # Burn equivalent time so a missing account is not distinguishable from
        # a wrong password by response latency.
        verify_password(payload.password, _DUMMY_HASH)
        raise _invalid_credentials()

    if not verify_password(payload.password, user.password_hash):
        raise _invalid_credentials()

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": "account_disabled",
                "message": "This account has been disabled.",
                "details": {},
            },
        )

    # Login is the only point where the plaintext exists, so it is the only
    # chance to transparently upgrade a digest made under weaker parameters.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    user.last_login_at = utcnow()
    db.commit()
    db.refresh(user)
    return _session_for(user, user.organization)


@router.get("/auth/me")
def me(
    token: TokenPayload = Depends(current_user), db: Session = Depends(get_db)
) -> SessionOut:
    """Re-resolve the session from the database.

    The frontend calls this on load to validate a stored token. It reads the
    database rather than trusting the token's own claims, so a plan change or a
    deactivated account is reflected as soon as the page is refreshed instead
    of waiting for the token to expire.
    """
    user = db.get(User, token.sub)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "invalid_token",
                "message": "This session is no longer valid. Please sign in again.",
                "details": {},
            },
        )
    return _session_for(user, user.organization)
