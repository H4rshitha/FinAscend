"""Declarative base and shared column conventions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware UTC. `datetime.utcnow()` returns a NAIVE datetime, which
    compares incorrectly against aware ones and is deprecated in 3.12."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class UUIDPrimaryKey:
    """String UUID rather than an autoincrement integer.

    Sequential integer ids leak business volume — /organizations/42 tells a
    competitor how many customers exist and lets anyone enumerate them. They
    also collide the moment two environments merge. The cost is a wider index;
    at this scale that is irrelevant, and the ids are generated in Python so
    the column type stays identical on SQLite and Postgres.
    """

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
