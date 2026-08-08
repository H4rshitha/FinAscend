"""User — a person who signs in, scoped to exactly one organisation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.schemas.base import UserRole

if TYPE_CHECKING:
    from app.models.organization import Organization


class User(UUIDPrimaryKey, TimestampMixin, Base):
    """An authenticated person.

    The email is stored **case-folded** in `email` and the form the user typed
    is kept in `email_display`. Addresses are case-insensitive in practice, so
    without folding "Anna@x.com" and "anna@x.com" become two accounts and the
    second signup succeeds confusingly instead of saying "you already have an
    account". Folding on write rather than comparing with LOWER() on read also
    keeps the unique index usable — a functional index would be needed
    otherwise, and SQLite and Postgres spell that differently.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    email_display: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Argon2id digest. Named `password_hash` rather than `password` so that a
    # stray log line or serialiser that grabs attributes by name cannot make
    # "password" appear in output and look like a plaintext field.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, native_enum=False, length=16),
        nullable=False,
        default=UserRole.OWNER,
    )

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    organization: Mapped["Organization"] = relationship(back_populates="users")

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email!r} role={self.role}>"
