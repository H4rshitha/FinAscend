"""Organization — the tenant that holds a plan and owns the financial data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Enum as SAEnum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.entitlements import CompanySize, Plan
from app.models.base import Base, TimestampMixin, UUIDPrimaryKey

if TYPE_CHECKING:
    from app.models.user import User


class Organization(UUIDPrimaryKey, TimestampMixin, Base):
    """A business using FinAscend.

    Users belong to an organisation rather than owning data directly, even
    though today most organisations have exactly one user. Retrofitting a
    tenant boundary after rows already reference a user id is a migration that
    touches every table, so the indirection is cheap now and expensive later —
    and the product already assumes it, because the owner/accountant split in
    `UserRole` only means anything if two people can share one set of books.
    """

    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Self-declared at signup. Stored separately from `plan` because size is a
    # fact about the business and plan is a commercial state; a business that
    # upgrades does not thereby employ more people.
    company_size: Mapped[CompanySize] = mapped_column(
        SAEnum(CompanySize, native_enum=False, length=16), nullable=False
    )
    plan: Mapped[Plan] = mapped_column(
        SAEnum(Plan, native_enum=False, length=16), nullable=False
    )

    industry: Mapped[str | None] = mapped_column(String(120), nullable=True)
    country: Mapped[str | None] = mapped_column(String(2), nullable=True)

    users: Mapped[list["User"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Organization {self.name!r} size={self.company_size} plan={self.plan}>"
