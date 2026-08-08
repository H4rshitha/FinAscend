"""ORM models.

Every model is imported here so that `Base.metadata` is fully populated by the
time Alembic autogenerate or `create_all` inspects it. A model that is only
imported by the module that uses it is invisible to migrations, which shows up
as a table that silently never gets created.
"""

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey, new_id, utcnow
from app.models.organization import Organization
from app.models.user import User

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKey",
    "Organization",
    "User",
    "new_id",
    "utcnow",
]
