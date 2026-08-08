"""Database engine and session management.

WHY SQLITE, AND WHAT MAKES THE SWITCH CHEAP
-------------------------------------------
The data here is inherently relational — a user belongs to an organisation,
an organisation holds one plan, a plan grants a set of entitlements — so a
document store would be modelling joins by hand. That narrows it to a SQL
engine, and then to SQLite vs PostgreSQL.

PostgreSQL is the right *production* answer and this module does not argue
otherwise. It is not the right answer for *this* task today: every other part
of this project starts from a single virtualenv with no server process and no
container, and requiring a running database daemon before a user can sign up
would be the heaviest dependency in the repository. SQLite ships inside Python,
so `signup` works on a fresh clone with no setup at all.

What keeps that from being a trap is that nothing above this file knows which
engine it is talking to. Models use plain SQLAlchemy 2.0 types, the connection
is chosen by `DATABASE_URL`, and moving to Postgres is:

    pip install "psycopg[binary]"
    set FINASCEND_DATABASE_URL=postgresql+psycopg://user:pw@host/finascend
    alembic upgrade head

No model, route or query changes. The two engine-specific details are both
isolated here: the `check_same_thread` connect arg, and the `PRAGMA
foreign_keys=ON` pragma below.

TWO SQLITE FOOTGUNS, BOTH HANDLED
---------------------------------
1. **Foreign keys are OFF by default.** SQLite silently ignores foreign-key
   constraints unless the pragma is set per connection. Without it a user row
   could reference a deleted organisation and nothing would complain, which is
   precisely the class of corruption the constraint exists to prevent.

2. **The file must not live in a cloud-synced folder.** This project sits under
   OneDrive, and a sync client copying a database mid-write is a documented way
   to corrupt SQLite — it does not coordinate with the file locks. The default
   path therefore resolves to a `.data/` directory that is gitignored, and the
   location is overridable so it can be moved off the synced tree entirely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Repo-root/.data/finascend.db by default. Kept out of the source tree and out
# of git, because this file holds password hashes.
_DEFAULT_DIR = Path(__file__).resolve().parents[3] / ".data"
_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_URL = f"sqlite:///{(_DEFAULT_DIR / 'finascend.db').as_posix()}"

DATABASE_URL = os.environ.get("FINASCEND_DATABASE_URL", _DEFAULT_URL)
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine: Engine = create_engine(
    DATABASE_URL,
    # SQLite's default thread check rejects a connection reused across threads,
    # which FastAPI's threadpool does routinely. Safe to disable because each
    # request gets its own Session from the pool below.
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=True,
    future=True,
)


if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _enforce_sqlite_foreign_keys(dbapi_connection, _record) -> None:
        """Turn on foreign-key enforcement for every SQLite connection.

        SQLite defaults this OFF for backwards compatibility, which means
        `ForeignKey` declarations are decorative until this runs. Postgres
        needs no equivalent — it has always enforced them.
        """
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session.

    The session is closed in a `finally` so a failed request cannot leak a
    connection out of the pool — under SQLite that would eventually hold a
    write lock open and hang every subsequent write.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
