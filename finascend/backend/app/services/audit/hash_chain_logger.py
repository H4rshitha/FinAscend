"""Section B — tamper-evident hash-chain audit log.

Each entry commits to the previous entry's hash, so altering any historical
record invalidates every hash after it. This is the same construction as a
blockchain's chain-of-hashes, without any distributed consensus — which is the
right trade here, since the threat model is "an insider quietly edits a past
decision", not "mutually distrusting parties agree on an ordering".

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
It proves **tamper-evidence**: a modified past entry is detectable, because
recomputing the chain no longer reproduces the stored hashes.

It does NOT prove tamper-*resistance*. Anyone who can rewrite the whole table
can also recompute every subsequent hash and produce a consistent chain. Real
protection requires the head hash to be published somewhere the attacker does
not control — countersigned by an external timestamping service, or simply
mailed to the accountant daily. That limitation is stated because a hash chain
is frequently oversold as making records "immutable", which it does not.

Canonical serialization matters: the same logical entry must always produce
the same bytes, or verification fails for reasons that have nothing to do with
tampering. `json.dumps(..., sort_keys=True, separators=...)` pins that down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Optional

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditLogEntry:
    """One link in the chain."""

    sequence: int
    timestamp: str
    actor_id: str
    action: str
    entity_type: str
    entity_id: str
    payload: dict[str, Any]
    previous_hash: str
    entry_hash: str


def canonical_bytes(
    *,
    sequence: int,
    timestamp: str,
    actor_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, Any],
    previous_hash: str,
) -> bytes:
    """Deterministic serialization of an entry's contents.

    `sort_keys=True` and fixed separators guarantee that two logically
    identical entries hash identically regardless of dict insertion order or
    the Python version's formatting defaults. Without this, verification would
    fail spuriously and the failure would look exactly like tampering.
    """
    return json.dumps(
        {
            "sequence": sequence,
            "timestamp": timestamp,
            "actor_id": actor_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": payload,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


@dataclass
class HashChainLog:
    """An append-only, tamper-evident log.

    In-memory here; the architecture plan places it in PostgreSQL. The chain
    logic is storage-agnostic on purpose, so the same `verify` runs against
    either.
    """

    entries: list[AuditLogEntry] = field(default_factory=list)

    @property
    def head_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS_HASH

    def append(
        self,
        *,
        actor_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> AuditLogEntry:
        """Append an entry committing to the current head."""
        seq = len(self.entries)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        prev = self.head_hash
        body = canonical_bytes(
            sequence=seq,
            timestamp=ts,
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload or {},
            previous_hash=prev,
        )
        entry = AuditLogEntry(
            sequence=seq,
            timestamp=ts,
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload or {},
            previous_hash=prev,
            entry_hash=sha256(body).hexdigest(),
        )
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, Optional[int], str]:
        """Recompute the whole chain and report the FIRST broken link.

        Returning the first break rather than a bare boolean is deliberate: it
        localizes the tampering to a sequence number, and because every
        subsequent hash also breaks, only the earliest one identifies where the
        edit actually happened.

        Returns:
            (is_valid, first_bad_sequence_or_None, human_readable_message)
        """
        prev = GENESIS_HASH
        for e in self.entries:
            if e.previous_hash != prev:
                return (
                    False,
                    e.sequence,
                    f"Entry {e.sequence} does not link to the previous entry: "
                    f"stored previous_hash {e.previous_hash[:12]}... but the "
                    f"actual preceding hash is {prev[:12]}...",
                )
            recomputed = sha256(
                canonical_bytes(
                    sequence=e.sequence,
                    timestamp=e.timestamp,
                    actor_id=e.actor_id,
                    action=e.action,
                    entity_type=e.entity_type,
                    entity_id=e.entity_id,
                    payload=e.payload,
                    previous_hash=e.previous_hash,
                )
            ).hexdigest()
            if recomputed != e.entry_hash:
                return (
                    False,
                    e.sequence,
                    f"Entry {e.sequence} has been modified: its contents hash "
                    f"to {recomputed[:12]}... but {e.entry_hash[:12]}... is stored.",
                )
            prev = e.entry_hash
        return (True, None, f"Chain intact: {len(self.entries)} entries verified.")

    def trail_for(self, entity_type: str, entity_id: str) -> list[AuditLogEntry]:
        """Every entry touching one entity, in order."""
        return [
            e
            for e in self.entries
            if e.entity_type == entity_type and e.entity_id == entity_id
        ]
