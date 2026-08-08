"""Password hashing.

ARGON2ID, NOT BCRYPT OR SHA
---------------------------
Argon2id is the current OWASP first choice for new applications. It is
memory-hard, so the GPU and ASIC parallelism that makes bcrypt-cracking cheap
buys an attacker far less — the bottleneck becomes RAM per guess rather than
raw hash throughput.

`argon2-cffi` is used directly rather than through passlib. passlib's bcrypt
backend is well known for breaking against bcrypt 4.x (it reads a private
`__about__` attribute that was removed), and passlib itself is unmaintained;
depending on it here would put an unmaintained shim in the authentication path
for no benefit.

The library handles the two things hand-rolled hashing gets wrong: the salt is
generated per password and embedded in the digest, and `verify` compares in
constant time. There is no separate salt column by design — the encoded digest
already contains the algorithm, version, parameters and salt, which is what
makes `needs_rehash` below possible.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

# Defaults are argon2-cffi's current recommended profile. Left explicit so a
# future tuning change is a visible diff rather than a silent library upgrade.
_hasher = PasswordHasher()

# Minimum length only. No composition rules (upper/digit/symbol): they push
# people toward "Password1!" and measurably reduce entropy, and NIST 800-63B
# recommends against them. Length plus a breach check is what actually helps.
MIN_PASSWORD_LENGTH = 10


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verification. Returns False rather than raising, so a
    malformed stored hash cannot 500 the login route and thereby distinguish
    itself from a wrong password."""
    try:
        return _hasher.verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError, Exception):
        return False


def needs_rehash(hashed: str) -> bool:
    """True when the digest was made with weaker parameters than current
    policy. Login is the only moment the plaintext is available, so it is the
    only moment a transparent upgrade is possible."""
    try:
        return _hasher.check_needs_rehash(hashed)
    except Exception:
        return False


def validate_password_strength(plain: str) -> str | None:
    """Return a human-readable problem, or None if acceptable.

    Returns the message rather than raising so the caller decides the status
    code, and so the same rule can be reused for a future 'change password'
    route without duplicating the wording.
    """
    if len(plain) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if plain.lower() in _COMMON:
        return "That password is too common — please pick something else."
    return None


# A deliberately tiny deny-list of the passwords that dominate every credential
# dump. Not a substitute for a real breach corpus (Have I Been Pwned's range
# API is the honest upgrade); it catches the worst cases at zero dependency
# cost, and it is better to state that limit than to imply full coverage.
_COMMON = {
    "password", "password1", "password123", "12345678", "123456789",
    "1234567890", "qwertyuiop", "letmein123", "welcome123", "admin12345",
    "iloveyou1", "finascend", "changeme1", "passw0rd1",
}
