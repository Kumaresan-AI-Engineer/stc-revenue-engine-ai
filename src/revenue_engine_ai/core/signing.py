"""HMAC-SHA256 signatures for job callbacks, the backend's format (its core/signing.py verifies them).

The header reads `t=<unix seconds>,v1=<hex digest>`; the digest covers `<t>.<raw body>`. Pure: no I/O.
"""

import hashlib
import hmac
import time

SIGNATURE_HEADER = "X-RE-Signature"


def sign(secret: str, body: bytes, timestamp: int | None = None) -> str:
    signed_at = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(secret.encode(), f"{signed_at}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={signed_at},v1={digest}"
