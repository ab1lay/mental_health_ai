import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional


def _pbkdf2(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = _pbkdf2(password, salt)
    return base64.b64encode(salt).decode('ascii') + '$' + base64.b64encode(digest).decode('ascii')


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, digest_b64 = stored.split('$', 1)
        salt = base64.b64decode(salt_b64.encode('ascii'))
        expected = base64.b64decode(digest_b64.encode('ascii'))
        actual = _pbkdf2(password, salt)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _sign(data: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode('utf-8'), data, hashlib.sha256).digest()


def encode_session(payload: dict, secret: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    body = payload.copy()
    body['exp'] = now + ttl_seconds
    raw = json.dumps(body, separators=(',', ':')).encode('utf-8')
    sig = _sign(raw, secret)
    token = base64.urlsafe_b64encode(raw + b'.' + sig).decode('ascii')
    return token


def decode_session(token: str, secret: str) -> Optional[dict]:
    try:
        raw = base64.urlsafe_b64decode(token.encode('ascii'))
        body_raw, sig = raw.rsplit(b'.', 1)
        expected = _sign(body_raw, secret)
        if not hmac.compare_digest(sig, expected):
            return None
        body = json.loads(body_raw.decode('utf-8'))
        if int(body.get('exp', 0)) < int(time.time()):
            return None
        return body
    except Exception:
        return None
