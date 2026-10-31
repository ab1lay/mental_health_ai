import time

from healthcare_web.app.auth import (
    decode_session,
    encode_session,
    hash_password,
    verify_password,
)

SECRET = "unit-test-secret"


def test_session_roundtrip_is_stable_for_many_payloads():
    # The HMAC signature is binary and often contains b'.'; every token must
    # still decode (previously ~12% of sessions were silently rejected).
    for user_id in range(500):
        token = encode_session({"user_id": user_id}, SECRET, ttl_seconds=60)
        data = decode_session(token, SECRET)
        assert data is not None, user_id
        assert data["user_id"] == user_id


def test_session_rejects_wrong_secret_and_tampering():
    token = encode_session({"user_id": 1}, SECRET)
    assert decode_session(token, "other-secret") is None
    forged = encode_session({"user_id": 2}, "attacker")
    assert decode_session(forged, SECRET) is None
    assert decode_session("not-a-token", SECRET) is None
    assert decode_session("", SECRET) is None


def test_session_expires():
    token = encode_session({"user_id": 1}, SECRET, ttl_seconds=-1)
    assert decode_session(token, SECRET) is None


def test_password_hashing():
    stored = hash_password("s3cret")
    assert verify_password("s3cret", stored)
    assert not verify_password("wrong", stored)
    assert not verify_password("s3cret", "garbage")
    assert hash_password("s3cret") != stored  # salted
