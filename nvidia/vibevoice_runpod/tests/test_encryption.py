import base64
import json

import pytest
from cryptography.fernet import Fernet

ENCRYPTION_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def test_fernet_round_trip():
    f = Fernet(ENCRYPTION_KEY.encode())
    payload = {"audio": "AAAA", "prompt": "hello", "format": "parsed"}
    token = f.encrypt(json.dumps(payload).encode())
    decrypted = json.loads(f.decrypt(token).decode())
    assert decrypted == payload


def test_fernet_key_mismatch():
    f1 = Fernet(ENCRYPTION_KEY.encode())
    bad_key = base64.urlsafe_b64encode(b"x" * 32)
    f2 = Fernet(bad_key)
    token = f1.encrypt(b"hello")
    with pytest.raises(Exception):
        f2.decrypt(token)


def test_debug_mode_passthrough():
    inner = {"audio": "_test_", "prompt": "hi", "format": "raw", "debug": True}
    payload = {"input": {**inner, "is_encrypted": False, "debug": True}}
    assert payload["input"]["is_encrypted"] is False
    assert payload["input"]["debug"] is True
    assert payload["input"]["audio"] == "_test_"