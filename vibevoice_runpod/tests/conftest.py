import os
import pytest
from helpers import base64_audio, make_audio_bytes


@pytest.fixture(autouse=True)
def env_vars():
    os.environ.setdefault("ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    os.environ.setdefault("LOCAL_MODEL_PATH", "/mock/model")
    yield