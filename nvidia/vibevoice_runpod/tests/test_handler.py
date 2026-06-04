import sys
import json
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from helpers import base64_audio


@pytest.fixture
def mock_handler_deps():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = MagicMock()

    mock_processor = MagicMock()
    mock_processor.apply_transcription_request.return_value.to.return_value = {
        "input_ids": MagicMock(shape=(1, 10))
    }
    mock_processor.decode.return_value = [{"text": "hello world"}]

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock_model.generate.return_value.shape = (1, 20)

    mock_vibevoice = MagicMock()
    mock_vibevoice.from_pretrained.return_value = mock_model

    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = mock_processor

    mock_transformers = MagicMock()
    mock_transformers.AutoProcessor = mock_auto
    mock_transformers.VibeVoiceAsrForConditionalGeneration = mock_vibevoice

    with (
        patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }),
    ):
        import importlib
        import vibevoice_runpod.src.handler as h
        importlib.reload(h)

    with (
        patch.object(h, "processor", mock_processor),
        patch.object(h, "model", mock_model),
        patch.object(h, "torch", mock_torch),
    ):
        yield {"handler": h.handler, "processor": mock_processor, "model": mock_model}


def test_basic_transcription(mock_handler_deps):
    result = mock_handler_deps["handler"](
        {"input": {"audio": base64_audio()}}
    )
    assert "transcription" in result
    assert result["transcription"] == {"text": "hello world"}


def test_missing_audio_returns_error(mock_handler_deps):
    result = mock_handler_deps["handler"]({"input": {}})
    assert result["status"] == "error"
    assert "No audio" in result["message"]


def test_invalid_base64_returns_error(mock_handler_deps):
    result = mock_handler_deps["handler"]({"input": {"audio": "not-valid-base64!!!"}})
    assert result["status"] == "error"


def test_encrypted_flow_full_round_trip(mock_handler_deps):
    mock_handler_deps["processor"].decode.return_value = [{"text": "encrypted ok"}]

    cipher = Fernet(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    inner = {"audio": base64_audio(), "prompt": "test"}
    token = cipher.encrypt(json.dumps(inner).encode()).decode()

    result = mock_handler_deps["handler"](
        {"input": {"encrypted_input": token, "is_encrypted": True}}
    )
    assert result["status"] == "success"
    assert result["output"]["transcription"] == {"text": "encrypted ok"}


def test_debug_mode_passthrough(mock_handler_deps):
    result = mock_handler_deps["handler"]({
        "input": {
            "audio": base64_audio(),
            "debug": True,
            "is_encrypted": False,
        }
    })
    assert result["status"] == "success"


def test_tokenizer_chunk_size_passed_to_generate(mock_handler_deps):
    mock_handler_deps["handler"]({
        "input": {
            "audio": base64_audio(),
            "tokenizer_chunk_size": 999,
        }
    })
    mock_handler_deps["model"].generate.assert_called_once()
    _, kwargs = mock_handler_deps["model"].generate.call_args
    assert kwargs.get("tokenizer_chunk_size") == 999


def test_prompt_passed_to_apply_transcription(mock_handler_deps):
    mock_handler_deps["handler"]({
        "input": {
            "audio": base64_audio(),
            "prompt": "context words",
        }
    })
    mock_handler_deps["processor"].apply_transcription_request.assert_called_once()
    _, kwargs = mock_handler_deps["processor"].apply_transcription_request.call_args
    assert kwargs.get("prompt") == "context words"
    assert "audio" in kwargs


def test_format_parsed_default(mock_handler_deps):
    mock_handler_deps["processor"].decode.return_value = [{"text": "parsed result"}]
    mock_handler_deps["handler"](
        {"input": {"audio": base64_audio()}}
    )
    mock_handler_deps["processor"].decode.assert_called_once()
    _, kwargs = mock_handler_deps["processor"].decode.call_args
    assert kwargs.get("return_format") == "parsed"


def test_format_raw(mock_handler_deps):
    mock_handler_deps["processor"].decode.return_value = ["raw transcription"]
    mock_handler_deps["handler"]({
        "input": {
            "audio": base64_audio(),
            "format": "raw",
        }
    })
    mock_handler_deps["processor"].decode.assert_called_once()
    _, kwargs = mock_handler_deps["processor"].decode.call_args
    assert kwargs.get("return_format") == "raw"