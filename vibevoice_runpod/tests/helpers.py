import io
import base64
import numpy as np
import soundfile as sf


def make_audio_bytes(sample_rate=24000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def base64_audio(sample_rate=24000, duration=0.5):
    return base64.b64encode(make_audio_bytes(sample_rate, duration)).decode("utf-8")