import io
import base64
import numpy as np
import soundfile as sf


def make_sine_wave(sr=24000, duration=0.5, freq=440):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_base64_decode_to_wav():
    audio_np = make_sine_wave()
    buf = io.BytesIO()
    sf.write(buf, audio_np, 24000, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    decoded = base64.b64decode(b64)
    data, sr = sf.read(io.BytesIO(decoded))
    assert sr == 24000
    assert data.dtype == np.float64
    assert np.allclose(data, audio_np, atol=1e-4)


def test_stereo_to_mono():
    stereo = np.column_stack([make_sine_wave(), make_sine_wave(freq=880)])
    assert stereo.ndim == 2
    mono = stereo.mean(axis=1)
    assert mono.ndim == 1
    assert mono.shape[0] == stereo.shape[0]


def test_resample_to_24k():
    import librosa
    audio_16k = make_sine_wave(sr=16000, duration=1.0)
    resampled = librosa.resample(audio_16k, orig_sr=16000, target_sr=24000)
    expected_len = int(24000 * 1.0)
    assert abs(resampled.shape[0] - expected_len) <= 1


def test_cast_to_float32():
    audio = np.array([0, 0.5, 1.0], dtype=np.float64)
    cast = audio.astype(np.float32)
    assert cast.dtype == np.float32
    assert np.allclose(cast, audio, atol=1e-7)


def test_full_pipeline_from_b64_to_float32_mono_24k():
    import librosa

    audio_44k = make_sine_wave(sr=44100, duration=0.3)
    buf = io.BytesIO()
    sf.write(buf, audio_44k, 44100, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    raw = base64.b64decode(b64)
    data, sr = sf.read(io.BytesIO(raw))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 24000:
        data = librosa.resample(data, orig_sr=sr, target_sr=24000)
    data = data.astype(np.float32)

    assert data.dtype == np.float32
    assert data.ndim == 1
    assert abs(data.shape[0] - int(24000 * 0.3)) <= 1