"""Tests de modules.scan : unitaires (numpy pur) + intégration réelle ffmpeg/librosa/OpenCV.

La transcription est injectée (scan.transcript_path) : faster-whisper n'est pas exécuté en CI.
"""

import json
import shutil
import subprocess
import sys
from typing import ClassVar

import numpy as np
import pytest

from core.scoring import FEATURES
from modules import scan as M

needs_media = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")


# ---------------------------------------------------------------- importabilité
def test_import_does_not_load_heavy_deps():
    code = ("import sys, modules.scan, modules.render; "
            "heavy = {'cv2','librosa','faster_whisper','yt_dlp','torch'} & set(sys.modules); "
            "print(sorted(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------- unitaires
def _transcript():
    seg = lambda s, e, text: {"start": s, "end": e, "text": text, "words": [
        {"start": s + i * (e - s) / len(text.split()), "end": s + (i + 1) * (e - s) / len(text.split()), "word": w}
        for i, w in enumerate(text.split())]}
    segs = [seg(t, t + 4, "okay we drive to the store now.") for t in range(0, 56, 5)]
    segs.append(seg(56.5, 60.0, "watch me jump this"))                    # phrase inachevée
    segs.append(seg(60.5, 64.0, "OH MY GOD no way no no no!"))
    segs += [seg(t, t + 4, "alright that was fun anyway.") for t in range(66, 116, 5)]
    return {"language": "en", "model": "test", "segments": segs}


def test_text_signals_detects_hype_repetition_cutoffs():
    ts = M.text_signals(_transcript(), 120, ["no way", "oh my god", "what"])
    assert ts.hype[60:64].sum() > 3 and ts.hype[:55].sum() == 0
    assert ts.repetition[60:65].sum() >= 2
    assert 60.0 in ts.cutoffs
    assert {k for _, k in ts.matches} == {"no way", "oh my god"}


def test_refine_bounds_respects_clip_max_and_words():
    words = M.words_from_transcript(_transcript())
    a, b = M.refine_bounds(40, 85, 61.5, words, clip_min=8, clip_max=29.5, duration=120)
    assert b - a <= 29.5 and a <= 61.5 <= b
    assert not any(w["start"] < b < w["end"] for w in words)            # pas de mot coupé


def test_window_features_and_selection_without_overlap():
    n = 300
    sig = {f: np.zeros(n) for f in FEATURES}
    sig["audio_loudness_peak"][100:105] = 10
    sig["audio_loudness_peak"][220:224] = 8
    tl = M.Timeline(n, sig, {"audio_loudness_peak"}, [], [])
    starts, lens, X = M.window_features(tl, [15, 30, 45], 5)
    assert X.shape == (len(starts), len(FEATURES)) and X.max() <= 1
    scores = X[:, FEATURES.index("audio_loudness_peak")] * 100
    picked = M.select_moments(starts, lens, scores, tl, {"audio_loudness_peak": 1.0}, count=3, min_gap=30,
                              clip_min=8, clip_max=29.5)
    assert len(picked) >= 2
    for i, p in enumerate(picked):
        for q in picked[i + 1:]:
            assert p["end"] + 30 <= q["start"] or q["end"] + 30 <= p["start"]
    assert picked[0]["start"] <= 102 <= picked[0]["end"]


def test_heatmap_timeline():
    tl = M.heatmap_timeline([{"start_time": 10, "end_time": 20, "value": 1.0}], 30)
    assert tl[10:20].min() == 1.0 and tl[:10].max() == 0 and M.heatmap_timeline(None, 10) is None
    assert M.is_youtube_url("https://www.youtube.com/watch?v=x") and not M.is_youtube_url("https://twitch.tv/v/1")


def test_choose_whisper_model_cpu_fallback(monkeypatch):
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 0)
    assert M.choose_whisper_model({"model": "auto", "device": "cuda"}) == ("small", "cpu", "int8")
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(M, "cuda_vram_gb", lambda: 15.0)
    assert M.choose_whisper_model({"model": "auto"})[0] == "large-v3"
    monkeypatch.setattr(M, "cuda_vram_gb", lambda: 8.0)
    assert M.choose_whisper_model({"model": "auto"})[0] == "medium"


# ---------------------------------------------------------------- intégration
@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """120 s : plan fixe calme, sauf 60-70 s (mire animée) et 61-66 s (son fort)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg requis")
    out = tmp_path_factory.mktemp("media") / "gta_test_stream.mp4"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=15:d=120",
           "-f", "lavfi", "-i", "testsrc2=s=320x180:r=15:d=10",
           "-f", "lavfi", "-i", "aevalsrc='if(between(t,61,66),0.8,0.01)*sin(2*PI*440*t)':d=120:s=16000",
           "-filter_complex", "[1:v]setpts=PTS+60/TB[m];[0:v][m]overlay=enable='between(t,60,70)'[v]",
           "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
           str(out)]
    subprocess.run(cmd, check=True)
    return out


def _cfg(project, tmp_path, **scan_overrides):
    from core.config import load_config

    cfg = load_config(project / "config.yaml")
    tpath = tmp_path / "transcript.json"
    tpath.write_text(json.dumps(_transcript()))
    cfg["scan"].update({"transcript_path": str(tpath), "work_dir": str(tmp_path / "work"),
                        "tmp_dir": str(tmp_path / "tmp"), **scan_overrides})
    cfg["job"] = {"video_id": "twitch_123", "streamer": "kaicenat", "job_id": "j1"}
    return cfg


@needs_media
def test_scan_end_to_end(project, tmp_path, synthetic_video):
    cfg = _cfg(project, tmp_path)
    moments = M.scan(str(synthetic_video), str(tmp_path / "moments"), cfg)

    assert 1 <= len(moments) <= 3 and all(m["score"] >= 40 for m in moments)
    best = moments[0]
    assert best["start"] <= 62 and best["end"] >= 64, best            # le moment fort est trouvé
    assert best["duration"] < 30 and 0 <= best["score"] <= 100
    assert len(best["hooks"]) == 3 and all(len(h) <= 60 for h in best["hooks"])
    assert "no way" in best["keywords"] and best["words"]
    assert "\nOH MY GOD" in best["transcript"]                       # phrases séparées aux silences
    assert set(best["signals"]) == set(FEATURES)
    assert [m["score"] for m in moments] == sorted((m["score"] for m in moments), reverse=True)

    doc = json.loads((tmp_path / "moments" / "twitch_123.json").read_text())
    assert doc["format"] == M.SCAN_FORMAT and doc["scorer"]["name"] == "weighted"
    assert {"audio_loudness_peak", "visual_motion", "text_hype_keywords"} <= set(doc["signals_available"])
    assert "most_replayed" not in doc["signals_available"]
    assert all(not m["hooks"] for m in doc["candidates"])
    assert (tmp_path / "moments" / "twitch_123.transcript.json").is_file()
    assert not (tmp_path / "tmp" / "twitch_123").exists()            # WAV temporaire nettoyé


@needs_media
def test_rescan_reuses_checkpoints_and_picks_up_model(project, tmp_path, synthetic_video, monkeypatch):
    import joblib

    from tests.test_scoring import _bundle

    cfg = _cfg(project, tmp_path)
    first = M.scan(str(synthetic_video), str(tmp_path / "moments"), cfg)

    def boom(*a, **k):
        raise AssertionError("ne doit pas être recalculé")

    monkeypatch.setattr(M, "visual_signals", boom)
    monkeypatch.setattr(M, "audio_loudness", boom)
    joblib.dump(_bundle(), tmp_path / "model.pkl")
    cfg["scan"]["model_path"] = str(tmp_path / "model.pkl")
    second = M.scan(str(synthetic_video), str(tmp_path / "moments"), cfg)
    doc = json.loads((tmp_path / "moments" / "twitch_123.json").read_text())
    assert doc["scorer"]["name"] == "model" and doc["scorer"]["n_samples"] == 60
    assert len(second) == len(first)


@needs_media
def test_scan_survives_visual_failure(project, tmp_path, synthetic_video, monkeypatch):
    def broken(*a, **k):
        raise M.ScanError("decoder crash")

    monkeypatch.setattr(M, "visual_signals", broken)
    moments = M.scan(str(synthetic_video), str(tmp_path / "m"), _cfg(project, tmp_path))
    assert moments and moments[0]["signals"]["visual_motion"] == 0


def test_scan_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.scan(str(tmp_path / "nope.mp4"), str(tmp_path), {"scan": {}})


# ---------------------------------------------------------------- whisper (faux modèle, pas de réseau)
class _W:
    def __init__(self, s, e, w, p=0.9):
        self.start, self.end, self.word, self.probability = s, e, w, p


class _Seg:
    def __init__(self, s, e, text, words):
        self.start, self.end, self.text, self.words = s, e, text, words


def _tiny_wav(path, seconds=1.0, rate=16000, channels=1):
    import wave

    n = int(seconds * rate)
    samples = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)) * 12000).astype("<i2")
    if channels == 2:
        samples = np.repeat(samples, 2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return str(path)


def test_load_wav_16k(tmp_path):
    audio = M.load_wav_16k(_tiny_wav(tmp_path / "a.wav", seconds=0.5))
    assert audio.dtype == np.float32 and audio.shape == (8000,) and 0.3 < np.abs(audio).max() <= 1.0
    assert M.load_wav_16k(_tiny_wav(tmp_path / "s.wav", channels=2)).shape == (16000,)
    with pytest.raises(M.ScanError, match="16 000 Hz"):
        M.load_wav_16k(_tiny_wav(tmp_path / "b.wav", rate=44100))


@needs_media
def test_load_wav_from_real_extraction(tmp_path, synthetic_video):
    wav = M.extract_audio(str(synthetic_video), tmp_path / "x.wav")
    audio = M.load_wav_16k(str(wav))
    assert len(audio) == pytest.approx(120 * 16000, rel=0.01)


def test_transcribe_format_and_oom_fallback(monkeypatch, tmp_path):
    import faster_whisper

    wav = _tiny_wav(tmp_path / "a.wav")

    loaded = []

    class FakeModel:
        def __init__(self, name, device, compute_type):
            loaded.append((name, compute_type))
            if name == "large-v3":
                raise RuntimeError("CUDA failed with error out of memory")

        def transcribe(self, audio, language=None, beam_size=5, vad_filter=True, word_timestamps=False,
                       condition_on_previous_text=True):
            # Régression Colab : un chemin ferait décoder l'audio par PyAV (incompatible) -> tableau numpy requis.
            assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
            assert word_timestamps is True and condition_on_previous_text is False
            info = type("I", (), {"language": "en", "language_probability": 0.98, "duration": 10.0})()
            return iter([_Seg(0.0, 2.0, " No way! ", [_W(0.0, 0.8, " No"), _W(0.8, 1.6, " way!")])]), info

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(M, "cuda_vram_gb", lambda: 15.0)
    result = M.transcribe(wav, {"model": "auto", "batch_size": 0, "fallback_models": ["medium", "small"]})
    assert loaded == [("large-v3", "float16"), ("medium", "int8_float16")]
    assert result["model"] == "medium" and result["language"] == "en"
    assert result["segments"] == [{"start": 0.0, "end": 2.0, "text": "No way!", "words": [
        {"start": 0.0, "end": 0.8, "word": "No", "prob": 0.9}, {"start": 0.8, "end": 1.6, "word": "way!", "prob": 0.9}]}]


def test_transcribe_non_oom_error_propagates(monkeypatch, tmp_path):
    import faster_whisper

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("model file corrupted")

    monkeypatch.setattr(faster_whisper, "WhisperModel", Broken)
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="corrupted"):
        M.transcribe(_tiny_wav(tmp_path / "a.wav"), {"model": "small"})


class ChunkModel:
    """Faux whisper : un segment par tronçon, à 1 s du début du tronçon ; peut échouer au n-ième appel."""

    calls: ClassVar[list] = []
    fail_at: int | None = None
    oom_names: ClassVar[tuple] = ()

    def __init__(self, name, device, compute_type):
        self.name = name
        if name in ChunkModel.oom_names:
            raise RuntimeError("CUDA failed with error out of memory")

    def transcribe(self, audio, language=None, **kw):
        assert isinstance(audio, np.ndarray) and len(audio) <= 60 * 16000 + 1   # jamais plus d'un tronçon en RAM
        ChunkModel.calls.append((self.name, len(audio), language))
        if ChunkModel.fail_at is not None and len(ChunkModel.calls) == ChunkModel.fail_at:
            raise KeyboardInterrupt("session coupée")
        info = type("I", (), {"language": "fr", "language_probability": 0.9, "duration": len(audio) / 16000})()
        return iter([_Seg(1.0, 2.0, "salut", [_W(1.0, 1.5, "salut")])]), info


def test_transcribe_in_chunks_resumes_after_interruption(monkeypatch, tmp_path):
    import faster_whisper

    wav = _tiny_wav(tmp_path / "long.wav", seconds=150)          # 150 s -> 3 tronçons de 60 s
    monkeypatch.setattr(faster_whisper, "WhisperModel", ChunkModel)
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 0)
    cfg = {"model": "small", "chunk_minutes": 1, "batch_size": 0}
    ChunkModel.calls, ChunkModel.fail_at, ChunkModel.oom_names = [], 2, ()
    with pytest.raises(KeyboardInterrupt):                         # coupure pendant le 2e tronçon
        M.transcribe(wav, cfg, checkpoint_dir=tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "transcript_chunks" / "chunk_000.json").is_file()

    ChunkModel.calls, ChunkModel.fail_at = [], None
    res = M.transcribe(wav, cfg, checkpoint_dir=tmp_path / "ckpt")
    assert len(ChunkModel.calls) == 2                              # le tronçon 0 n'est pas refait
    assert all(lang == "fr" for _, _, lang in ChunkModel.calls)    # langue du 1er tronçon réutilisée
    assert [s["start"] for s in res["segments"]] == [1.0, 61.0, 121.0]   # horodatages décalés
    assert res["segments"][2]["words"][0]["start"] == 121.0 and res["chunks"] == 3 and res["duration"] == 150


def test_transcribe_chunks_oom_fallback(monkeypatch, tmp_path):
    import faster_whisper

    wav = _tiny_wav(tmp_path / "long.wav", seconds=90)
    monkeypatch.setattr(faster_whisper, "WhisperModel", ChunkModel)
    monkeypatch.setattr(M, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(M, "cuda_vram_gb", lambda: 15.0)
    ChunkModel.calls, ChunkModel.fail_at, ChunkModel.oom_names = [], None, ("large-v3",)
    res = M.transcribe(wav, {"model": "auto", "chunk_minutes": 1, "batch_size": 0})
    assert res["model"] == "medium" and len(res["segments"]) == 2
