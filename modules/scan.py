"""Module 3 — scan (CLOUD GPU) : trouve les moments les plus viraux d'une vidéo.

RAPPEL : ce module doit rester importable tel quel depuis le notebook ET depuis le CLI.
    - aucun ``if __name__ == "__main__"`` ;
    - aucun effet de bord à l'import ;
    - imports lourds (faster_whisper, librosa, cv2, yt_dlp, torch) faits paresseusement dans les fonctions.

Pipeline (chaque étape coûteuse est checkpointée -> reprise après une déconnexion Colab) :
    1. probe          ffprobe : durée, flux audio/vidéo
    2. transcribe     faster-whisper (large-v3/medium selon VRAM), word_timestamps=True, repli OOM
    3. audio          librosa (streaming) : loudness dB par 0,1 s -> pics et variations
    4. visual         ffmpeg (NVDEC si dispo) -> frames 160x90 gris à 2 fps -> OpenCV : flux optique + coupes
    5. heatmap        yt-dlp : graphe « most replayed » (VODs YouTube uniquement)
    6. scoring        timeline 1 s -> fenêtres glissantes 15-45 s -> score 0-100 (core.scoring / model.pkl)
    7. sélection      top-k sans chevauchement, bornes recalées sur la parole, durée <= max campagne
    8. hooks          2-3 accroches < 60 caractères par moment (templates locaux)

Sortie : ``<output_dir>/<video_id>.json`` et la liste des moments retenus.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger, log_step
from core.scoring import FEATURES, build_reason, generate_hooks, load_scorer, robust_normalize
from core.storage import Checkpoint, read_json, slugify, utc_now_iso, write_json

log = get_logger(__name__)

SCAN_FORMAT = "clipgod-moments/1"
_TERMINAL_PUNCT = (".", "!", "?", "…")
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOPWORDS = frozenset(
    ["a", "an", "and", "the", "to", "of", "in", "on", "it", "is", "i", "you", "he", "she", "we", "they", "that", "this", "was", "for", "with", "be", "at", "so", "but", "or", "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "à", "au", "il", "elle", "on", "je", "tu", "nous", "vous", "ils", "est", "que", "qui", "pas", "ne", "ça", "ce"]
)

# Agrégation d'un signal par seconde sur une fenêtre
_PEAK_FEATURES = {"audio_loudness_peak", "audio_loudness_delta", "text_unfinished_then_burst",
                  "most_replayed", "text_speech_rate"}
_SUM_FEATURES = {"text_hype_keywords", "text_repetition", "visual_scene_changes"}
_TOPMEAN_FEATURES = {"visual_motion"}


class ScanError(RuntimeError):
    """Échec bloquant du scan (vidéo illisible, ffmpeg absent, transcription impossible…)."""


# ===========================================================================
# Helpers ffmpeg
# ===========================================================================
def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ScanError(f"{name} introuvable dans le PATH (Colab : apt-get install -y ffmpeg).")
    return path


def probe_video(video_path: str) -> dict[str, Any]:
    """Durée et flux d'une vidéo via ffprobe."""
    cmd = [_require_binary("ffprobe"), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(video_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ScanError(f"ffprobe a échoué sur {video_path} : {proc.stderr.strip()[:500]}")
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float((info.get("format") or {}).get("duration") or (video or {}).get("duration") or 0.0)
    fps = 0.0
    if video and video.get("avg_frame_rate") not in (None, "0/0"):
        with contextlib.suppress(ValueError, ZeroDivisionError):
            fps = float(Fraction(video["avg_frame_rate"]))
    if duration <= 0:
        raise ScanError(f"Durée invalide pour {video_path}.")
    return {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": int(video.get("width", 0)) if video else 0,
        "height": int(video.get("height", 0)) if video else 0,
        "fps": round(fps, 3),
    }


def extract_audio(video_path: str, wav_path: Path, sample_rate: int = 16000) -> Path:
    """Extrait l'audio en WAV mono 16 kHz (entrée de whisper et de librosa). Réutilise un WAV existant."""
    if wav_path.is_file() and wav_path.stat().st_size > 44:
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav_path.with_suffix(".part.wav")
    cmd = [_require_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_path), "-vn", "-sn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ScanError(f"Extraction audio impossible : {proc.stderr.strip()[:500]}")
    tmp.replace(wav_path)
    return wav_path


# ===========================================================================
# Transcription (faster-whisper)
# ===========================================================================
def _cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return 0


def cuda_vram_gb() -> float | None:
    """VRAM totale du GPU 0 en Go (torch, sinon nvidia-smi), None si pas de GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception as exc:  # noqa: BLE001 — torch absent ou CUDA cassé : on tente nvidia-smi
        log.debug("torch indisponible pour la VRAM : %s", exc)
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=False)
        with contextlib.suppress(ValueError, IndexError):
            return float(proc.stdout.strip().splitlines()[0]) / 1024
    return None


def choose_whisper_model(wcfg: Mapping[str, Any]) -> tuple[str, str, str]:
    """(modèle, device, compute_type). ``model: auto`` -> large-v3 si VRAM >= seuil, sinon medium.

    Sans GPU (debug local), bascule sur CPU int8 avec un modèle ``small`` (sauf choix explicite).
    """
    requested = str(wcfg.get("model", "auto"))
    device = str(wcfg.get("device", "cuda"))
    compute = str(wcfg.get("compute_type", "float16"))
    if device == "cuda" and _cuda_device_count() == 0:
        log.warning("Aucun GPU CUDA détecté : transcription sur CPU (lent). Sur Colab : Exécution > GPU T4.")
        device, compute = "cpu", "int8"
    if requested != "auto":
        return requested, device, compute
    if device == "cpu":
        return "small", device, compute
    vram = cuda_vram_gb()
    threshold = float(wcfg.get("auto_vram_threshold_gb", 10))
    model = "large-v3" if vram is None or vram >= threshold else "medium"
    log.info("Whisper auto : %s (VRAM %.1f Go)", model, vram or -1)
    return model, device, compute


def _free_gpu() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda failed" in msg or "cublas" in msg or "cudnn" in msg


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Garde les arguments acceptés par ``fn`` (compatibilité entre versions de faster-whisper)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _run_whisper(model: Any, audio_path: str, wcfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language": wcfg.get("language") or None,
        "beam_size": int(wcfg.get("beam_size", 5)),
        "vad_filter": bool(wcfg.get("vad_filter", True)),
        "word_timestamps": True,
        "condition_on_previous_text": False,  # évite les boucles d'hallucination sur les longs streams
    }
    batch_size = int(wcfg.get("batch_size", 0) or 0)
    if batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model=model)
        kwargs["batch_size"] = batch_size
        segments, info = pipeline.transcribe(audio_path, **_filter_kwargs(pipeline.transcribe, kwargs))
    else:
        segments, info = model.transcribe(audio_path, **_filter_kwargs(model.transcribe, kwargs))

    total = float(getattr(info, "duration", 0.0) or 0.0)
    next_report = 0.1
    out: list[dict[str, Any]] = []
    for seg in segments:
        words = [
            {"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word.strip(),
             "prob": round(float(getattr(w, "probability", 1.0)), 3)}
            for w in (seg.words or []) if w.word.strip()
        ]
        out.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                    "text": seg.text.strip(), "words": words})
        if total and seg.end / total >= next_report:
            log.info("transcription %d%%", int(100 * seg.end / total))
            next_report += 0.1
    return {
        "language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
        "duration": total,
        "segments": out,
    }


def transcribe(audio_path: str, wcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Transcrit avec faster-whisper ; en cas d'OOM GPU, retente avec un modèle plus petit."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ScanError("faster-whisper n'est pas installé (pip install -e '.[cloud]').") from exc

    model_name, device, compute = choose_whisper_model(wcfg)
    chain = [model_name] + [m for m in wcfg.get("fallback_models", ["medium", "small"]) if m != model_name]
    last_exc: BaseException | None = None
    for name in chain:
        model = None
        try:
            log.info("Chargement whisper %s (%s/%s)", name, device, compute)
            model = WhisperModel(name, device=device, compute_type=compute)
            result = _run_whisper(model, audio_path, wcfg)
            result.update({"model": name, "device": device, "compute_type": compute})
            return result
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            last_exc = exc
            log.warning("Whisper %s : mémoire GPU insuffisante, essai du modèle suivant.", name)
            if compute == "float16":
                compute = "int8_float16"
        finally:
            del model
            _free_gpu()
    raise ScanError(f"Transcription impossible (mémoire GPU) : {last_exc}")


def words_from_transcript(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Liste plate des mots (start, end, word, token). Segment sans mots -> mots répartis uniformément."""
    words: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []):
        seg_words = seg.get("words") or []
        if not seg_words and seg.get("text"):
            raw = seg["text"].split()
            span = max(0.01, float(seg["end"]) - float(seg["start"]))
            step = span / max(1, len(raw))
            seg_words = [{"start": seg["start"] + i * step, "end": seg["start"] + (i + 1) * step, "word": w}
                         for i, w in enumerate(raw)]
        for w in seg_words:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            tokens = _TOKEN_RE.findall(text.lower())
            words.append({"start": float(w["start"]), "end": float(w["end"]), "word": text,
                          "token": tokens[0] if tokens else ""})
    words.sort(key=lambda w: w["start"])
    return words


# ===========================================================================
# Signaux texte
# ===========================================================================
@dataclass
class TextSignals:
    hype: np.ndarray
    repetition: np.ndarray
    speech_rate: np.ndarray
    cutoffs: list[float]                       # fins de phrases inachevées (secondes)
    matches: list[tuple[float, str]] = field(default_factory=list)  # (temps, mot-clé)


def text_signals(transcript: Mapping[str, Any], n_bins: int, hype_keywords: Sequence[str]) -> TextSignals:
    words = words_from_transcript(transcript)
    hype = np.zeros(n_bins)
    repetition = np.zeros(n_bins)
    speech = np.zeros(n_bins)
    matches: list[tuple[float, str]] = []

    phrases = sorted({tuple(_TOKEN_RE.findall(k.lower())) for k in hype_keywords} - {()}, key=len, reverse=True)
    tokens = [w["token"] for w in words]

    def bin_of(t: float) -> int:
        return min(n_bins - 1, max(0, int(t)))

    for i, w in enumerate(words):
        b = bin_of(w["start"])
        speech[b] += 1
        for phrase in phrases:
            k = len(phrase)
            if i + 1 >= k and tuple(tokens[i - k + 1:i + 1]) == phrase:
                hype[b] += 1.0 + 0.5 * (k - 1)
                matches.append((w["start"], " ".join(phrase)))
                break
        raw = w["word"]
        letters = [c for c in raw if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            hype[b] += 0.5                       # MAJUSCULES = cri
        if "!" in raw:
            hype[b] += 0.3
        tok = w["token"]
        if len(tok) >= 2 and tok not in _STOPWORDS and tok in tokens[max(0, i - 3):i]:
            repetition[b] += 1.0                 # "no no no", "what what"

    cutoffs: list[float] = []
    segments = transcript.get("segments", [])
    for idx, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        nxt = segments[idx + 1] if idx + 1 < len(segments) else None
        gap = (float(nxt["start"]) - float(seg["end"])) if nxt else 99.0
        if not text.endswith(_TERMINAL_PUNCT) or text.endswith(("-", "—", "…")) or gap > 1.5:
            cutoffs.append(float(seg["end"]))
    return TextSignals(hype, repetition, speech, cutoffs, matches)


# ===========================================================================
# Signaux audio (librosa)
# ===========================================================================
def audio_loudness(wav_path: str, n_bins: int, *, sample_rate: int = 16000, hop_seconds: float = 0.1) -> np.ndarray:
    """Loudness max (dBFS) par seconde, calculée en streaming (mémoire constante, même sur 6 h de VOD)."""
    import librosa

    hop = int(sample_rate * hop_seconds)
    frame = 2 * hop
    rms_chunks = []
    stream_kwargs = _filter_kwargs(librosa.stream, {"sr": None, "mono": True, "fill_value": 0.0})
    stream = librosa.stream(wav_path, block_length=512, frame_length=frame, hop_length=hop, **stream_kwargs)
    for block in stream:
        rms_chunks.append(librosa.feature.rms(y=block, frame_length=frame, hop_length=hop, center=False)[0])
    rms = np.concatenate(rms_chunks) if rms_chunks else np.zeros(1)
    db = librosa.amplitude_to_db(rms, ref=1.0, amin=1e-5, top_db=None)
    per_bin = round(1.0 / hop_seconds)
    padded = np.full(n_bins * per_bin, -100.0)
    m = min(len(db), padded.size)
    padded[:m] = db[:m]
    return padded.reshape(n_bins, per_bin).max(axis=1)


# ===========================================================================
# Signaux visuels (ffmpeg + OpenCV)
# ===========================================================================
def _hwaccel_args(choice: str | None) -> list[str]:
    if choice in (None, "", "none", False):
        return []
    if choice == "auto":
        return ["-hwaccel", "cuda"] if shutil.which("nvidia-smi") else []
    return ["-hwaccel", str(choice)]


def visual_signals(video_path: str, n_bins: int, vcfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Mouvement (flux optique Farneback) et coupes de scène (distance d'histogrammes) par seconde."""
    import cv2

    fps = float(vcfg.get("sample_fps", 2))
    width, height = int(vcfg.get("frame_width", 160)), int(vcfg.get("frame_height", 90))
    threshold = float(vcfg.get("scene_threshold", 0.4))
    frame_bytes = width * height
    ffmpeg = _require_binary("ffmpeg")

    def run(hw: list[str]) -> dict[str, np.ndarray]:
        motion = np.zeros(n_bins)
        scenes = np.zeros(n_bins)
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *hw, "-i", str(video_path),
               "-an", "-sn", "-vf", f"fps={fps},scale={width}:{height}:flags=area,format=gray",
               "-f", "rawvideo", "pipe:1"]
        with tempfile.TemporaryFile() as err, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err) as proc:
            prev = prev_hist = None
            idx = 0
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                cur = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
                hist = cv2.calcHist([cur], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
                b = min(n_bins - 1, int(idx / fps))
                if prev is not None:
                    dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if dist > threshold:
                        scenes[b] += 1.0         # coupe : le flux optique n'a pas de sens ici
                    else:
                        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                        mag = float(np.sqrt((flow ** 2).sum(axis=2)).mean())
                        motion[b] = max(motion[b], mag)
                prev, prev_hist = cur, hist
                idx += 1
            proc.wait()
            if proc.returncode != 0:
                err.seek(0)
                raise ScanError(f"ffmpeg (frames) : {err.read().decode(errors='replace').strip()[:400]}")
        if idx == 0:
            raise ScanError("Aucune frame décodée.")
        return {"motion": motion, "scenes": scenes}

    hw = _hwaccel_args(vcfg.get("hwaccel", "auto"))
    try:
        return run(hw)
    except ScanError:
        if not hw:
            raise
        log.warning("Décodage matériel indisponible : repli sur le décodage CPU.")
        return run([])


# ===========================================================================
# Most replayed (yt-dlp)
# ===========================================================================
def is_youtube_url(url: str | None) -> bool:
    return bool(url) and re.search(r"(youtube\.com|youtu\.be)/", str(url)) is not None


def fetch_heatmap(url: str) -> list[dict[str, float]] | None:
    """Graphe « most replayed » d'une vidéo YouTube : [{start_time, end_time, value (0..1)}] ou None."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    heatmap = (info or {}).get("heatmap")
    return heatmap or None


def heatmap_timeline(heatmap: Sequence[Mapping[str, float]] | None, n_bins: int) -> np.ndarray | None:
    if not heatmap:
        return None
    out = np.zeros(n_bins)
    for point in heatmap:
        a = max(0, int(float(point.get("start_time", 0))))
        b = min(n_bins, max(a + 1, round(float(point.get("end_time", a + 1)))))
        out[a:b] = np.maximum(out[a:b], float(point.get("value", 0.0)))
    return out


# ===========================================================================
# Timeline, fenêtres, sélection
# ===========================================================================
@dataclass
class Timeline:
    duration: float
    signals: dict[str, np.ndarray]            # FEATURES -> valeur brute par seconde
    available: set[str]
    words: list[dict[str, Any]]
    matches: list[tuple[float, str]]

    @property
    def n(self) -> int:
        return len(next(iter(self.signals.values())))


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(np.float64)
    return np.convolve(x, np.ones(k) / k, mode="same")


def build_timeline(duration: float, *, transcript: Mapping[str, Any] | None, loudness_db: np.ndarray | None,
                   visual: Mapping[str, np.ndarray] | None, replayed: np.ndarray | None,
                   hype_keywords: Sequence[str]) -> Timeline:
    """Assemble les signaux bruts par seconde (une entrée par élément de :data:`FEATURES`)."""
    n = max(1, int(np.ceil(duration)))
    zeros = np.zeros(n)
    sig: dict[str, np.ndarray] = {f: zeros.copy() for f in FEATURES}
    available: set[str] = set()

    loud = None
    if loudness_db is not None and np.any(loudness_db > -99):
        loud = np.asarray(loudness_db, dtype=np.float64)[:n]
        loud = np.pad(loud, (0, n - loud.size), constant_values=-100.0)
        rel = loud - np.median(loud)
        sig["audio_loudness_peak"] = _moving_average(np.clip(rel, 0, None), 3)
        prev = np.convolve(np.pad(loud, (4, 0), mode="edge"), np.ones(4) / 4, mode="valid")[:n]
        sig["audio_loudness_delta"] = np.clip(loud - prev, 0, None)
        available |= {"audio_loudness_peak", "audio_loudness_delta"}

    words: list[dict[str, Any]] = []
    matches: list[tuple[float, str]] = []
    if transcript and transcript.get("segments"):
        ts = text_signals(transcript, n, hype_keywords)
        words, matches = words_from_transcript(transcript), ts.matches
        sig["text_hype_keywords"] = ts.hype
        sig["text_repetition"] = ts.repetition
        sig["text_speech_rate"] = _moving_average(ts.speech_rate, 3)
        if words:
            available |= {"text_hype_keywords", "text_repetition", "text_speech_rate"}
        if loud is not None:
            burst = zeros.copy()
            for t in ts.cutoffs:
                i = min(n - 1, int(t))
                after = loud[i:min(n, i + 4)]
                before = loud[max(0, i - 5):i] if i > 0 else loud[:1]
                rise = float(after.max() - before.mean()) if after.size and before.size else 0.0
                burst[i] = max(burst[i], max(0.0, rise))
            sig["text_unfinished_then_burst"] = burst
            available.add("text_unfinished_then_burst")

    if visual is not None:
        sig["visual_motion"] = np.asarray(visual["motion"], dtype=np.float64)[:n]
        sig["visual_scene_changes"] = np.asarray(visual["scenes"], dtype=np.float64)[:n]
        available |= {"visual_motion", "visual_scene_changes"}

    if replayed is not None and np.any(replayed > 0):
        sig["most_replayed"] = np.asarray(replayed, dtype=np.float64)[:n]
        available.add("most_replayed")

    for f in FEATURES:  # aligne toutes les longueurs sur n
        arr = sig[f]
        sig[f] = np.pad(arr, (0, n - arr.size)) if arr.size < n else arr[:n]
    return Timeline(duration, sig, available, words, matches)


def window_features(tl: Timeline, lengths: Sequence[int], step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fenêtres glissantes -> (starts, lengths, X normalisé (n_fenêtres, len(FEATURES)))."""
    n = tl.n
    starts_all, lens_all, cols_all = [], [], []
    usable = sorted({min(int(L), n) for L in lengths if L > 0})
    for L in usable:
        starts = np.arange(0, n - L + 1, max(1, step))
        if starts.size == 0:
            continue
        cols = []
        for f in FEATURES:
            view = np.lib.stride_tricks.sliding_window_view(tl.signals[f], L)[starts]
            if f in _PEAK_FEATURES:
                cols.append(view.max(axis=1))
            elif f in _SUM_FEATURES:
                cols.append(view.sum(axis=1) / np.sqrt(L / 15.0))
            else:  # moyenne du quart le plus intense
                top = max(1, L // 4)
                cols.append(np.sort(view, axis=1)[:, -top:].mean(axis=1))
        starts_all.append(starts)
        lens_all.append(np.full(starts.size, L))
        cols_all.append(np.column_stack(cols))
    if not starts_all:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, len(FEATURES)))
    X = np.vstack(cols_all)
    for j in range(X.shape[1]):
        X[:, j] = robust_normalize(X[:, j])
    return np.concatenate(starts_all), np.concatenate(lens_all), X


def intensity_curve(tl: Timeline, weights: Mapping[str, float]) -> np.ndarray:
    """Intensité par seconde (signaux normalisés pondérés) : sert à localiser le pic dans une fenêtre."""
    curve = np.zeros(tl.n)
    for f in FEATURES:
        w = weights.get(f, 0.0)
        if w > 0 and f in tl.available:
            curve += w * robust_normalize(tl.signals[f])
    return curve


def refine_bounds(start: float, end: float, peak: float, words: Sequence[Mapping[str, Any]], *,
                  clip_min: float, clip_max: float, duration: float, lead: float = 0.35) -> tuple[float, float]:
    """Ajuste [start, end] : durée <= clip_max centrée sur le pic, début sur une prise de parole, fin hors mot."""
    if end - start > clip_max:
        start = max(start, peak - lead * clip_max)
        end = start + clip_max
    if end > duration:
        start, end = max(0.0, start - (end - duration)), duration

    # Début : le mot dans [start-2, start+0.5] précédé du plus long silence (on ne commence pas en pleine phrase)
    best, best_gap = None, -1.0
    for i, w in enumerate(words):
        if w["start"] < start - 2.0:
            continue
        if w["start"] > start + 0.5:
            break
        gap = w["start"] - (words[i - 1]["end"] if i > 0 else 0.0)
        if gap > best_gap:
            best, best_gap = w, gap
    if best is not None and best_gap >= 0.25 and peak - (best["start"] - 0.15) <= clip_max:
        start = max(0.0, best["start"] - 0.15)

    # Fin : plafonnée après recalage du début, puis jamais au milieu d'un mot (prolonger si la durée
    # le permet, sinon s'arrêter juste avant le mot).
    end = min(end, start + clip_max, duration)
    for w in words:
        if w["start"] < end < w["end"]:
            end = min(duration, w["end"] + 0.1) if w["end"] + 0.1 - start <= clip_max else w["start"]
            break
    if end - start < clip_min:
        end = min(duration, start + clip_min)
        start = max(0.0, min(start, end - clip_min))
    return round(start, 2), round(end, 2)


def select_moments(starts: np.ndarray, lens: np.ndarray, scores: np.ndarray, tl: Timeline,
                   weights: Mapping[str, float], *, count: int, min_gap: float, clip_min: float,
                   clip_max: float) -> list[dict[str, Any]]:
    """Sélection gloutonne par score décroissant, sans chevauchement (écart minimal ``min_gap``)."""
    curve = intensity_curve(tl, weights)
    chosen: list[dict[str, Any]] = []
    for i in np.argsort(-scores, kind="stable"):
        s, L = int(starts[i]), int(lens[i])
        segment = curve[s:s + L]
        peak = float(s + (int(np.argmax(segment)) if segment.size else 0)) + 0.5
        a, b = refine_bounds(float(s), float(s + L), peak, tl.words, clip_min=clip_min, clip_max=clip_max,
                             duration=tl.duration)
        if any(a < c["end"] + min_gap and b > c["start"] - min_gap for c in chosen):
            continue
        chosen.append({"window": i, "start": a, "end": b, "peak": round(peak, 2), "score": float(scores[i])})
        if len(chosen) >= count:
            break
    return chosen


# ===========================================================================
# Orchestration
# ===========================================================================
def resolve_video_id(video_path: str, config: Mapping[str, Any]) -> str:
    job = config.get("job") or {}
    return str(job.get("video_id") or slugify(Path(video_path).stem, max_len=80))


def _step_json(ckpt: Checkpoint, step: str, path: Path, fn: Callable[[], Any]) -> Any:
    """Exécute ``fn`` une seule fois : résultat stocké dans ``path``, référencé par le checkpoint."""
    if ckpt.is_done(step) and path.is_file():
        log.info("↺ %s : repris depuis le checkpoint", step)
        return read_json(path)
    try:
        data = fn()
    except Exception as exc:
        ckpt.mark_failed(step, exc)
        raise
    write_json(path, data, indent=None)
    ckpt.mark_done(step, str(path))
    return data


def _optional(step: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        log.warning("Signal %s indisponible (%s) : scan poursuivi sans lui.", step, exc, exc_info=True)
        return None


def _words_to_text(words: Sequence[Mapping[str, Any]], pause: float = 0.4) -> str:
    """Texte d'un moment ; une pause > ``pause`` s devient un saut de ligne (frontière de phrase)."""
    parts: list[str] = []
    prev_end = None
    for w in words:
        if prev_end is not None:
            parts.append("\n" if w["start"] - prev_end > pause else " ")
        parts.append(str(w["word"]))
        prev_end = w["end"]
    return "".join(parts)


def _round_list(arr: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(x), digits) for x in arr]


def scan(video_path: str, output_dir: str, config: dict) -> list[dict]:
    """Analyse une vidéo et écrit ``<output_dir>/<video_id>.json``. Retourne les ``top_k`` moments.

    RAPPEL : fonction importable telle quelle depuis le notebook ET le CLI (pas d'effet de bord à l'import).

    Paramètres utiles de ``config`` (en plus de la section ``scan``) :
        config["job"]            : {video_id, streamer, source_url, ...} (fourni par le job Drive)
        scan.transcript_path     : réutiliser une transcription existante (re-scan sans GPU)
        scan.work_dir            : dossier des checkpoints (mettre sur Drive dans le notebook)
        scan.model_path          : model.pkl à utiliser (sinon paths.model)

    Chaque moment : {idx, rank, start, end, duration, peak, score, hooks, reason, signals, keywords,
    transcript, words, video_id, streamer}.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")
    scfg: Mapping[str, Any] = config.get("scan") or {}
    job: Mapping[str, Any] = config.get("job") or {}
    video_id = resolve_video_id(str(video), config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(scfg.get("work_dir") or out_dir / ".work") / video_id
    work.mkdir(parents=True, exist_ok=True)
    tmp = Path(scfg.get("tmp_dir") or Path(tempfile.gettempdir()) / "clipgod") / video_id
    ckpt = Checkpoint(work, "scan")

    with log_step(log, "scan", video_id=video_id) as summary:
        meta = ckpt.run("probe", probe_video, str(video))
        duration = float(meta["duration"])
        n_bins = max(1, int(np.ceil(duration)))
        wav: Path | None = None

        def get_wav() -> Path:
            nonlocal wav
            if wav is None:
                wav = extract_audio(str(video), tmp / "audio16k.wav")
            return wav

        # -- transcription ---------------------------------------------------
        transcript: dict[str, Any] | None = None
        if scfg.get("transcript_path"):
            transcript = read_json(scfg["transcript_path"])
        elif meta["has_audio"]:
            try:
                with log_step(log, "transcribe", video_id=video_id):
                    transcript = _step_json(ckpt, "transcribe", work / "transcript.json",
                                            lambda: transcribe(str(get_wav()), scfg.get("whisper") or {}))
            except Exception:
                if scfg.get("require_transcript", True):
                    raise
                log.warning("Transcription échouée : scan sans signaux texte (sous-titres impossibles).",
                            exc_info=True)
        if transcript is not None:
            write_json(out_dir / f"{video_id}.transcript.json", transcript, indent=None)

        # -- audio / visuel / most replayed ----------------------------------
        loud = None
        if meta["has_audio"]:
            data = _optional("audio", lambda: _step_json(
                ckpt, "audio", work / "audio.json",
                lambda: {"loudness_db": _round_list(audio_loudness(str(get_wav()), n_bins), 2)}))
            loud = np.asarray(data["loudness_db"]) if data else None

        visual = None
        if meta["has_video"] and scfg.get("visual", {}).get("enabled", True):
            def compute_visual() -> dict[str, list[float]]:
                with log_step(log, "visual", video_id=video_id):
                    v = visual_signals(str(video), n_bins, scfg.get("visual") or {})
                return {k: _round_list(a) for k, a in v.items()}

            data = _optional("visual", lambda: _step_json(ckpt, "visual", work / "visual.json", compute_visual))
            visual = {k: np.asarray(v) for k, v in data.items()} if data else None

        replayed = None
        source_url = job.get("source_url") or scfg.get("source_url")
        if scfg.get("most_replayed", True) and is_youtube_url(source_url):
            heat = _optional("heatmap", lambda: _step_json(
                ckpt, "heatmap", work / "heatmap.json", lambda: {"heatmap": fetch_heatmap(source_url)}))
            replayed = heatmap_timeline((heat or {}).get("heatmap"), n_bins)

        # -- scoring ---------------------------------------------------------
        tl = build_timeline(duration, transcript=transcript, loudness_db=loud, visual=visual,
                            replayed=replayed, hype_keywords=scfg.get("hype_keywords") or [])
        if not tl.available:
            raise ScanError("Aucun signal exploitable (ni audio, ni texte, ni visuel).")
        scorer = load_scorer(config)
        weights = scorer.effective_weights(tl.available)

        wcfg = scfg.get("windows") or {}
        wmin, wmax = int(wcfg.get("min_seconds", 15)), int(wcfg.get("max_seconds", 45))
        lengths = list(range(wmin, wmax + 1, int(wcfg.get("length_step_seconds", 5)))) or [wmin]
        starts, lens, X = window_features(tl, lengths, int(wcfg.get("step_seconds", 5)))
        if X.shape[0] == 0:
            raise ScanError("Vidéo trop courte pour être découpée.")
        scores = scorer.score(X, FEATURES, tl.available)

        rules = (config.get("monetize") or {}).get("rules_default") or {}
        clip_max = float(scfg.get("clip_max_seconds") or float(rules.get("max_duration_seconds", 30)) - 0.5)
        clip_min = float(scfg.get("clip_min_seconds") or max(float(rules.get("min_duration_seconds", 7)), 8.0))
        top_k = int(scfg.get("top_k_moments", 3))
        picked = select_moments(
            starts, lens, scores, tl, weights,
            count=top_k + int(scfg.get("max_candidates", 5)),
            min_gap=float(scfg.get("min_gap_between_moments_seconds", 30)),
            clip_min=clip_min, clip_max=min(clip_max, float(wmax)),
        )

        # -- moments + hooks -------------------------------------------------
        lang = scfg.get("hook_language") or (config.get("project") or {}).get("hook_language") or "en"
        streamer = job.get("streamer")
        moments: list[dict[str, Any]] = []
        for rank, p in enumerate(picked):
            a, b = p["start"], p["end"]
            signals = {f: round(float(X[p["window"], j]), 4) for j, f in enumerate(FEATURES)}
            moment_words = [{"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]}
                            for w in tl.words if w["end"] > a and w["start"] < b]
            text = _words_to_text(moment_words)
            keywords = list(dict.fromkeys(k for t, k in tl.matches if a <= t <= b))
            moment = {
                "idx": rank, "rank": rank + 1, "video_id": video_id, "streamer": streamer,
                "start": a, "end": b, "duration": round(b - a, 2), "peak": p["peak"],
                "score": round(p["score"], 1),
                "hooks": [],
                "reason": build_reason({f: signals[f] for f in tl.available}, weights, keywords=keywords,
                                       lang="fr"),
                "signals": signals, "keywords": keywords, "transcript": text, "words": moment_words,
            }
            if rank < top_k:
                moment["hooks"] = generate_hooks(
                    transcript=text, signals={f: signals[f] for f in tl.available}, weights=weights,
                    keywords=keywords, streamer=streamer, n=int(scfg.get("hooks_per_moment", 3)),
                    max_chars=int(scfg.get("hook_max_chars", 60)), lang=lang, seed=f"{video_id}:{a}",
                )
            moments.append(moment)

        # Un moment faible n'est jamais rendu : il reste en candidat (on garde au moins le meilleur).
        min_score = float(scfg.get("min_score", 40))
        selected = [m for m in moments[:top_k] if m["score"] >= min_score] or moments[:1]
        candidates = [m for m in moments if m not in selected]
        for m in candidates:
            m["hooks"] = []
        if len(selected) < min(top_k, len(moments)):
            log.warning("%d moment(s) sous le score minimal %.0f : gardés en candidats.",
                        min(top_k, len(moments)) - len(selected), min_score)
        doc = {
            "format": SCAN_FORMAT,
            "video_id": video_id,
            "job_id": job.get("job_id"),
            "source": job.get("source_url") or str(video),
            "streamer": streamer,
            "duration": round(duration, 2),
            "generated_at": utc_now_iso(),
            "scorer": scorer.describe(),
            "signals_available": sorted(tl.available),
            "transcript_model": (transcript or {}).get("model"),
            "language": (transcript or {}).get("language"),
            "windows_evaluated": int(X.shape[0]),
            "moments": selected,
            "candidates": candidates,
        }
        write_json(out_dir / f"{video_id}.json", doc)
        ckpt.mark_done("moments", str(out_dir / f"{video_id}.json"))
        summary.update(moments=len(selected), best_score=selected[0]["score"] if selected else None)
    if wav is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return selected
