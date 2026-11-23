"""Module 4 — render (CLOUD GPU) : transforme chaque moment en clip vertical 1080x1920 prêt à publier.

RAPPEL : ce module doit rester importable tel quel depuis le notebook ET depuis le CLI.
    - aucun ``if __name__ == "__main__"`` ; aucun effet de bord à l'import ;
    - imports lourds (cv2) faits paresseusement ; ffmpeg est appelé en sous-processus.

Pour chaque moment :
    1. analyse   ffmpeg -> frames 640 px à 2 fps -> OpenCV YuNet : y a-t-il une facecam (visage stable et petit) ?
    2. layout    facecam : gameplay recadré en haut (70 %) + facecam en bas (30 %) + bande de flou de séparation
                 sinon   : recadrage 9:16 plein écran, centré, qui suit en douceur un éventuel visage (lissage EMA)
    3. texte     sous-titres mot à mot (ASS/libass) depuis les timestamps faster-whisper : gros, gras, contour noir,
                 à 64 % de la hauteur, mot courant surligné ; @streamer en bas pendant tout le clip
    4. hook      boîte de texte animée (pop) pendant les 2 premières secondes — une variante par hook (A/B test)
    5. audio     loudnorm deux passes -> -14 LUFS ; voix off Piper optionnelle (lit le hook, musique baissée)
    6. encodage  H.264 (NVENC sur T4 si disponible, sinon libx264), 30 fps, AAC 48 kHz, +faststart

Sorties dans ``output_dir`` :
    <video_id>_<idx>.mp4          variante du hook n°1 (nom attendu par sync/publish)
    <video_id>_<idx>_h<k>.mp4     variantes A/B des hooks suivants (si ``render.ab_variants``)
    <video_id>_<idx>[_h<k>].json  métadonnées (hook, timing, layout, transformations) lues par monetize/publish
Un clip déjà rendu pour le même moment est réutilisé (reprise après interruption).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger, log_step
from core.storage import utc_now_iso, write_json

log = get_logger(__name__)

RENDER_FORMAT = "clipgod-clip/1"
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿️‍]+")


class RenderError(RuntimeError):
    """Échec du rendu (ffmpeg absent, source illisible, tous les clips en échec…)."""


# ===========================================================================
# Utilitaires
# ===========================================================================
def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RenderError("ffmpeg introuvable dans le PATH (Colab : apt-get install -y ffmpeg).")
    return path


def _run(cmd: Sequence[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise RenderError(f"ffmpeg ({what}) a échoué :\n{tail}")
    return proc


def _cache_dir(rcfg: Mapping[str, Any]) -> Path:
    d = Path(rcfg.get("cache_dir") or Path.home() / ".cache" / "clipgod").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_cached(url: str, target: Path, *, timeout: int = 60) -> Path | None:
    """Télécharge ``url`` vers ``target`` si absent. None (avec warning) si le réseau échoue."""
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ClipGod/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(target)
        return target
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        log.warning("Téléchargement impossible (%s) : %s", url, exc)
        return None


def _ass_path(p: Path) -> str:
    """Échappe un chemin pour une option de filtre ffmpeg (``ass=filename=...``)."""
    s = str(p).replace("\\", "/")
    return s.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


def strip_emoji(text: str) -> str:
    return re.sub(r"\s+", " ", _EMOJI_RE.sub("", text)).strip()


# ===========================================================================
# Encodeur
# ===========================================================================
@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True si h264_nvenc encode réellement (compilé ET GPU présent)."""
    if not shutil.which("nvidia-smi"):
        return False
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
         "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0


def encoder_args(rcfg: Mapping[str, Any], *, quality: str = "final") -> list[str]:
    """Arguments vidéo. ``quality="intermediate"`` : quasi sans perte (base réencodée par les variantes)."""
    choice = str(rcfg.get("encoder", "auto"))
    use_nvenc = choice == "h264_nvenc" or (choice == "auto" and nvenc_available())
    common = ["-pix_fmt", "yuv420p", "-r", str(int(rcfg.get("fps", 30)))]
    if use_nvenc:
        cq = 16 if quality == "intermediate" else int(rcfg.get("nvenc_cq", 21))
        return ["-c:v", "h264_nvenc", "-preset", str(rcfg.get("nvenc_preset", "p5")), "-rc", "vbr",
                "-cq", str(cq), "-b:v", "0", "-profile:v", "high", *common]
    crf = 14 if quality == "intermediate" else int(rcfg.get("crf", 20))
    preset = "veryfast" if quality == "intermediate" else str(rcfg.get("preset", "medium"))
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", *common]


def audio_args(rcfg: Mapping[str, Any]) -> list[str]:
    return ["-c:a", "aac", "-b:a", str(rcfg.get("audio_bitrate", "192k")), "-ar", "48000", "-ac", "2"]


# ===========================================================================
# Analyse visage (YuNet)
# ===========================================================================
@dataclass
class FaceSample:
    t: float
    faces: list[tuple[float, float, float, float, float]]  # (cx, cy, w, h, score) normalisés 0..1


@dataclass
class Layout:
    kind: str                                    # "facecam" | "center"
    facecam: tuple[float, float, float, float] | None = None   # (x, y, w, h) normalisés de la facecam
    track: list[tuple[float, float]] = field(default_factory=list)  # (t, centre x normalisé) du recadrage


def sample_frames(video: str, start: float, duration: float, *, fps: float = 2.0, width: int = 640,
                  src_w: int, src_h: int) -> list[tuple[float, np.ndarray]]:
    """Frames BGR (largeur ``width``) échantillonnées à ``fps`` dans [start, start+duration]."""
    height = max(2, round(width * src_h / max(1, src_w) / 2) * 2)
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t",
           f"{duration:.3f}", "-i", video, "-an", "-sn", "-vf", f"fps={fps},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RenderError(f"échantillonnage des frames : {proc.stderr.decode(errors='replace')[-300:]}")
    size = width * height * 3
    raw = proc.stdout
    return [(i / fps, np.frombuffer(raw[i * size:(i + 1) * size], np.uint8).reshape(height, width, 3))
            for i in range(len(raw) // size)]


class YuNetDetector:
    """Détecteur de visages OpenCV YuNet (modèle ONNX ~230 Ko, téléchargé une fois)."""

    def __init__(self, model_path: Path, score_threshold: float = 0.7):
        import cv2

        self._cv2 = cv2
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 50)
        self._size: tuple[int, int] | None = None

    def __call__(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(frame)
        out = []
        for f in faces if faces is not None else []:
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            out.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, score))
        return out


def load_detector(rcfg: Mapping[str, Any]) -> YuNetDetector | None:
    lcfg = rcfg.get("layout") or {}
    url = lcfg.get("yunet_model_url")
    if not url:
        return None
    model = download_cached(url, _cache_dir(rcfg) / "models" / Path(url).name)
    if model is None:
        return None
    try:
        detector = YuNetDetector(model, float(lcfg.get("face_score_threshold", 0.7)))
        log.info("Détection de visages YuNet prête (%s).", Path(url).name)
        return detector
    except Exception as exc:  # noqa: BLE001 — OpenCV trop ancien, modèle corrompu…
        log.warning("YuNet indisponible (%s) : détection de facecam désactivée.", exc)
        return None


def analyze_faces(frames: Sequence[tuple[float, np.ndarray]], detector: Any) -> list[FaceSample]:
    return [FaceSample(t, detector(frame)) for t, frame in frames]


def hook_y_ratio(samples: Sequence[FaceSample], layout: Layout, *, src_aspect: float, out_aspect: float,
                 default: float = 0.2) -> float:
    """Hauteur (0..1) du hook : ``default``, ou juste sous le visage si un visage occupe cette zone du clip.

    Ne concerne que le recadrage centré pleine hauteur (en layout facecam, le visage est dans le panneau du bas).
    """
    if layout.kind != "center" or not samples or src_aspect <= out_aspect:
        return default
    window = min(1.0, out_aspect / src_aspect)                  # largeur de la fenêtre 9:16 dans la source
    centre = float(np.mean([x for _, x in layout.track])) if layout.track else 0.5
    visible = []
    for s in samples:
        inside = [f for f in s.faces if abs(f[0] - centre) < window / 2 and f[2] >= 0.02]
        if inside:
            visible.append(max(inside, key=lambda f: f[2] * f[3]))
    if len(visible) < 0.3 * len(samples):
        return default
    cy = float(np.median([f[1] for f in visible]))
    fh = float(np.median([f[3] for f in visible]))
    face_zone = (cy - fh, cy + 2.2 * fh)                         # tête + épaules / cadre de la webcam
    if default + 0.06 < face_zone[0] or default - 0.06 > face_zone[1]:
        return default
    return round(min(0.5, face_zone[1] + 0.04), 3)


def decide_layout(samples: Sequence[FaceSample], lcfg: Mapping[str, Any], *, src_aspect: float,
                  manual_region: Sequence[float] | None = None, panel_aspect: float = 1080 / 576) -> Layout:
    """Choisit le layout à partir des détections.

    Facecam = un visage *stable* (même position dans >= ``facecam_min_ratio`` des frames) et *petit*
    (largeur < ``facecam_max_face_width``) : c'est l'incrustation webcam du streamer. Un visage qui bouge ou
    occupe l'écran (PNJ, cinématique, IRL) n'est pas une facecam.
    """
    mode = str(lcfg.get("mode", "auto"))
    if manual_region:
        return Layout("facecam", tuple(float(v) for v in manual_region))  # type: ignore[arg-type]
    if src_aspect < 1.0 or mode == "center":
        return Layout("center", track=_track(samples, lcfg))

    best = [(s.t, max(s.faces, key=lambda f: f[4])) for s in samples if s.faces]
    min_ratio = float(lcfg.get("facecam_min_ratio", 0.5))
    if best and len(samples):
        cx = float(np.median([f[0] for _, f in best]))
        cy = float(np.median([f[1] for _, f in best]))
        fw = float(np.median([f[2] for _, f in best]))
        fh = float(np.median([f[3] for _, f in best]))
        tol = float(lcfg.get("facecam_position_tolerance", 0.05))
        stable = [f for _, f in best if abs(f[0] - cx) <= tol and abs(f[1] - cy) <= tol]
        small = fw < float(lcfg.get("facecam_max_face_width", 0.25))
        if small and len(stable) / len(samples) >= min_ratio:
            return Layout("facecam", facecam_region(cx, cy, fw, fh, lcfg, src_aspect, panel_aspect))
    if mode == "facecam":
        log.warning("Mode facecam forcé mais aucune facecam stable détectée : recadrage centré.")
    return Layout("center", track=_track(samples, lcfg))


def facecam_region(cx: float, cy: float, fw: float, fh: float, lcfg: Mapping[str, Any],
                   src_aspect: float, panel_aspect: float = 1080 / 576) -> tuple[float, float, float, float]:
    """Rectangle (normalisé) autour du visage, au ratio du panneau du bas (1080x576 -> 1.875)."""
    zoom = float(lcfg.get("facecam_zoom", 2.6))
    h = min(1.0, fh * zoom)
    w = min(1.0, h * panel_aspect / src_aspect)       # en coordonnées normalisées : w_px/h_px = w*A/h
    if w >= 1.0:
        w, h = 1.0, min(1.0, src_aspect / panel_aspect)
    x = min(max(0.0, cx - w / 2), 1.0 - w)
    y = min(max(0.0, cy - h * 0.45), 1.0 - h)          # visage un peu au-dessus du centre
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def _track(samples: Sequence[FaceSample], lcfg: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Trajectoire lissée (EMA) du centre horizontal du recadrage ; 0.5 = centre sans visage."""
    alpha = float(lcfg.get("smoothing_alpha", 0.15))
    min_w = float(lcfg.get("track_min_face_width", 0.08))
    out: list[tuple[float, float]] = []
    x = 0.5
    for s in samples:
        big = [f for f in s.faces if f[2] >= min_w]
        target = max(big, key=lambda f: f[2] * f[4])[0] if big else 0.5
        x = x + alpha * (target - x)
        out.append((s.t, round(x, 4)))
    return out or [(0.0, 0.5)]


def piecewise_expr(points: Sequence[tuple[float, float]], *, scale: float, offset: float = 0.0,
                   lo: float, hi: float) -> str:
    """Expression ffmpeg (en fonction de ``t``) interpolant linéairement ``points`` (t, valeur)."""
    vals = [(t, min(hi, max(lo, v * scale + offset))) for t, v in points]
    if len(vals) == 1 or max(v for _, v in vals) - min(v for _, v in vals) < 2:
        return f"{vals[0][1]:.1f}"
    expr = f"{vals[-1][1]:.1f}"
    for (t0, v0), (t1, v1) in reversed(list(itertools.pairwise(vals))):
        slope = (v1 - v0) / max(1e-6, t1 - t0)
        expr = f"if(lt(t,{t1:.3f}),{v0:.1f}+({slope:.3f})*(t-{t0:.3f}),{expr})"
    return f"if(lt(t,{vals[0][0]:.3f}),{vals[0][1]:.1f},{expr})"


# ===========================================================================
# Sous-titres ASS
# ===========================================================================
def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = round(t * 100)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def _ass_color(value: str, default: str) -> str:
    """Accepte ``&H00BBGGRR`` (ASS), ``#RRGGBB`` ou un nom simple."""
    names = {"white": "#FFFFFF", "black": "#000000", "yellow": "#FFE600", "red": "#FF2D2D", "green": "#39FF14"}
    v = names.get(str(value).lower(), str(value or default))
    if v.startswith("&H"):
        return v
    if v.startswith("#") and len(v) == 7:
        r, g, b = v[1:3], v[3:5], v[5:7]
        return f"&H00{b}{g}{r}".upper()
    return default


def chunk_words(words: Sequence[Mapping[str, Any]], *, max_words: int = 3, max_chars: int = 18,
                max_gap: float = 0.6) -> list[list[dict[str, Any]]]:
    """Regroupe les mots en petits blocs lisibles (coupure sur pause, ponctuation finale ou longueur)."""
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        text = str(w["word"]).strip()
        if not text:
            continue
        item = {"start": float(w["start"]), "end": float(w["end"]), "word": text}
        too_long = cur and (len(cur) >= max_words or len(" ".join(x["word"] for x in cur)) + len(text) + 1 > max_chars)
        gap = cur and item["start"] - cur[-1]["end"] > max_gap
        if too_long or gap:
            chunks.append(cur)
            cur = []
        cur.append(item)
        if text.endswith((".", "!", "?", "…")):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(*, words: Sequence[Mapping[str, Any]], clip_start: float, clip_duration: float,
              rcfg: Mapping[str, Any], hook: str | None = None, streamer: str | None = None,
              include_subtitles: bool = True) -> str:
    """Script ASS complet : sous-titres mot à mot + @streamer + boîte de hook animée."""
    W, H = int(rcfg.get("width", 1080)), int(rcfg.get("height", 1920))
    sub = rcfg.get("subtitles") or {}
    hk = rcfg.get("hook") or {}
    tag = rcfg.get("streamer_overlay") or {}
    font = str(sub.get("font_name") or "Montserrat ExtraBold")
    upper = bool(sub.get("uppercase", True))
    base_color = _ass_color(sub.get("color", "white"), "&H00FFFFFF")
    hi_color = _ass_color(sub.get("highlight_color", "&H0000E6FF"), "&H0000E6FF")
    sub_y = int(H * float(sub.get("y_position_ratio", 0.64)))
    hook_y = int(H * float(hk.get("y_position_ratio", 0.2)))
    tag_y = int(H * float(tag.get("y_position_ratio", 0.94)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},{int(sub.get("font_size", 86))},{base_color},{base_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{int(sub.get("outline", 6))},2,5,60,60,0,1
Style: Hook,{font},{int(hk.get("font_size", 92))},{_ass_color(hk.get("text_color", "black"), "&H00000000")},&H00000000,{_ass_color(hk.get("box_color", "white"), "&H00FFFFFF")},&H00000000,-1,0,0,0,100,100,0,0,3,{int(hk.get("box_padding", 22))},0,5,90,90,0,1
Style: Tag,{font},{int(tag.get("font_size", 44))},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    end_all = _ass_time(clip_duration)

    if include_subtitles:
        rel = [{"start": float(w["start"]) - clip_start, "end": float(w["end"]) - clip_start,
                "word": strip_emoji(str(w["word"]))}
               for w in words if float(w["end"]) > clip_start and float(w["start"]) < clip_start + clip_duration]
        rel = [w for w in rel if w["word"]]
        chunks = chunk_words(rel, max_words=int(sub.get("max_words", 3)), max_chars=int(sub.get("max_chars", 18)))
        for ci, chunk in enumerate(chunks):
            nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else clip_duration
            chunk_end = min(clip_duration, max(chunk[-1]["end"], min(nxt, chunk[-1]["end"] + 0.4)))
            labels = [_ass_escape(w["word"].upper() if upper else w["word"]) for w in chunk]
            for wi, w in enumerate(chunk):
                start = max(0.0, w["start"])
                end = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else chunk_end
                if end - start < 0.05:
                    continue
                pop = r"{\fscx112\fscy112\t(0,90,\fscx100\fscy100)}" if wi == 0 else ""
                text = " ".join(
                    (rf"{{\c{hi_color}}}{lab}{{\c{base_color}}}" if k == wi else lab) for k, lab in enumerate(labels)
                )
                events.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Sub,,0,0,0,,"
                              f"{{\\an5\\pos({W // 2},{sub_y})}}{pop}{text}")

    if streamer and tag.get("enabled", True):
        label = str(tag.get("template", "@{streamer}")).format(streamer=str(streamer).lstrip("@"))
        events.append(f"Dialogue: 0,{_ass_time(0)},{end_all},Tag,,0,0,0,,"
                      f"{{\\an5\\pos({W // 2},{tag_y})\\alpha&H30&}}{_ass_escape(label)}")

    if hook:
        dur = float(hk.get("duration_seconds", 2.0))
        anim = {
            "pop": r"\fscx40\fscy40\t(0,160,\fscx108\fscy108)\t(160,260,\fscx100\fscy100)",
            "slide": rf"\move({W // 2},{hook_y - 120},{W // 2},{hook_y},0,220)",
            "none": "",
        }.get(str(hk.get("animation", "pop")), "")
        pos = "" if "move" in anim else rf"\pos({W // 2},{hook_y})"
        text = _ass_escape(strip_emoji(hook) if hk.get("strip_emoji", True) else hook)
        events.append(f"Dialogue: 2,{_ass_time(0)},{_ass_time(min(dur, clip_duration))},Hook,,0,0,0,,"
                      f"{{\\an5{pos}\\fad(0,180){anim}}}{text}")
    return header + "\n".join(events) + "\n"


def ensure_font(rcfg: Mapping[str, Any]) -> Path | None:
    """Télécharge la police des sous-titres dans le cache (libass retombe sur une police système sinon)."""
    url = (rcfg.get("subtitles") or {}).get("font_url")
    if not url:
        return None
    path = download_cached(url, _cache_dir(rcfg) / "fonts" / Path(url).name)
    return path.parent if path else None


# ===========================================================================
# Audio : loudnorm deux passes + voix off
# ===========================================================================
def measure_loudness(video: str, start: float, duration: float, target: float) -> dict[str, str] | None:
    """Première passe loudnorm (audio seul, rapide) -> mesures pour la passe linéaire."""
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", video,
           "-vn", "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.DOTALL)
    if proc.returncode != 0 or not match:
        log.warning("Mesure loudnorm impossible : normalisation en une passe.")
        return None
    data = json.loads(match.group(0))
    if data.get("input_i") in ("-inf", None):
        return None
    return data


def loudnorm_filter(target: float, measured: Mapping[str, str] | None) -> str:
    base = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    if not measured:
        return base
    return (f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


def piper_voice_urls(voice: str, base_url: str) -> tuple[str, str]:
    """``en_US-ryan-high`` -> URLs .onnx et .onnx.json du dépôt rhasspy/piper-voices."""
    lang_region, speaker, quality = voice.split("-", 2)
    family = lang_region.split("_")[0]
    root = f"{base_url.rstrip('/')}/{family}/{lang_region}/{speaker}/{quality}/{voice}"
    return f"{root}.onnx", f"{root}.onnx.json"


def synthesize_voice(text: str, out_wav: Path, rcfg: Mapping[str, Any]) -> Path | None:
    """Voix off locale Piper. None (warning) si piper / la voix sont indisponibles : le clip est rendu sans."""
    vcfg = rcfg.get("voiceover") or {}
    text = strip_emoji(text)
    if not text:
        return None
    try:
        import piper  # noqa: F401
    except ImportError:
        log.warning("piper-tts non installé (bash scripts/colab_setup.sh --with-tts) : voix off ignorée.")
        return None
    voice = str(vcfg.get("voice", "en_US-ryan-high"))
    base = str(vcfg.get("voices_base_url", "https://huggingface.co/rhasspy/piper-voices/resolve/main"))
    onnx_url, json_url = piper_voice_urls(voice, base)
    vdir = _cache_dir(rcfg) / "voices"
    onnx = download_cached(onnx_url, vdir / f"{voice}.onnx", timeout=300)
    meta = download_cached(json_url, vdir / f"{voice}.onnx.json")
    if onnx is None or meta is None:
        return None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "piper", "--model", str(onnx), "--output_file", str(out_wav)],
                          input=text, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_wav.is_file():
        log.warning("Piper a échoué : %s", proc.stderr.strip()[-300:])
        return None
    return out_wav


def _wav_duration(path: Path) -> float:
    import wave

    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / float(w.getframerate())


# ===========================================================================
# Construction du graphe ffmpeg
# ===========================================================================
def detect_letterbox(frames: Sequence[tuple[float, np.ndarray]], src_h: int, *, min_ratio: float = 0.04,
                     min_share: float = 0.8) -> tuple[int, int] | None:
    """Bandes noires (cinématiques « cinéma ») présentes sur >= ``min_share`` des images.

    Retourne ``(y, h)`` de la zone utile en pixels source (pairs), ou None s'il n'y a rien à retirer.
    """
    from modules.scan import letterbox_bars

    tops, bottoms = [], []
    for _, frame in frames:
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        top, bottom = letterbox_bars(gray)
        if top >= gray.shape[0]:                      # image noire (fondu) : ignorée
            continue
        tops.append(top / gray.shape[0])
        bottoms.append(bottom / gray.shape[0])
    if len(tops) < 2:
        return None
    # Quantile bas : une bande doit être là sur (presque) toutes les images pour être retirée.
    q = 100 * (1 - min_share)
    top_r, bot_r = float(np.percentile(tops, q)), float(np.percentile(bottoms, q))
    if top_r < min_ratio or bot_r < min_ratio:
        return None
    y = int(top_r * src_h) // 2 * 2 + 2                   # +2 px : pas de liseré sombre
    h = (src_h - y - int(bot_r * src_h) - 2) // 2 * 2
    return (y, h) if h > src_h * 0.5 else None


def video_filter(layout: Layout, *, src_w: int, src_h: int, rcfg: Mapping[str, Any],
                 letterbox: tuple[int, int] | None = None) -> str:
    """Graphe vidéo (entrée [0:v], sortie [vl]) : layout 1080x1920 sans texte.

    ``letterbox`` (y, h) : zone utile d'une image à bandes noires, recadrée avant le format vertical.
    """
    if letterbox is not None:
        y, h = letterbox
        inner = video_filter(layout, src_w=src_w, src_h=h, rcfg=rcfg)
        return f"[0:v]crop={src_w}:{h}:0:{y}[lbx];" + inner.replace("[0:v]", "[lbx]", 1)
    W, H = int(rcfg.get("width", 1080)), int(rcfg.get("height", 1920))
    lcfg = rcfg.get("layout") or {}
    fps = int(rcfg.get("fps", 30))
    even = lambda v: max(2, int(v) // 2 * 2)

    if layout.kind == "facecam" and layout.facecam:
        top_h = even(H * float(lcfg.get("gameplay_ratio", 0.70)))
        bot_h = H - top_h
        # Gameplay : bande centrale pleine hauteur au ratio du panneau du haut.
        gw = even(min(src_w, src_h * W / top_h))
        gx = even((src_w - gw) / 2)
        fx, fy, fw, fh = layout.facecam
        cw, ch = even(fw * src_w), even(fh * src_h)
        cx, cy = even(fx * src_w), even(fy * src_h)
        blur = int(lcfg.get("separator_blur", 25))
        sep = even(lcfg.get("separator_height", 48))
        return (
            f"[0:v]split=3[g][c][b];"
            f"[g]crop={gw}:{src_h}:{gx}:0,scale={W}:{top_h}:flags=lanczos,setsar=1[top];"
            f"[b]crop={gw}:{src_h}:{gx}:0,scale={W}:{bot_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{bot_h},boxblur={blur}:2,eq=brightness=-0.12,setsar=1[bg];"
            f"[c]crop={cw}:{ch}:{cx}:{cy},scale={W}:{bot_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"setsar=1[cam];"
            f"[bg][cam]overlay=(W-w)/2:(H-h)/2[bot];"
            f"[top][bot]vstack=inputs=2,fps={fps}[st];"
            f"[st]split[s1][s2];[s2]crop={W}:{sep}:0:{top_h - sep // 2},boxblur={max(4, blur // 2)}:1[sep];"
            f"[s1][sep]overlay=0:{top_h - sep // 2},format=yuv420p[vl]"
        )

    # Recadrage 9:16 plein écran (centré, suivi lissé d'un visage éventuel)
    if src_w / src_h <= W / H:  # source déjà verticale (ou plus étroite) : on adapte simplement
        return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},format=yuv420p[vl]")
    cw = even(src_h * W / H)
    x_expr = piecewise_expr(layout.track or [(0.0, 0.5)], scale=src_w, offset=-cw / 2, lo=0, hi=src_w - cw)
    return (f"[0:v]crop={cw}:{src_h}:'{x_expr}':0,scale={W}:{H}:flags=lanczos,setsar=1,"
            f"fps={fps},format=yuv420p[vl]")


# ===========================================================================
# Rendu d'un moment
# ===========================================================================
def clip_stem(video_id: str, idx: int, variant: int) -> str:
    return f"{video_id}_{idx}" if variant == 0 else f"{video_id}_{idx}_h{variant}"


def render_signature(rcfg: Mapping[str, Any], streamer: str | None) -> str:
    """Empreinte des réglages de rendu : un changement de config force un nouveau rendu."""
    import hashlib

    payload = json.dumps({"render": rcfg, "streamer": streamer}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _already_rendered(mp4: Path, sidecar: Path, moment: Mapping[str, Any], hook: str | None,
                      signature: str) -> bool:
    if not (mp4.is_file() and mp4.stat().st_size > 0 and sidecar.is_file()):
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (abs(float(meta.get("start", -1)) - float(moment["start"])) < 0.01
            and abs(float(meta.get("end", -1)) - float(moment["end"])) < 0.01 and meta.get("hook") == hook
            and meta.get("render_signature") == signature)


def render_moment(moment: Mapping[str, Any], video_path: str, output_dir: Path, config: Mapping[str, Any], *,
                  src: Mapping[str, Any], detector: Any, fonts_dir: Path | None) -> list[str]:
    """Rend toutes les variantes (une par hook) d'un moment. Retourne les chemins .mp4."""
    rcfg: Mapping[str, Any] = config.get("render") or {}
    lcfg = rcfg.get("layout") or {}
    job = config.get("job") or {}
    video_id = str(moment.get("video_id") or job.get("video_id") or Path(video_path).stem)
    idx = int(moment.get("idx", 0))
    start, end = float(moment["start"]), float(moment["end"])
    duration = round(end - start, 3)
    if duration <= 0.5:
        raise RenderError(f"Moment {idx} trop court ({duration}s).")
    streamer = moment.get("streamer") or job.get("streamer")
    hooks: list[str | None] = list(moment.get("hooks") or []) or [None]
    if not rcfg.get("ab_variants", True):
        hooks = hooks[:1]

    signature = render_signature(rcfg, streamer)
    pending = [(k, h) for k, h in enumerate(hooks)
               if not _already_rendered(output_dir / f"{clip_stem(video_id, idx, k)}.mp4",
                                        output_dir / f"{clip_stem(video_id, idx, k)}.json", moment, h, signature)]
    done = [str(output_dir / f"{clip_stem(video_id, idx, k)}.mp4") for k, _ in enumerate(hooks)
            if (k, hooks[k]) not in pending]
    if not pending:
        log.info("↺ clip %s_%d déjà rendu", video_id, idx)
        return done

    src_w, src_h = int(src["width"]), int(src["height"])
    regions = lcfg.get("facecam_regions") or {}
    manual = regions.get(str(streamer or "").lstrip("@").lower()) if streamer else None
    samples: list[FaceSample] = []
    frames: list[tuple[float, np.ndarray]] = []
    want_faces = detector is not None and not manual and lcfg.get("mode", "auto") != "center"
    if want_faces or rcfg.get("remove_letterbox", True):
        frames = sample_frames(video_path, start, duration, fps=float(lcfg.get("analysis_fps", 2)),
                               src_w=src_w, src_h=src_h)
    if want_faces:
        samples = analyze_faces(frames, detector)
    out_w, out_h = int(rcfg.get("width", 1080)), int(rcfg.get("height", 1920))
    panel_h = out_h - int(out_h * float(lcfg.get("gameplay_ratio", 0.70))) // 2 * 2
    layout = decide_layout(samples, lcfg, src_aspect=src_w / src_h, manual_region=manual,
                           panel_aspect=out_w / max(1, panel_h))
    with_faces = sum(1 for s in samples if s.faces)
    if detector is None and not manual:
        log.info("Détection de visages inactive (YuNet non chargé) : layout %s par défaut.", layout.kind)
    log.info("layout %s pour %s_%d · visages dans %d/%d images", layout.kind, video_id, idx, with_faces,
             len(samples), extra={"facecam": layout.facecam})
    hk = rcfg.get("hook") or {}
    hook_y = hook_y_ratio(samples, layout, src_aspect=src_w / src_h, out_aspect=out_w / out_h,
                          default=float(hk.get("y_position_ratio", 0.2)))
    if hook_y != float(hk.get("y_position_ratio", 0.2)):
        log.info("hook déplacé à %.0f %% de la hauteur pour ne pas couvrir le visage", hook_y * 100)
    text_cfg = {**rcfg, "hook": {**hk, "y_position_ratio": hook_y}}

    # Bandes noires retirées seulement en plein écran : avec facecam, l'incrustation peut chevaucher la bande.
    letterbox = detect_letterbox(frames, src_h) if rcfg.get("remove_letterbox", True) and layout.kind == "center" \
        else None
    if letterbox:
        log.info("bandes noires retirées : zone utile %dpx sur %dpx de haut", letterbox[1], src_h)
    vfilter = video_filter(layout, src_w=src_w, src_h=src_h, rcfg=rcfg, letterbox=letterbox)
    target = float(rcfg.get("loudness_lufs", -14))
    measured = measure_loudness(video_path, start, duration, target) if src.get("has_audio") else None
    voiceover = bool((rcfg.get("voiceover") or {}).get("enabled"))
    words = moment.get("words") or []
    fonts_opt = f":fontsdir={_ass_path(fonts_dir)}" if fonts_dir else ""
    ff = _ffmpeg()

    with tempfile.TemporaryDirectory(prefix="clipgod_render_") as tmpd:
        tmp = Path(tmpd)
        single_pass = len(pending) == 1 and not voiceover
        # Passe A : layout + sous-titres + @streamer + audio normalisé (+ hook si une seule variante).
        base_ass = tmp / "base.ass"
        base_ass.write_text(build_ass(words=words, clip_start=start, clip_duration=duration, rcfg=text_cfg,
                                      streamer=streamer, hook=pending[0][1] if single_pass else None),
                            encoding="utf-8")
        afilter = (f"[0:a]{loudnorm_filter(target, measured)},aresample=48000[al]"
                   if src.get("has_audio") else "anullsrc=r=48000:cl=stereo[al]")
        graph = f"{vfilter};[vl]ass=filename={_ass_path(base_ass)}{fonts_opt}[v];{afilter}"
        base_out = (output_dir / f"{clip_stem(video_id, idx, pending[0][0])}.mp4") if single_pass else tmp / "base.mp4"
        cmd = [ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t",
               f"{duration:.3f}", "-i", video_path, "-filter_complex", graph, "-map", "[v]", "-map", "[al]",
               "-t", f"{duration:.3f}", *encoder_args(rcfg, quality="final" if single_pass else "intermediate"),
               *audio_args(rcfg), "-movflags", "+faststart", str(base_out)]
        try:
            _run(cmd, "rendu")
        except RenderError:
            if encoder_args(rcfg)[1] != "h264_nvenc":
                raise
            log.warning("NVENC en échec : nouvel essai en libx264.")
            nvenc_available.cache_clear()
            cpu_cfg = {**rcfg, "encoder": "libx264"}
            cmd = [*cmd[:cmd.index("-c:v")],
                   *encoder_args(cpu_cfg, quality="final" if single_pass else "intermediate"),
                   *audio_args(rcfg), "-movflags", "+faststart", str(base_out)]
            _run(cmd, "rendu libx264")
            rcfg = cpu_cfg

        outputs: list[str] = []
        for k, hook in pending:
            final = output_dir / f"{clip_stem(video_id, idx, k)}.mp4"
            vo_used = False
            if not single_pass:
                hook_ass = tmp / f"hook_{k}.ass"
                hook_ass.write_text(build_ass(words=[], clip_start=start, clip_duration=duration, rcfg=text_cfg,
                                              hook=hook, include_subtitles=False), encoding="utf-8")
                inputs = ["-i", str(base_out)]
                agraph = "[0:a]anull[a]"
                vo = synthesize_voice(hook, tmp / f"vo_{k}.wav", rcfg) if voiceover and hook else None
                if vo is not None:
                    vo_dur = _wav_duration(vo) + 0.3
                    duck = float((rcfg.get("voiceover") or {}).get("duck_volume", 0.3))
                    inputs += ["-i", str(vo)]
                    agraph = (f"[0:a]volume=enable='between(t,0,{vo_dur:.2f})':volume={duck}[bed];"
                              f"[1:a]aresample=48000,adelay=150|150,volume=1.4[vo];"
                              f"[bed][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
                    vo_used = True
                graph = f"[0:v]ass=filename={_ass_path(hook_ass)}{fonts_opt}[v];{agraph}"
                cmd = [ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", graph,
                       "-map", "[v]", "-map", "[a]", *encoder_args(rcfg), *audio_args(rcfg),
                       "-movflags", "+faststart", str(final)]
                _run(cmd, f"variante h{k}")
            sidecar = {
                "format": RENDER_FORMAT,
                "clip_id": final.stem, "file": final.name, "video_id": video_id, "job_id": job.get("job_id"),
                "moment_idx": idx, "variant": k, "hook": hook, "hooks": moment.get("hooks") or [],
                "start": start, "end": end, "duration": duration, "score": moment.get("score"),
                "signals": moment.get("signals") or {}, "reason": moment.get("reason"),
                "streamer": streamer, "platform": job.get("platform"), "source_url": job.get("source_url"),
                "layout": layout.kind, "facecam_region": layout.facecam,
                "width": int(rcfg.get("width", 1080)), "height": int(rcfg.get("height", 1920)),
                "fps": int(rcfg.get("fps", 30)), "loudness_lufs": target, "voiceover": vo_used,
                "hook_at_seconds": 0.0 if hook else None,
                "transformations": [t for t, on in (
                    ("vertical_reframe", True), ("facecam_split", layout.kind == "facecam"),
                    ("word_subtitles", bool(words)), ("hook_overlay", bool(hook)),
                    ("streamer_tag", bool(streamer)), ("loudness_normalized", bool(src.get("has_audio"))),
                    ("voiceover", vo_used)) if on],
                "render_signature": signature,
                "rendered_at": utc_now_iso(),
            }
            write_json(final.with_suffix(".json"), sidecar)
            outputs.append(str(final))
    return done + outputs


# ===========================================================================
# Point d'entrée
# ===========================================================================
def render(moments: list[dict], video_path: str, output_dir: str, config: dict) -> list[str]:
    """Rend chaque moment en clip(s) vertical(aux) 1080x1920 H.264. Retourne la liste des .mp4 produits.

    RAPPEL : fonction importable telle quelle depuis le notebook ET le CLI (pas d'effet de bord à l'import).

    Un moment en échec est journalisé et n'empêche pas les autres ; si TOUS échouent -> :class:`RenderError`.
    ``config["render"]["voiceover"]["enabled"]`` active la voix off ; ``config["job"]`` fournit video_id/streamer.
    """
    from modules.scan import probe_video  # léger : ffprobe uniquement

    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")
    if not moments:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rcfg: Mapping[str, Any] = config.get("render") or {}
    src = probe_video(str(video))
    if not src["has_video"]:
        raise RenderError(f"Pas de flux vidéo dans {video}.")
    lcfg = rcfg.get("layout") or {}
    detector = load_detector(rcfg) if lcfg.get("mode", "auto") != "center" else None
    fonts_dir = ensure_font(rcfg)

    produced: list[str] = []
    errors: list[str] = []
    with log_step(log, "render", video=video.name, moments=len(moments)) as summary:
        for moment in moments:
            idx = moment.get("idx", "?")
            try:
                with log_step(log, "render.moment", idx=idx, start=moment.get("start"), end=moment.get("end")):
                    produced += render_moment(moment, str(video), out, config, src=src, detector=detector,
                                              fonts_dir=fonts_dir)
            except Exception as exc:  # noqa: BLE001 — un moment raté ne bloque pas les autres
                errors.append(f"moment {idx}: {exc}")
                log.error("Rendu du moment %s impossible : %s", idx, exc)
        summary.update(clips=len(produced), errors=len(errors))
    if errors and not produced:
        raise RenderError("Aucun clip rendu :\n" + "\n".join(errors))
    return produced
