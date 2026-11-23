"""Moments viraux (laptop) : les Shorts GTA 6 les plus vus -> le même moment recoupé dans la VOD d'origine.

``clipgod viral`` prépare les extraits que ``clipgod bestof --viral`` assemblera, classés par vues :

    1. Shorts   ``search.list`` (Shorts GTA 6 des dernières ``hours`` h, les plus vus d'abord) + ``videos.list`` ;
    2. VOD      pour chaque Short, VODs d'origine candidates : liens de la description (YouTube, Twitch),
                dernières vidéos longues de la même chaîne (streamers qui postent leurs propres clips),
                dernières vidéos longues des chaînes @mentionnées ;
    3. seconde  empreinte audio (énergie de 24 bandes, variations dans le temps) du Short comparée à celle de la VOD
                par corrélation (FFT) : 3 morceaux du Short doivent tomber au même endroit -> début du moment ;
                l'audio est téléchargé en basse qualité, analysé puis supprimé ;
    4. extrait  seuls les ~30-50 s utiles de la VOD sont téléchargés (yt-dlp ``download_ranges``) puis déposés
                comme job Drive avec ``fixed_moment`` : Colab transcrit l'extrait et monte le clip (sous-titres,
                hook, crédit au streamer), sans réanalyser la VOD.

Le Short lui-même n'est jamais réutilisé dans le montage : il sert seulement à choisir et localiser le moment.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get, path, resolve_path
from core.logger import get_logger
from core.quota import QuotaExceeded, YouTubeQuota
from core.storage import StateStore, utc_now, utc_now_iso

log = get_logger(__name__)

SAMPLE_RATE = 8000
FRAME = 512
HOP = 256                              # 32 ms -> 31,25 trames/s
FPS = SAMPLE_RATE / HOP
N_BANDS = 24

YT_LINK = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^\s]*&)?v=|live/)|youtu\.be/)([\w-]{11})")
TWITCH_LINK = re.compile(r"twitch\.tv/videos/(\d+)")
MENTION = re.compile(r"(?<![\w.])@([A-Za-z0-9_.-]{3,30})")


class ViralError(RuntimeError):
    """Clé YouTube manquante, ffmpeg absent…"""


@dataclass
class Short:
    id: str
    title: str
    channel: str
    channel_id: str
    views: int
    duration: float
    published_at: datetime
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/shorts/{self.id}"


@dataclass
class Located:
    short: Short
    vod_url: str
    vod_title: str
    streamer: str
    offset: float                     # début du moment du Short dans la VOD (s)
    confidence: float
    clip_start: float = 0.0
    clip_end: float = 0.0
    job_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Empreinte audio
# ---------------------------------------------------------------------------
def _band_edges() -> np.ndarray:
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    edges_hz = np.geomspace(150, 3600, N_BANDS + 1)
    return np.searchsorted(freqs, edges_hz)


def _log_bands(pcm: np.ndarray) -> np.ndarray:
    """PCM mono 8 kHz -> (trames, 24) log-énergie par bande."""
    if pcm.size < FRAME:
        return np.zeros((0, N_BANDS), np.float32)
    n = 1 + (pcm.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(pcm[idx] * np.hanning(FRAME)[None, :], axis=1)) ** 2
    edges = _band_edges()
    bands = np.stack([spec[:, edges[i]:max(edges[i] + 1, edges[i + 1])].sum(axis=1) for i in range(N_BANDS)], axis=1)
    return np.log10(bands + 1e-6).astype(np.float32)


def _finish(logb: np.ndarray) -> np.ndarray:
    # Variations dans le temps : insensibles au volume et à l'égaliseur de la chaîne de clips.
    return np.diff(logb, axis=0, prepend=logb[:1]).astype(np.float32) if logb.size else logb


def features_from_pcm(pcm: np.ndarray) -> np.ndarray:
    """PCM mono 8 kHz (float) -> empreinte (trames, 24)."""
    return _finish(_log_bands(pcm))


def features_from_source(source: str | Path, *, start: float | None = None, duration: float | None = None,
                         block_frames: int = 8192) -> np.ndarray:
    """Empreinte d'un fichier audio/vidéo, décodé par ffmpeg **en flux** (mémoire constante, même pour 6 h)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ViralError("ffmpeg introuvable : installez-le (voir docs/INSTALLATION.md).")
    cmd = [ff, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
    parts: list[np.ndarray] = []
    carry = np.zeros(0, np.float32)
    block = block_frames * HOP * 2                               # octets lus par tour
    with tempfile.TemporaryFile() as err, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err) as proc:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(block)
            if not raw:
                break
            pcm = np.concatenate([carry, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0])
            n = 0 if pcm.size < FRAME else 1 + (pcm.size - FRAME) // HOP
            if n:
                parts.append(_log_bands(pcm[: (n - 1) * HOP + FRAME]))
            carry = pcm[n * HOP:]
        proc.wait()
        if proc.returncode != 0:
            err.seek(0)
            raise ViralError(f"ffmpeg (audio) : {err.read().decode(errors='replace')[-300:]}")
    return _finish(np.concatenate(parts) if parts else np.zeros((0, N_BANDS), np.float32))


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def correlate(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Corrélation normalisée (≈ cosinus, -1..1) de ``query`` contre chaque position de ``ref`` (trames)."""
    tq, tr = query.shape[0], ref.shape[0]
    if tq == 0 or tr < tq:
        return np.zeros(0)
    q, r = _zscore(query), _zscore(ref)
    n = 1 << int(np.ceil(np.log2(tr + tq)))
    acc = np.zeros(n)
    for b in range(q.shape[1]):
        acc += np.fft.irfft(np.fft.rfft(r[:, b], n) * np.conj(np.fft.rfft(q[:, b], n)), n)
    raw = acc[: tr - tq + 1]
    energy = np.concatenate([[0.0], np.cumsum((r ** 2).sum(axis=1))])
    window = np.sqrt(np.maximum(energy[tq:] - energy[:-tq], 1e-9))[: raw.size]
    return raw / (window * np.sqrt((q ** 2).sum()) + 1e-9)


def locate(short_feat: np.ndarray, vod_feat: np.ndarray, *, chunk_seconds: float = 8.0, chunks: int = 5,
           agree: int = 3, tolerance: float = 1.0, min_corr: float = 0.3) -> tuple[float, float] | None:
    """Début du Short dans la VOD (s) et confiance (corrélation moyenne), ou None.

    ``chunks`` morceaux répartis sur le Short sont cherchés séparément dans la VOD ; au moins ``agree`` doivent
    désigner le même début (± ``tolerance`` s). Un audio sans rapport donne des positions éparpillées : ce
    consensus est ce qui écarte les faux positifs, et il résiste aux intros/outros ajoutées par les chaînes de clips.
    """
    total = short_feat.shape[0] / FPS
    clen = min(chunk_seconds, max(3.0, total / 2))
    n = max(1, min(chunks, int(total // max(1.0, clen / 2))))
    starts = np.linspace(0.0, max(0.0, total - clen), n)
    votes: list[tuple[float, float]] = []
    for s in starts:
        a, b = int(s * FPS), int((s + clen) * FPS)
        corr = correlate(short_feat[a:b], vod_feat)
        if corr.size == 0:
            continue
        k = int(np.argmax(corr))
        if float(corr[k]) >= min_corr:
            votes.append((k / FPS - float(s), float(corr[k])))
    need = min(agree, max(2, n - 1)) if n > 1 else 1
    best: tuple[float, float, int] | None = None
    for off, _ in votes:
        group = [(o, c) for o, c in votes if abs(o - off) <= tolerance]
        if len(group) >= need:
            cand = (float(np.median([o for o, _ in group])), float(np.mean([c for _, c in group])), len(group))
            if best is None or (cand[2], cand[1]) > (best[2], best[1]):
                best = cand
    if best is None:
        return None
    return max(0.0, round(best[0], 3)), round(best[1], 3)


# ---------------------------------------------------------------------------
# yt-dlp : audio léger, extrait vidéo
# ---------------------------------------------------------------------------
def download_audio(url: str, out_dir: Path, name: str) -> Path:
    """Piste audio la plus légère (suffisante pour l'empreinte)."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"format": "wa/ba/worst", "outtmpl": str(out_dir / f"{name}.%(ext)s"), "noplaylist": True,
            "quiet": True, "no_warnings": True, "noprogress": True, "retries": 5}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = [d.get("filepath") for d in (info or {}).get("requested_downloads") or [] if d.get("filepath")]
    found = [Path(f) for f in files if Path(f).is_file()] or sorted(out_dir.glob(f"{name}.*"))
    if not found:
        raise ViralError(f"Audio introuvable pour {url}")
    return found[0]


def download_section(url: str, start: float, end: float, out_dir: Path, name: str, *, max_height: int = 1080) -> Path:
    """Télécharge uniquement [start, end] de la VOD (MP4 vidéo + audio)."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    out_dir.mkdir(parents=True, exist_ok=True)
    h = int(max_height)
    opts = {
        "format": f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b",
        "merge_output_format": "mp4", "outtmpl": str(out_dir / f"{name}.%(ext)s"), "noplaylist": True,
        "quiet": True, "no_warnings": True, "noprogress": True, "retries": 5,
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
        "force_keyframes_at_cuts": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = [d.get("filepath") for d in (info or {}).get("requested_downloads") or [] if d.get("filepath")]
    found = [Path(f) for f in files if Path(f).is_file()] or sorted(out_dir.glob(f"{name}.*"))
    if not found:
        raise ViralError(f"Extrait introuvable pour {url}")
    return found[0]


# ---------------------------------------------------------------------------
# YouTube : Shorts viraux et VODs d'origine
# ---------------------------------------------------------------------------
def _parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def viral_shorts(cfg: Mapping[str, Any], yt: Any, *, now: datetime) -> list[Short]:
    """Shorts GTA 6 des dernières heures, les plus vus d'abord (titre GTA 6, langue visée)."""
    from modules.discover import SourceItem, language_ok, parse_iso8601_duration, title_matches

    dcfg = cfg.get("discover") or {}
    vcfg = get(cfg, "viral", {}) or {}
    query = "|".join(f'"{k}"' for k in dcfg.get("keywords", ["GTA 6"]))
    since = now - timedelta(hours=float(vcfg.get("hours", 24)))
    ids = yt.search_recent(query, video_duration="short", published_after=since, limit=int(vcfg.get("max_shorts", 25)))
    out: list[Short] = []
    for v in yt.videos(ids) if ids else []:
        sn, st = v.get("snippet", {}), v.get("statistics", {})
        title = sn.get("title", "")
        item = SourceItem(url="", title=title, viewers=0, duration=0, streamer="", platform="youtube", status="vod",
                          extra={"language": (sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "")
                                 .split("-")[0].lower() or None})
        if not title_matches(title, dcfg.get("keywords") or ["GTA 6"]) or not language_ok(item, dcfg):
            continue
        out.append(Short(id=v["id"], title=title, channel=sn.get("channelTitle", ""),
                         channel_id=sn.get("channelId", ""), views=int(st.get("viewCount") or 0),
                         duration=parse_iso8601_duration((v.get("contentDetails") or {}).get("duration")),
                         published_at=_parse_dt(sn.get("publishedAt")) or now,
                         description=sn.get("description", ""), tags=list(sn.get("tags") or [])))
    out.sort(key=lambda s: -s.views)
    return out


# Le Short montre-t-il du gameplay ? YouTube ne le dit pas : on lit ce que la vidéo déclare (titre, tags,
# description). Listes modifiables dans config.yaml (viral.gameplay.include / exclude).
GAMEPLAY_INCLUDE = [
    "gameplay", "physics", "crash", "chase", "police", "cops", "wanted", "mission", "heist", "glitch", "bug",
    "stunt", "drift", "jump", "fight", "shootout", "fail", "funny", "moment", "moments", "secret", "easter egg",
    "hidden", "you can", "can you", "play as", "4k", "60fps", "free roam", "open world", "graphics", "realistic",
    "detail", "details", "npc", "npcs", "helicopter", "plane", "boat", "car", "cars", "bike", "tank", "weapon",
    "weapons", "gun", "guns", "money", "robbery", "rob", "lucia", "jason", "vice city", "leonida", "map",
    "wildlife", "alligator", "gator", "beach", "this guy", "never do", "don't do", "devs knew", "attention to detail",
    "rampage", "chaos", "explosion", "escape", "sniper", "5 stars", "wanted level", "destroy", "insane",
]
GAMEPLAY_EXCLUDE = [
    "vs gta", "gta 5 vs", "gta v vs", "gta 6 vs", "gta vi vs", "vs rdr2", "vs rdr", "rdr2 vs", "vs real life",
    "versus", "better than", "compare", "compares", "comparison", "review", "reaction", "react", "reacts",
    "podcast", "interview", "news", "leak", "leaked", "leaks", "trailer", "theory", "theories", "explained",
    "opinion", "rant", "price", "release date", "delay", "delayed", "how will", "will be like", "ranking",
    "tier list", "diy", "paper", "puzzle", "drawing", "lego", "cosplay", "unboxing", "irl", "real life", "ai",
    "ending", "story", "lore", "stream", "streamer", "live", "cried", "crying", "rockstar confirmed",
    "best game ever", "worth it", "should you", "buy",
]


def _has(text: str, words: Sequence[str]) -> list[str]:
    return [w for w in words if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text)]


def gameplay_verdict(short: Short, cfg: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """(gameplay ?, raison). Exclu dès qu'un mot « discussion / réaction / hors-jeu » apparaît dans le titre ou
    les tags ; retenu seulement si un mot « gameplay » apparaît (titre, tags, puis début de la description)."""
    gcfg = get(cfg or {}, "viral.gameplay", {}) or {}
    include = [w.lower() for w in (gcfg.get("include") or GAMEPLAY_INCLUDE)]
    exclude = [w.lower() for w in (gcfg.get("exclude") or GAMEPLAY_EXCLUDE)]
    head = f"{short.title} {' '.join(short.tags)}".lower()
    bad = _has(head, exclude)
    if bad:
        return False, f"pas du gameplay ({', '.join(bad[:3])})"
    good = _has(head, include) or _has(short.description[:300].lower(), include)
    if good:
        return True, f"gameplay ({', '.join(good[:3])})"
    return False, "indéterminé (aucun mot de gameplay)"


def top_shorts(cfg: Mapping[str, Any], yt: Any, *, now: datetime, hours: float = 48, count: int = 10,
               gameplay_only: bool = True) -> tuple[list[tuple[Short, str]], list[tuple[Short, str]]]:
    """Les ``count`` Shorts GTA 6 les plus vus des ``hours`` dernières heures : (retenus, écartés), avec raison."""
    vcfg = {**(get(cfg, "viral", {}) or {}), "hours": hours, "max_shorts": 50}
    shorts = viral_shorts({**cfg, "viral": vcfg}, yt, now=now)
    kept: list[tuple[Short, str]] = []
    dropped: list[tuple[Short, str]] = []
    for sh in shorts:
        ok, why = gameplay_verdict(sh, cfg) if gameplay_only else (True, "")
        (kept if ok else dropped).append((sh, why))
    return kept[:count], dropped


def write_top_shorts(path_: str | Path, kept: Sequence[tuple[Short, str]], dropped: Sequence[tuple[Short, str]], *,
                     hours: float, gameplay_only: bool) -> Path:
    def line(i: int, sh: Short, why: str) -> str:
        views = f"{sh.views:,}".replace(",", " ")
        extra = f" - {why}" if why else ""
        return (f"{i}. {sh.title}\n   {sh.channel} - {views} vues - publié le {sh.published_at:%d/%m à %H:%M}{extra}"
                f"\n   {sh.url}\n")

    kind = " gameplay" if gameplay_only else ""
    parts = [f"Top {len(kept)} Shorts GTA 6{kind} les plus vus ({hours:g} h)\n"]
    parts += [line(i, sh, why) for i, (sh, why) in enumerate(kept, 1)]
    if gameplay_only and dropped:
        parts.append(f"\n--- Écartés ({len(dropped)}) : vérifie qu'aucun gameplay n'a été oublié ---\n")
        parts += [f"- {sh.title} ({sh.channel}, {sh.views:,} vues) : {why}\n   {sh.url}".replace(",", " ")
                  for sh, why in dropped]
    out = Path(path_)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8-sig")
    return out


def candidate_vods(short: Short, yt: Any, *, max_candidates: int = 3,
                   max_age_days: float = 7.0) -> list[tuple[str, str, str]]:
    """VODs d'origine possibles : (url, titre, streamer). Liens de la description, même chaîne, @mentions."""
    from modules.discover import parse_iso8601_duration

    explicit: list[tuple[str, str, str]] = []
    others: list[tuple[str, str, str]] = []
    twitch: list[tuple[str, str, str]] = []
    text = f"{short.title}\n{short.description}"
    linked = [v for v in YT_LINK.findall(text) if v != short.id]
    for tw in TWITCH_LINK.findall(text):
        mention = next(iter(MENTION.findall(text)), short.channel)
        twitch.append((f"https://www.twitch.tv/videos/{tw}", "VOD Twitch", mention))
    channels = [short.channel_id] + [f"@{m}" for m in MENTION.findall(text)[:2]]
    ids = list(linked)
    for ch in channels:
        if not ch:
            continue
        try:
            ids += yt.channel_uploads(ch.lstrip("@") if ch.startswith("@") else ch, limit=10)
        except QuotaExceeded:
            break
    ids = [i for i in dict.fromkeys(ids) if i != short.id]
    details = {v["id"]: v for v in (yt.videos(ids) if ids else [])}
    for vid in ids:
        v = details.get(vid)
        if v is None:
            continue
        sn = v.get("snippet", {})
        dur = parse_iso8601_duration((v.get("contentDetails") or {}).get("duration"))
        published = _parse_dt((v.get("liveStreamingDetails") or {}).get("actualStartTime")) or \
            _parse_dt(sn.get("publishedAt"))
        if dur < 20 * 60 or published is None:
            continue
        explicit_link = v["id"] in linked
        if not explicit_link and not (short.published_at - timedelta(days=max_age_days) <= published
                                 <= short.published_at + timedelta(hours=2)):
            continue
        cand = (f"https://www.youtube.com/watch?v={v['id']}", sn.get("title", ""), sn.get("channelTitle", ""))
        (explicit if explicit_link else others).append(cand)
    # Liens explicites d'abord (ordre de la description), puis Twitch, puis les vidéos des chaînes.
    return list(dict.fromkeys(explicit + twitch + others))[:max_candidates]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clip_bounds(offset: float, short_duration: float, vcfg: Mapping[str, Any]) -> tuple[float, float]:
    """Bornes du clip dans la VOD : un peu de contexte avant, le moment du Short, la réaction après (< 30 s)."""
    before, after = float(vcfg.get("context_before", 3.0)), float(vcfg.get("context_after", 2.5))
    max_len = float(vcfg.get("max_clip_seconds", 29.0))
    core_len = min(short_duration, max_len - before - after)
    start = max(0.0, offset - before)
    return round(start, 2), round(min(start + max_len, offset + core_len + after), 2)


def run_viral(cfg: Mapping[str, Any], drive: Any, *, now: datetime | None = None, service: Any = None,
              count: int | None = None, plan_only: bool = False,
              audio_fn: Callable[[str, Path, str], Path] = download_audio,
              section_fn: Callable[..., Path] = download_section,
              prepare_fn: Callable[..., dict[str, Any]] | None = None,
              on_event: Callable[[str], None] | None = None) -> list[Located]:
    """Localise les ``count`` Shorts les plus vus dans leurs VODs et dépose un job Drive par moment."""
    from modules.discover import YouTubeDiscover, _discover_units_today, _youtube_service

    vcfg = get(cfg, "viral", {}) or {}
    now = now or utc_now()
    count = int(count or vcfg.get("count", 5))
    say = on_event or (lambda msg: log.info("%s", msg))
    service = service or _youtube_service(cfg)
    quota = YouTubeQuota.from_config(cfg)
    budget = min(int(vcfg.get("max_units", 400)),
                 max(0, int(get(cfg, "discover.youtube.daily_budget", 1500)) - _discover_units_today(quota)))
    yt = YouTubeDiscover(service, quota, cfg, budget=budget)
    state = StateStore(path(cfg, "state"))
    seen = state.load().get("viral") or {}
    retry_after = timedelta(hours=float(vcfg.get("retry_failed_after_hours", 72)))

    shorts = viral_shorts(cfg, yt, now=now)
    say(f"{len(shorts)} Short(s) GTA 6 viral(aux) trouvé(s)")
    cache = resolve_path(cfg, get(cfg, "paths.cache", "cache")) / "viral"
    vod_features: dict[str, np.ndarray] = {}
    found: list[Located] = []
    downloads = 0
    with tempfile.TemporaryDirectory(prefix="clipgod_viral_") as tmpd:
        tmp = Path(tmpd)
        for short in shorts:
            if len(found) >= count:
                break
            prev = seen.get(short.id) or {}
            if prev.get("status") in ("prepared", "used"):
                continue
            if prev.get("status") == "failed" and (_parse_dt(prev.get("at")) or now) > now - retry_after:
                continue
            say(f"▶ {short.views:,} vues · {short.channel} · {short.title[:60]}".replace(",", " "))
            try:
                candidates = candidate_vods(short, yt, max_candidates=int(vcfg.get("candidates_per_short", 3)),
                                            max_age_days=float(vcfg.get("max_vod_age_days", 7)))
            except QuotaExceeded as exc:
                say(f"  quota YouTube : {exc}")
                break
            if not candidates:
                say("  aucune VOD d'origine trouvée")
                _remember(state, short, "failed", reason="no_vod")
                continue
            try:
                short_feat = features_from_source(audio_fn(short.url, tmp, f"short_{short.id}"))
            except Exception as exc:  # noqa: BLE001
                say(f"  audio du Short illisible ({exc})")
                continue
            hit: Located | None = None
            for url, title, streamer in candidates:
                if url not in vod_features:
                    if downloads >= int(vcfg.get("max_vod_downloads", 12)):
                        break
                    feat_file = cache / f"{re.sub(r'[^A-Za-z0-9_-]', '_', url.split('/')[-1])}.npy"
                    try:
                        if feat_file.is_file():
                            vod_features[url] = np.load(feat_file)
                        else:
                            downloads += 1
                            say(f"  écoute de la VOD : {title[:50]}")
                            audio = audio_fn(url, tmp, f"vod_{downloads}")
                            vod_features[url] = features_from_source(audio)
                            audio.unlink(missing_ok=True)
                            cache.mkdir(parents=True, exist_ok=True)
                            np.save(feat_file, vod_features[url])
                    except Exception as exc:  # noqa: BLE001
                        say(f"  VOD illisible ({exc})")
                        vod_features[url] = np.zeros((0, N_BANDS), np.float32)
                        continue
                res = locate(short_feat, vod_features[url])
                if res:
                    hit = Located(short, url, title, streamer or short.channel, res[0], res[1])
                    break
            if hit is None:
                say("  moment introuvable dans les VODs candidates")
                _remember(state, short, "failed", reason="no_match")
                continue
            hit.clip_start, hit.clip_end = clip_bounds(hit.offset, short.duration, vcfg)
            m, s = divmod(int(hit.offset), 60)
            say(f"  ✔ trouvé à {m // 60}h{m % 60:02d}m{s:02d}s dans « {hit.vod_title[:40]} » "
                f"(confiance {hit.confidence:.2f})")
            if not plan_only:
                _prepare_segment(cfg, drive, hit, tmp, vcfg, section_fn=section_fn, prepare_fn=prepare_fn, say=say)
                _remember(state, short, "prepared", vod=hit.vod_url, offset=hit.offset, job_id=hit.job_id)
            found.append(hit)
    return found


def _prepare_segment(cfg: Mapping[str, Any], drive: Any, hit: Located, tmp: Path, vcfg: Mapping[str, Any], *,
                     section_fn: Callable[..., Path], prepare_fn: Callable[..., dict[str, Any]] | None,
                     say: Callable[[str], None]) -> None:
    from modules.prepare import prepare

    pad = float(vcfg.get("download_padding", 8.0))
    seg_start = max(0.0, hit.clip_start - pad)
    say("  téléchargement de l'extrait")
    seg = section_fn(hit.vod_url, seg_start, hit.clip_end + pad, tmp, f"viral_{hit.short.id}",
                     max_height=int(get(cfg, "prepare.max_height", 1080)))
    extra = {
        "source_url": hit.vod_url,
        "title": hit.vod_title,
        "fixed_moment": {"start": round(hit.clip_start - seg_start, 2), "end": round(hit.clip_end - seg_start, 2),
                         "score": 90.0},
        "viral": {"short_id": hit.short.id, "short_url": hit.short.url, "views": hit.short.views,
                  "short_channel": hit.short.channel, "short_title": hit.short.title,
                  "vod_offset": round(hit.offset, 2), "confidence": hit.confidence},
    }
    job = (prepare_fn or prepare)(str(seg), cfg, drive, streamer=hit.streamer, download="local",
                                  video_id=f"viral_{re.sub(r'[^A-Za-z0-9_]', '_', hit.short.id)}", extra_job=extra)
    hit.job_id = job.get("job_id")
    seg.unlink(missing_ok=True)


def _remember(state: StateStore, short: Short, status: str, **fields: Any) -> None:
    with state.transaction() as data:
        data.setdefault("viral", {})[short.id] = {"status": status, "views": short.views, "title": short.title,
                                                  "at": utc_now_iso(), **fields}


def viral_order(picks: Sequence[Any]) -> list[Any]:
    """Classement par vues du Short d'origine (rang 1 = le plus vu)."""
    return sorted(picks, key=lambda p: -int(((p.meta.get("viral") or {}).get("views")) or 0))
