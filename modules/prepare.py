"""prepare (laptop) : dépose une source et son job dans Google Drive pour le notebook GPU.

    clipgod prepare --source <url | fichier.mp4> [--streamer X] [--campaign Y] [--voiceover]

Étapes :
    1. source   URL -> métadonnées yt-dlp (titre, chaîne, durée, graphe « most replayed ») ;
                fichier local -> identifiant stable (nom + empreinte).
    2. vidéo    selon ``prepare.download`` :
                  local  : téléchargement sur le laptop (yt-dlp) puis upload dans ``ClipGod/sources/`` ;
                  cloud  : rien à télécharger ici, le notebook récupère la VOD lui-même (bande passante Colab) ;
                  auto   : YouTube -> local (YouTube bloque souvent les IP de datacenter), Twitch/Kick -> cloud.
                Un fichier local est toujours uploadé.
    3. modèle   ``data/model.pkl`` envoyé dans ``ClipGod/model.pkl`` s'il a changé (poids appris par learn).
    4. job      ``ClipGod/jobs/<job_id>.json`` + état local ``data/state.json`` (status queued).

Léger : aucune dépendance GPU/vidéo ; yt-dlp est importé seulement pour une URL.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import get, job_config, path, resolve_path
from core.drive import DriveBackend, done_job_ids, pending_jobs
from core.logger import get_logger
from core.storage import StateStore, file_sha256, new_job_id, slugify, utc_now_iso

log = get_logger(__name__)

JOB_FORMAT = "clipgod-job/1"
MODEL_REMOTE = "model.pkl"
ProgressCallback = Callable[[int, int], None]


class PrepareError(RuntimeError):
    """Source introuvable, URL non supportée, ffmpeg manquant pour un téléchargement local…"""


@dataclass
class Source:
    kind: str                      # "url" | "file"
    platform: str                  # youtube | twitch | kick | web | local
    video_id: str
    url: str | None = None
    local_path: Path | None = None
    title: str | None = None
    streamer: str | None = None
    duration: float | None = None
    heatmap: list[dict[str, float]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Résolution de la source
# ---------------------------------------------------------------------------
def detect_platform(url: str) -> str:
    u = url.lower()
    if "youtube.com/" in u or "youtu.be/" in u:
        return "youtube"
    if "twitch.tv/" in u:
        return "twitch"
    if "kick.com/" in u:
        return "kick"
    return "web"


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:80] or "video"


def probe_url(url: str) -> dict[str, Any]:
    """Métadonnées yt-dlp sans téléchargement (inclut ``heatmap`` pour YouTube)."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise PrepareError("yt-dlp n'est pas installé (pip install yt-dlp).") from exc
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise PrepareError(f"URL illisible par yt-dlp : {url} ({exc})") from exc
    if not info:
        raise PrepareError(f"Aucune vidéo trouvée pour {url}")
    if info.get("_type") == "playlist":
        raise PrepareError("URL de playlist/chaîne : donnez l'URL d'une VOD précise.")
    return info


def resolve_source(source: str, *, streamer: str | None = None,
                   probe: Callable[[str], dict[str, Any]] = probe_url) -> Source:
    """URL (http/https) ou chemin de fichier local -> :class:`Source`."""
    if re.match(r"^https?://", source, re.IGNORECASE):
        platform = detect_platform(source)
        info = probe(source)
        if info.get("is_live") or info.get("live_status") == "is_live":
            raise PrepareError("Stream en direct : utilisez `clipgod watch` ou attendez la VOD.")
        return Source(
            kind="url", platform=platform, url=info.get("webpage_url") or source,
            video_id=f"{platform}_{_safe_id(str(info.get('id') or slugify(source)))}",
            title=info.get("title"),
            streamer=streamer or info.get("uploader_id") or info.get("channel") or info.get("uploader"),
            duration=float(info["duration"]) if info.get("duration") else None,
            heatmap=info.get("heatmap") or None,
            extra={"upload_date": info.get("upload_date"), "view_count": info.get("view_count")},
        )
    p = Path(source).expanduser()
    if not p.is_file():
        raise PrepareError(f"Fichier introuvable : {p}")
    with open(p, "rb") as fh:  # empreinte rapide : taille + premier Mo (pas de lecture complète de 5 Go)
        digest = hashlib.sha256(fh.read(1024 * 1024) + str(p.stat().st_size).encode()).hexdigest()[:8]
    return Source(kind="file", platform="local", local_path=p, title=p.stem, streamer=streamer,
                  video_id=f"local_{slugify(p.stem, max_len=50).replace('-', '_')}_{digest}")


def choose_download_mode(src: Source, setting: str) -> str:
    if src.kind == "file":
        return "local"
    if setting in ("local", "cloud"):
        return setting
    return "local" if src.platform == "youtube" else "cloud"


# ---------------------------------------------------------------------------
# Téléchargement (laptop ou notebook)
# ---------------------------------------------------------------------------
def download_url(url: str, out_dir: str | Path, video_id: str, *, max_height: int = 1080,
                 progress: ProgressCallback | None = None) -> Path:
    """Télécharge une VOD en MP4 (vidéo <= ``max_height`` + audio) avec yt-dlp. Utilisé aussi par le worker."""
    import yt_dlp

    if not shutil.which("ffmpeg"):
        raise PrepareError(
            "ffmpeg est requis pour assembler vidéo + audio. Windows : winget install Gyan.FFmpeg ; "
            "macOS : brew install ffmpeg ; Linux : apt install ffmpeg. "
            "Ou utilisez prepare.download: cloud (téléchargement par le notebook)."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / f"{video_id}.mp4"
    if existing.is_file() and existing.stat().st_size > 0:
        log.info("Déjà téléchargé : %s", existing)
        return existing

    def hook(d: dict[str, Any]) -> None:
        if progress and d.get("status") == "downloading":
            progress(int(d.get("downloaded_bytes") or 0), int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0))

    h = int(max_height)
    opts = {
        "format": f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out / f"{video_id}.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 10, "fragment_retries": 10, "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise PrepareError(f"Téléchargement impossible : {exc}") from exc
    files = [d.get("filepath") for d in (info or {}).get("requested_downloads") or [] if d.get("filepath")]
    candidates = [Path(f) for f in files if Path(f).is_file()] or sorted(out.glob(f"{video_id}.*"))
    if not candidates:
        raise PrepareError("yt-dlp n'a produit aucun fichier.")
    return candidates[0]  # .mp4 en général ; ffmpeg lit de toute façon n'importe quel conteneur


# ---------------------------------------------------------------------------
# Modèle appris
# ---------------------------------------------------------------------------
def push_model(cfg: Mapping[str, Any], drive: DriveBackend, state: StateStore) -> bool:
    """Envoie data/model.pkl dans ClipGod/model.pkl s'il a changé depuis le dernier envoi."""
    local = path(cfg, "model")
    if not local.is_file():
        return False
    digest = file_sha256(local)
    meta = state.load().get("model", {})
    if meta.get("sha256") == digest and drive.exists(MODEL_REMOTE):
        return False
    drive.upload_file(local, MODEL_REMOTE)
    with state.transaction() as data:
        data["model"] = {"sha256": digest, "pushed_at": utc_now_iso()}
    log.info("model.pkl envoyé sur Drive (nouveaux poids pour le prochain scan).")
    return True


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def find_existing_job(drive: DriveBackend, video_id: str) -> tuple[str, str] | None:
    """(job_id, statut) d'un job déjà déposé pour cette vidéo, sinon None."""
    for job in pending_jobs(drive):
        if job.get("video_id") == video_id:
            return str(job["job_id"]), "en attente"
    done = done_job_ids(drive)
    for f in drive.list_dir(drive.folders["jobs"], "*.json"):
        if f.stem in done:
            job = drive.read_json(f.path, default={})
            if isinstance(job, dict) and job.get("video_id") == video_id:
                return f.stem, "terminé"
    return None


def prepare(
    source: str,
    cfg: Mapping[str, Any],
    drive: DriveBackend,
    *,
    streamer: str | None = None,
    campaign: str | None = None,
    voiceover: bool = False,
    job_id: str | None = None,
    download: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    probe: Callable[[str], dict[str, Any]] = probe_url,
    downloader: Callable[..., Path] = download_url,
) -> dict[str, Any]:
    """Prépare un job et le dépose sur Drive. Retourne le job écrit (avec ``_existing`` si dédoublonné)."""
    pcfg = cfg.get("prepare") or {}
    state = StateStore(path(cfg, "state"))
    src = resolve_source(source, streamer=streamer, probe=probe)
    log.info("Source : %s (%s) %s", src.video_id, src.platform, src.title or "")

    drive.ensure_structure()
    if pcfg.get("dedupe", True) and not force:
        existing = find_existing_job(drive, src.video_id)
        if existing:
            log.warning("Un job existe déjà pour %s (%s, %s) : utilisez --force pour en créer un autre.",
                        src.video_id, existing[0], existing[1])
            return {"job_id": existing[0], "video_id": src.video_id, "_existing": existing[1]}

    mode = choose_download_mode(src, download or str(pcfg.get("download", "auto")))
    source_drive_path = None
    local_download: Path | None = None
    if mode == "local":
        if src.kind == "url":
            assert src.url
            local_download = downloader(src.url, resolve_path(cfg, pcfg.get("download_dir", "cache/downloads")),
                                        src.video_id, max_height=int(pcfg.get("max_height", 1080)), progress=progress)
            video_file = local_download
        else:
            assert src.local_path
            video_file = src.local_path
        remote = drive.remote("sources", f"{src.video_id}{video_file.suffix.lower() or '.mp4'}")
        info = drive.stat(remote)
        if info is not None and info.size == video_file.stat().st_size:
            log.info("Source déjà présente sur Drive : %s", drive.display(remote))
        else:
            drive.upload_file(video_file, remote, progress=progress)
        source_drive_path = drive.display(remote)

    model_pushed = push_model(cfg, drive, state)
    jid = job_id or new_job_id()
    job: dict[str, Any] = {
        "format": JOB_FORMAT,
        "job_id": jid,
        "video_id": src.video_id,
        "platform": src.platform,
        "source_url": src.url,
        "source_drive_path": source_drive_path,
        "download": mode,
        "title": src.title,
        "streamer": src.streamer,
        "duration": src.duration,
        "campaign": campaign,
        "heatmap": src.heatmap,
        "options": {"voiceover": bool(voiceover)},
        "created_at": utc_now_iso(),
        "created_by": "clipgod prepare",
        "config": job_config(cfg),
    }
    drive.write_json(drive.remote("jobs", f"{jid}.json"), job)
    state.upsert_job(jid, status="queued", video_id=src.video_id, platform=src.platform, source=source,
                     streamer=src.streamer, campaign=campaign, download=mode, model_pushed=model_pushed,
                     source_duration=src.duration)
    if local_download is not None and not pcfg.get("keep_local_download", False):
        local_download.unlink(missing_ok=True)
    log.info("Job %s déposé dans %s", jid, drive.display(drive.remote("jobs", f"{jid}.json")))
    return job


def colab_url(cfg: Mapping[str, Any]) -> str:
    return str(get(cfg, "prepare.colab_url", "https://colab.research.google.com/github/moonkamel/clipgod/blob/main/notebooks/clipgod_colab.ipynb"))
