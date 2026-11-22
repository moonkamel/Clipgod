"""Worker cloud : traite la file de jobs Drive (``ClipGod/jobs/``) — appelé par les notebooks Colab/Kaggle.

Toute la logique du notebook vit ici (testable, versionnée) ; le notebook se contente de :
    drive = get_drive(cfg); run_queue(drive, cfg, stages=("scan", "render"))

Cycle d'un job :
    1. réservation  : ``jobs/<job_id>.lock`` (bail de ``lease_hours``) — évite que Colab et Kaggle prennent le
                      même job. Best effort : Drive n'offre pas de verrou atomique.
    2. config       : config.yaml du repo  <-  ``job["config"]`` (réglages du laptop)  <-  overrides du notebook
                      + ``ClipGod/model.pkl`` s'il existe (poids appris par ``clipgod learn``).
    3. source       : ``ClipGod/sources/...`` copiée sur le disque local du runtime (lectures multiples rapides).
    4. scan         : ``modules.scan.scan`` -> ``ClipGod/moments/<video_id>.json`` (+ transcript).
    5. render       : ``modules.render.render`` -> ``ClipGod/clips/<video_id>_<i>.mp4`` (étape 5).
    6. fin          : ``ClipGod/jobs_done/<job_id>.json`` (status done | failed) et suppression du lock.

Un job en échec reste en file et sera retenté au prochain run, jusqu'à ``max_attempts`` ; ensuite il est marqué
``failed`` dans ``jobs_done/`` avec l'erreur (visible par ``clipgod sync --status``).

Les checkpoints du scan sont placés sur Drive quand il est monté (Colab) : une déconnexion en pleine
transcription ne fait pas perdre le travail déjà fait.
"""

from __future__ import annotations

import os
import socket
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.config import deep_merge, detect_runtime
from core.drive import DriveBackend, DriveError, LocalDriveBackend, mark_job_done, pending_jobs
from core.logger import get_logger, log_step
from core.storage import utc_now, utc_now_iso

log = get_logger(__name__)

MODEL_REMOTE = "model.pkl"          # ClipGod/model.pkl, déposé par le laptop (clipgod learn / prepare)
VALID_STAGES = ("scan", "render")


class JobError(RuntimeError):
    """Job invalide (champ manquant, source absente…) : inutile de le retenter."""


# ---------------------------------------------------------------------------
# Réservation (lock)
# ---------------------------------------------------------------------------
def worker_id() -> str:
    return f"{detect_runtime()}-{socket.gethostname()}-{os.getpid()}"


def _lock_path(drive: DriveBackend, job_id: str) -> str:
    return drive.remote("jobs", f"{job_id}.lock")


def _parse_ts(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def read_lock(drive: DriveBackend, job_id: str) -> dict[str, Any]:
    lock = drive.read_json(_lock_path(drive, job_id), default={})
    return lock if isinstance(lock, dict) else {}


def claim_job(drive: DriveBackend, job_id: str, me: str, *, lease_hours: float = 3.0) -> dict[str, Any] | None:
    """Réserve un job. Retourne le lock écrit, ou None si un autre worker détient un bail encore valide."""
    lock = read_lock(drive, job_id)
    expires = _parse_ts(lock.get("expires_at"))
    if lock.get("worker") not in (None, me) and expires and expires > utc_now():
        log.info("Job %s déjà pris par %s jusqu'à %s : ignoré.", job_id, lock["worker"], lock["expires_at"])
        return None
    new_lock = {
        "job_id": job_id,
        "worker": me,
        "claimed_at": utc_now_iso(),
        "expires_at": (utc_now() + timedelta(hours=lease_hours)).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "attempts": int(lock.get("attempts", 0)) + 1,
        "last_error": lock.get("last_error"),
    }
    drive.write_json(_lock_path(drive, job_id), new_lock)
    return new_lock


def release_job(drive: DriveBackend, job_id: str, *, error: str | None = None) -> None:
    """Succès : supprime le lock. Échec : garde le compteur d'essais et l'erreur, libère le bail."""
    if error is None:
        drive.delete(_lock_path(drive, job_id))
        return
    lock = read_lock(drive, job_id)
    lock.update({"worker": None, "expires_at": utc_now_iso(), "last_error": error[-2000:]})
    drive.write_json(_lock_path(drive, job_id), lock)


# ---------------------------------------------------------------------------
# Préparation d'un job
# ---------------------------------------------------------------------------
def validate_job(job: Mapping[str, Any]) -> None:
    missing = [k for k in ("job_id", "video_id") if not job.get(k)]
    if not job.get("source_drive_path") and not job.get("source_url"):
        missing.append("source_drive_path|source_url")
    if missing:
        raise JobError(f"Job invalide, champs manquants : {', '.join(missing)}")


def build_job_config(base_config: Mapping[str, Any], job: Mapping[str, Any],
                     overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """config.yaml du repo, écrasée par la config embarquée dans le job, puis par les overrides du notebook."""
    cfg = deep_merge(dict(base_config), job.get("config") or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["job"] = {k: v for k, v in job.items() if k != "config" and not k.startswith("_")}
    options = job.get("options") or {}
    if "voiceover" in options:
        cfg.setdefault("render", {}).setdefault("voiceover", {})["enabled"] = bool(options["voiceover"])
    return cfg


def fetch_model(drive: DriveBackend, local_dir: Path) -> Path | None:
    """Télécharge ``ClipGod/model.pkl`` (poids appris) s'il existe."""
    if not drive.exists(MODEL_REMOTE):
        return None
    return drive.download_file(MODEL_REMOTE, local_dir / "model.pkl")


def fetch_source(drive: DriveBackend, job: Mapping[str, Any], local_dir: Path, *, max_height: int = 1080) -> Path:
    """Copie la vidéo source en local (réutilisée si déjà présente avec la même taille).

    Job sans ``source_drive_path`` (``download: cloud``) : la VOD est téléchargée ici avec yt-dlp, ce qui
    épargne la connexion du laptop.
    """
    if not job.get("source_drive_path"):
        from modules.prepare import PrepareError, download_url

        log.info("Téléchargement de la VOD dans le cloud : %s", job["source_url"])
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if total and time.perf_counter() - last[0] > 15:
                last[0] = time.perf_counter()
                log.info("téléchargement VOD %d%% (%.0f Mo)", 100 * done / total, done / 1e6)

        try:
            return download_url(str(job["source_url"]), local_dir, str(job["video_id"]), max_height=max_height,
                                progress=progress)
        except PrepareError as exc:
            raise JobError(f"{exc} — si YouTube bloque Colab, relancez prepare avec --download local.") from exc
    remote = str(job["source_drive_path"])
    info = drive.stat(remote)
    if info is None or info.is_folder:
        raise JobError(f"Source absente sur Drive : {drive.display(remote)} (relancez clipgod prepare).")
    target = local_dir / Path(remote).name
    if target.is_file() and info.size is not None and target.stat().st_size == info.size:
        log.info("Source déjà présente en local : %s", target)
        return target
    started = time.perf_counter()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if total and time.perf_counter() - last[0] > 15:
            last[0] = time.perf_counter()
            log.info("téléchargement source %d%% (%.0f Mo)", 100 * done / total, done / 1e6)

    drive.download_file(remote, target, progress=progress)
    log.info("Source copiée en %.0fs (%.0f Mo)", time.perf_counter() - started, target.stat().st_size / 1e6)
    return target


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def _default_scan(video: str, out: str, cfg: dict) -> list[dict]:
    from modules.scan import scan

    return scan(video, out, cfg)


def _default_render(moments: list[dict], video: str, out: str, cfg: dict) -> list[str]:
    from modules.render import render

    return render(moments, video, out, cfg)


def process_job(
    job: Mapping[str, Any],
    drive: DriveBackend,
    base_config: Mapping[str, Any],
    *,
    workdir: str | os.PathLike,
    stages: Sequence[str] = VALID_STAGES,
    overrides: Mapping[str, Any] | None = None,
    scan_fn: Callable[[str, str, dict], list[dict]] = _default_scan,
    render_fn: Callable[[list[dict], str, str, dict], list[str]] = _default_render,
    keep_source: bool = False,
) -> dict[str, Any]:
    """Traite un job déjà réservé et retourne le payload écrit dans ``jobs_done/``."""
    validate_job(job)
    job_id, video_id = str(job["job_id"]), str(job["video_id"])
    root = Path(workdir) / job_id
    local_moments, local_clips = root / "moments", root / "clips"
    for d in (local_moments, local_clips):
        d.mkdir(parents=True, exist_ok=True)

    cfg = build_job_config(base_config, job, overrides)
    scan_cfg = cfg.setdefault("scan", {})
    # (config.yaml contient work_dir: null -> on teste la valeur, pas la présence de la clé)
    if not scan_cfg.get("work_dir"):
        if isinstance(drive, LocalDriveBackend):  # Colab : checkpoints sur Drive -> survivent à une déconnexion
            scan_cfg["work_dir"] = str(drive.root / drive.folders["moments"] / ".work")
        else:
            scan_cfg["work_dir"] = str(root / "work")
    if not scan_cfg.get("tmp_dir"):
        scan_cfg["tmp_dir"] = str(root / "tmp")
    model = fetch_model(drive, root)
    if model is not None:
        scan_cfg["model_path"] = str(model)

    started = time.perf_counter()
    source = fetch_source(drive, job, Path(workdir) / "sources",
                          max_height=int((cfg.get("prepare") or {}).get("max_height", 1080)))
    result: dict[str, Any] = {"video_id": video_id, "worker": worker_id(), "stages": [], "clips": [],
                              "model": "model.pkl" if model else "default_weights"}

    moments: list[dict] = []
    if "scan" in stages:
        with log_step(log, "job.scan", job_id=job_id, video_id=video_id):
            moments = scan_fn(str(source), str(local_moments), cfg)
        for name in (f"{video_id}.json", f"{video_id}.transcript.json"):
            if (local_moments / name).is_file():
                drive.upload_file(local_moments / name, drive.remote("moments", name))
        result.update({"stages": ["scan"], "moments": drive.remote("moments", f"{video_id}.json"),
                       "n_moments": len(moments), "best_score": moments[0]["score"] if moments else None})

    if "render" in stages:
        if not moments:
            moments_doc = drive.read_json(drive.remote("moments", f"{video_id}.json"))
            moments = list(moments_doc.get("moments", []))
        with log_step(log, "job.render", job_id=job_id, video_id=video_id, moments=len(moments)):
            clips = render_fn(moments, str(source), str(local_clips), cfg) if moments else []
        for clip in clips:
            remote = drive.remote("clips", Path(clip).name)
            drive.upload_file(clip, remote)
            result["clips"].append(remote)
            sidecar = Path(clip).with_suffix(".json")       # métadonnées lues par monetize / publish
            if sidecar.is_file():
                drive.upload_file(sidecar, drive.remote("clips", sidecar.name))
        result["stages"].append("render")

    result["duration_s"] = round(time.perf_counter() - started, 1)
    if not keep_source:
        source.unlink(missing_ok=True)
    return result


def run_queue(
    drive: DriveBackend,
    base_config: Mapping[str, Any],
    *,
    workdir: str | os.PathLike = "/content/clipgod_work",
    stages: Sequence[str] = VALID_STAGES,
    max_jobs: int | None = None,
    only_job: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    lease_hours: float = 3.0,
    max_attempts: int = 2,
    **process_kwargs: Any,
) -> list[dict[str, Any]]:
    """Traite les jobs en attente (du plus ancien au plus récent). Retourne un résumé par job tenté.

    Un échec n'arrête pas la file : le job suivant est traité.
    """
    unknown = set(stages) - set(VALID_STAGES)
    if unknown:
        raise ValueError(f"Étapes inconnues : {sorted(unknown)} (valides : {VALID_STAGES})")
    drive.ensure_structure()
    me = worker_id()
    summaries: list[dict[str, Any]] = []
    jobs = pending_jobs(drive)
    if only_job:
        jobs = [j for j in jobs if j["job_id"] == only_job]
    log.info("%d job(s) en attente", len(jobs), extra={"worker": me})

    for job in jobs:
        if max_jobs is not None and len(summaries) >= max_jobs:
            break
        job_id = str(job["job_id"])
        lock = claim_job(drive, job_id, me, lease_hours=lease_hours)
        if lock is None:
            continue
        try:
            payload = process_job(job, drive, base_config, stages=stages, workdir=workdir,
                                  overrides=overrides, **process_kwargs)
        except NotImplementedError:
            # Étape pas encore disponible : erreur de configuration du notebook, pas du job.
            lock = read_lock(drive, job_id)
            lock.update({"worker": None, "expires_at": utc_now_iso(), "attempts": max(0, lock.get("attempts", 1) - 1)})
            drive.write_json(_lock_path(drive, job_id), lock)
            raise
        except Exception as exc:  # noqa: BLE001 — on consigne et on passe au job suivant
            err = f"{type(exc).__name__}: {exc}"
            log.error("Job %s en échec (essai %d/%d) : %s", job_id, lock["attempts"], max_attempts, err,
                      extra={"job_id": job_id, "trace": traceback.format_exc()[-4000:]})
            final = isinstance(exc, JobError) or lock["attempts"] >= max_attempts
            try:
                if final:
                    mark_job_done(drive, job_id, {"status": "failed", "error": err, "attempts": lock["attempts"],
                                                  "video_id": job.get("video_id"), "worker": me})
                    release_job(drive, job_id)
                else:
                    release_job(drive, job_id, error=err)
            except DriveError as drive_exc:
                log.error("Impossible d'enregistrer l'échec du job %s sur Drive : %s", job_id, drive_exc)
            summaries.append({"job_id": job_id, "status": "failed" if final else "retry", "error": err})
            continue
        mark_job_done(drive, job_id, {**payload, "status": "done"})
        release_job(drive, job_id)
        summaries.append({"job_id": job_id, "status": "done", **payload})
        log.info("✔ job %s terminé : %s moment(s), %d clip(s)", job_id, payload.get("n_moments", "?"),
                 len(payload.get("clips", [])))
    return summaries


def queue_status(drive: DriveBackend) -> list[dict[str, Any]]:
    """Vue d'ensemble pour le notebook : jobs en attente (+ lock éventuel)."""
    rows = []
    for job in pending_jobs(drive):
        lock = read_lock(drive, job["job_id"])
        rows.append({"job_id": job["job_id"], "video_id": job.get("video_id"), "streamer": job.get("streamer"),
                     "created_at": job.get("created_at"), "attempts": lock.get("attempts", 0),
                     "locked_by": lock.get("worker"), "last_error": lock.get("last_error")})
    return rows
