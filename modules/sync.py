"""sync (laptop) : rapatrie les résultats du notebook depuis Google Drive.

    clipgod sync            télécharge clips + métadonnées + moments des jobs terminés -> output/
    clipgod sync --status   affiche l'état de la file sans rien télécharger

Pour chaque job terminé (``ClipGod/jobs_done/<job_id>.json``) :
    - ``status: done``   -> clips ``.mp4`` + sidecars ``.json`` -> ``output/clips/`` ; moments -> ``output/moments/`` ;
                            état local : job ``synced``, chaque clip ``synced`` (prêt pour monetize/publish) ;
                            option ``sync.cleanup_sources`` : la vidéo source est retirée de Drive (quota 15 Go).
    - ``status: failed`` -> erreur affichée et consignée dans l'état local (job ``failed``).
Idempotent : un fichier déjà présent avec la même taille n'est pas retéléchargé.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import path, resolve_path
from core.drive import DriveBackend, DriveError
from core.logger import get_logger
from core.storage import StateStore, read_json, utc_now, utc_now_iso

log = get_logger(__name__)


@dataclass
class JobStatus:
    job_id: str
    video_id: str | None
    status: str                       # queued | running | retry | done | failed | synced
    streamer: str | None = None
    created_at: str | None = None
    detail: str = ""
    clips: list[str] = field(default_factory=list)


def _parse(ts: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def queue_overview(drive: DriveBackend, state: StateStore) -> list[JobStatus]:
    """Vue d'ensemble : jobs Drive (en attente, en cours, terminés, échoués) croisés avec l'état local."""
    local_jobs = state.jobs()
    done_files = {f.stem: f for f in drive.list_dir(drive.folders["jobs_done"], "*.json")}
    rows: list[JobStatus] = []
    for f in drive.list_dir(drive.folders["jobs"], "*.json"):
        job = drive.read_json(f.path, default={})
        if not isinstance(job, dict):
            continue
        jid = str(job.get("job_id") or f.stem)
        row = JobStatus(jid, job.get("video_id"), "queued", job.get("streamer"), job.get("created_at"))
        if jid in done_files:
            done = drive.read_json(done_files[jid].path, default={}) or {}
            row.clips = list(done.get("clips") or [])
            if done.get("status") == "failed":
                row.status, row.detail = "failed", str(done.get("error", ""))[:200]
            else:
                row.status = "synced" if local_jobs.get(jid, {}).get("status") == "synced" else "done"
                row.detail = f"{done.get('n_moments', '?')} moment(s), {len(row.clips)} clip(s)"
        else:
            lock = drive.read_json(drive.remote("jobs", f"{jid}.lock"), default={}) or {}
            expires = _parse(lock.get("expires_at"))
            if lock.get("worker") and expires and expires > utc_now():
                row.status, row.detail = "running", f"pris par {lock['worker']}"
            elif lock.get("last_error"):
                row.status = "retry"
                row.detail = f"essai {lock.get('attempts', 1)} échoué : {str(lock['last_error'])[:150]}"
            else:
                row.detail = "ouvrez le notebook Colab"
        rows.append(row)
    return sorted(rows, key=lambda r: (r.created_at or "", r.job_id))


def _download_if_needed(drive: DriveBackend, remote: str, local: Path) -> bool:
    """True si téléchargé, False si déjà présent (même taille)."""
    info = drive.stat(remote)
    if info is None:
        raise DriveError(f"Absent sur Drive : {drive.display(remote)}")
    if local.is_file() and info.size is not None and local.stat().st_size == info.size:
        return False
    drive.download_file(remote, local)
    return True


@dataclass
class SyncReport:
    synced_jobs: list[str] = field(default_factory=list)
    failed_jobs: list[tuple[str, str]] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    skipped: int = 0
    cleaned_sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def sync(cfg: Mapping[str, Any], drive: DriveBackend, *, job_id: str | None = None,
         redownload: bool = False) -> SyncReport:
    """Télécharge les résultats des jobs terminés et met à jour ``data/state.json``."""
    scfg = cfg.get("sync") or {}
    clips_dir = resolve_path(cfg, scfg.get("clips_dir", "output/clips"))
    moments_dir = resolve_path(cfg, scfg.get("moments_dir", "output/moments"))
    state = StateStore(path(cfg, "state"))
    report = SyncReport()

    for row in queue_overview(drive, state):
        if job_id and row.job_id != job_id:
            continue
        if row.status == "failed":
            if (state.get_job(row.job_id) or {}).get("status") != "failed":
                state.upsert_job(row.job_id, status="failed", video_id=row.video_id, error=row.detail)
            report.failed_jobs.append((row.job_id, row.detail))
            continue
        if row.status not in ("done", "synced") or (row.status == "synced" and not redownload):
            continue
        try:
            _sync_job(drive, row, clips_dir, moments_dir, state, report, cleanup=bool(scfg.get("cleanup_sources", True)))
            report.synced_jobs.append(row.job_id)
        except DriveError as exc:
            report.errors.append(f"{row.job_id}: {exc}")
            log.error("Synchronisation du job %s incomplète : %s", row.job_id, exc)
    return report


def _sync_job(drive: DriveBackend, row: JobStatus, clips_dir: Path, moments_dir: Path, state: StateStore,
              report: SyncReport, *, cleanup: bool) -> None:
    done = drive.read_json(drive.remote("jobs_done", f"{row.job_id}.json"))
    job = drive.read_json(drive.remote("jobs", f"{row.job_id}.json"), default={}) or {}
    video_id = str(done.get("video_id") or row.video_id)

    moments_remote = done.get("moments") or drive.remote("moments", f"{video_id}.json")
    if drive.exists(moments_remote):
        target = moments_dir / Path(moments_remote).name
        if _download_if_needed(drive, moments_remote, target):
            report.downloaded.append(str(target))

    for clip_remote in done.get("clips") or []:
        mp4 = clips_dir / Path(clip_remote).name
        sidecar_remote = str(Path(clip_remote).with_suffix(".json"))
        sidecar = mp4.with_suffix(".json")
        for remote, local in ((clip_remote, mp4), (sidecar_remote, sidecar)):
            if remote == sidecar_remote and not drive.exists(remote):
                continue
            if _download_if_needed(drive, remote, local):
                report.downloaded.append(str(local))
            else:
                report.skipped += 1
        meta = read_json(sidecar, default={}) if sidecar.is_file() else {}
        existing = state.get_clip(mp4.stem) or {}
        fields = {
            "path": str(mp4), "job_id": row.job_id, "video_id": video_id,
            "hook": meta.get("hook"), "variant": meta.get("variant", 0), "moment_idx": meta.get("moment_idx"),
            "duration": meta.get("duration"), "score": meta.get("score"), "streamer": meta.get("streamer"),
            "campaign": job.get("campaign"), "synced_at": utc_now_iso(),
        }
        if existing.get("status") in (None, "synced"):
            fields["status"] = "synced"           # ne rétrograde pas un clip déjà validé / publié
        state.upsert_clip(mp4.stem, **fields)

    state.upsert_job(row.job_id, status="synced", video_id=video_id, clips=len(done.get("clips") or []),
                     synced_at=utc_now_iso(), **({"source_duration": job["duration"]} if job.get("duration") else {}))
    if cleanup:
        src = job.get("source_drive_path")
        if src and drive.exists(src):
            drive.delete(src)
            report.cleaned_sources.append(drive.display(src))
            log.info("Source retirée de Drive : %s", drive.display(src))
    log.info("✔ job %s synchronisé (%d clip(s))", row.job_id, len(done.get("clips") or []))
