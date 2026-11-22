"""Fichiers intermédiaires, état local et checkpoints de reprise.

- Écritures JSON **atomiques** (fichier temporaire + ``os.replace``) : un crash ou une déconnexion
  Colab ne laisse jamais un JSON à moitié écrit.
- :class:`StateStore` : état local du laptop (``data/state.json``) — jobs envoyés, clips synchronisés/publiés.
- :class:`Checkpoint` : étapes terminées d'un run, pour relancer un pipeline interrompu sans tout refaire.
  Dans le notebook, le dossier de checkpoints est placé sur Drive pour survivre à une déconnexion.

Stdlib uniquement : importable depuis le CLI et depuis le notebook.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from core.logger import get_logger

log = get_logger(__name__)
T = TypeVar("T")
_MISSING: Any = object()


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def new_job_id(now: datetime | None = None) -> str:
    """Identifiant triable chronologiquement : ``20261122-213045-a1b2c3``."""
    now = now or utc_now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def slugify(text: str, max_len: int = 60) -> str:
    """``"GTA 6 — Course poursuite !"`` -> ``"gta-6-course-poursuite"`` (sûr pour un nom de fichier)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-") or "untitled"


def file_sha256(path: str | os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_project_dirs(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Crée data/, inbox/, output/, cache/, cache/buffer/, logs/ sous la racine du projet."""
    from core.config import ConfigError
    from core.config import path as cfg_path

    created: dict[str, Path] = {}
    for name in ("data", "inbox", "output", "cache", "logs"):
        try:
            created[name] = ensure_dir(cfg_path(cfg, name))
        except ConfigError:  # clé absente de paths : on ignore
            continue
    if "cache" in created:
        created["buffer"] = ensure_dir(created["cache"] / "buffer")
        created["checkpoints"] = ensure_dir(created["cache"] / "checkpoints")
    return created


# ---------------------------------------------------------------------------
# I/O atomiques
# ---------------------------------------------------------------------------
def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def write_json(path: str | os.PathLike, data: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | os.PathLike, default: Any = _MISSING) -> Any:
    """Lit un JSON. Avec ``default`` : fichier absent ou corrompu -> ``default`` (le fichier corrompu
    est mis de côté en ``<nom>.corrupt-<horodatage>`` pour diagnostic). Sans ``default`` : lève."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is _MISSING:
            raise
        return default
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if default is _MISSING:
            raise
        backup = p.with_name(f"{p.name}.corrupt-{utc_now():%Y%m%d%H%M%S}")
        try:
            os.replace(p, backup)
        except OSError:
            backup = p
        log.warning("JSON corrompu ignoré : %s (%s) — sauvegardé sous %s", p, exc, backup.name)
        return default


# ---------------------------------------------------------------------------
# Stores JSON
# ---------------------------------------------------------------------------
class JsonStore:
    """Document JSON persistant avec transaction lecture-modification-écriture atomique."""

    def __init__(self, path: str | os.PathLike, default_factory: Callable[[], dict[str, Any]] = dict):
        self.path = Path(path)
        self._default_factory = default_factory

    def load(self) -> dict[str, Any]:
        data = read_json(self.path, default=None)
        if not isinstance(data, dict):
            return self._default_factory()
        base = self._default_factory()
        base.update(data)
        return base

    def save(self, data: Mapping[str, Any]) -> None:
        write_json(self.path, dict(data))

    @contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        """``with store.transaction() as data: data["x"] = 1`` — sauvegardé seulement si aucun exception."""
        data = self.load()
        yield data
        self.save(data)


def _empty_state() -> dict[str, Any]:
    return {"version": 1, "jobs": {}, "clips": {}}


class StateStore(JsonStore):
    """État local (``data/state.json``).

    ``jobs[job_id]``  : status (``queued`` | ``done`` | ``synced`` | ``failed``), video_id, source, dates…
    ``clips[clip_id]``: status (``synced`` | ``validated`` | ``rejected`` | ``scheduled`` | ``published``), chemins…
    """

    def __init__(self, path: str | os.PathLike):
        super().__init__(path, _empty_state)

    def _upsert(self, kind: str, key: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction() as data:
            entry = data[kind].setdefault(key, {"created_at": utc_now_iso()})
            entry.update(fields)
            entry["updated_at"] = utc_now_iso()
            return dict(entry)

    def _select(self, kind: str, status: str | None) -> dict[str, dict[str, Any]]:
        items = self.load()[kind]
        return {k: v for k, v in items.items() if status is None or v.get("status") == status}

    def upsert_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        return self._upsert("jobs", job_id, fields)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.load()["jobs"].get(job_id)

    def jobs(self, status: str | None = None) -> dict[str, dict[str, Any]]:
        return self._select("jobs", status)

    def upsert_clip(self, clip_id: str, **fields: Any) -> dict[str, Any]:
        return self._upsert("clips", clip_id, fields)

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        return self.load()["clips"].get(clip_id)

    def clips(self, status: str | None = None) -> dict[str, dict[str, Any]]:
        return self._select("clips", status)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
class Checkpoint:
    """Checkpoints d'un run (``<dir>/<run_id>.json``).

    >>> ckpt = Checkpoint("cache/checkpoints", "job-123")
    >>> transcript = ckpt.run("transcribe", transcribe, video)   # recalculé seulement si pas déjà fait
    >>> if not ckpt.is_done("render"): ...

    Les résultats stockés doivent être sérialisables en JSON. Pour un gros résultat, stockez-le dans
    un fichier et mettez le *chemin* dans le checkpoint.
    """

    def __init__(self, directory: str | os.PathLike, run_id: str):
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
        self.run_id = run_id
        self.path = Path(directory) / f"{safe_id}.json"
        self._store = JsonStore(self.path, lambda: {"run_id": run_id, "steps": {}})

    def _steps(self) -> dict[str, Any]:
        return self._store.load()["steps"]

    def is_done(self, step: str) -> bool:
        return self._steps().get(step, {}).get("status") == "done"

    def get(self, step: str, default: Any = None) -> Any:
        entry = self._steps().get(step)
        return entry.get("data", default) if entry and entry.get("status") == "done" else default

    def mark_done(self, step: str, data: Any = None) -> None:
        with self._store.transaction() as doc:
            doc["steps"][step] = {"status": "done", "at": utc_now_iso(), "data": data}
        log.debug("checkpoint %s/%s done", self.run_id, step)

    def mark_failed(self, step: str, error: BaseException | str) -> None:
        with self._store.transaction() as doc:
            previous = doc["steps"].get(step, {})
            doc["steps"][step] = {
                "status": "failed",
                "at": utc_now_iso(),
                "error": str(error)[:2000],
                "attempts": int(previous.get("attempts", 0)) + 1,
            }

    def attempts(self, step: str) -> int:
        return int(self._steps().get(step, {}).get("attempts", 0))

    def reset(self, step: str | None = None) -> None:
        """Oublie une étape (ou tout le run si ``step`` est None)."""
        if step is None:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        with self._store.transaction() as doc:
            doc["steps"].pop(step, None)

    def summary(self) -> dict[str, str]:
        return {name: entry.get("status", "?") for name, entry in self._steps().items()}

    def run(self, step: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Exécute ``fn`` si l'étape n'est pas déjà faite ; sinon renvoie le résultat mémorisé."""
        if self.is_done(step):
            log.info("↺ %s : déjà fait (checkpoint), étape sautée", step, extra={"run_id": self.run_id})
            return self.get(step)
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.mark_failed(step, exc)
            raise
        self.mark_done(step, result)
        return result
