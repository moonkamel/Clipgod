"""Helpers Google Drive — seul canal de communication laptop <-> notebook cloud.

Deux backends, même interface (:class:`DriveBackend`) :

- :class:`LocalDriveBackend` : le dossier ``ClipGod/`` est accessible comme un dossier local.
    * Colab : Drive monté sur ``/content/drive/MyDrive/ClipGod`` (``google.colab.drive.mount``).
    * Laptop avec « Google Drive pour ordinateur » : ``drive.local_root`` pointe vers le dossier synchronisé
      (zéro OAuth, zéro quota API).
- :class:`ApiDriveBackend` : API Drive v3 (google-api-python-client), upload/download résumables par chunks.
    * Laptop : OAuth « application de bureau » (navigateur au premier lancement, token mis en cache).
    * Kaggle : token OAuth fourni en JSON via ``CLIPGOD_GDRIVE_TOKEN_JSON`` (secret Kaggle), non interactif.

Les chemins distants sont **relatifs à la racine ``ClipGod/``** et en style POSIX : ``"jobs/<id>.json"``.
Le préfixe ``"ClipGod/"`` est accepté et retiré (les jobs stockent ``"ClipGod/sources/x.mp4"``).

Scope OAuth : ``drive`` complet et non ``drive.file``. Avec ``drive.file`` le laptop ne verrait pas les clips
écrits par le notebook via le montage Colab (fichiers non créés par l'application OAuth).

Les imports Google sont paresseux : le backend local fonctionne sans aucune dépendance Google.
"""

from __future__ import annotations

import fnmatch
import io
import json
import mimetypes
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from core.config import ConfigError, get, get_env, resolve_path
from core.logger import get_logger
from core.storage import atomic_write_bytes, atomic_write_text, utc_now_iso

log = get_logger(__name__)

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_JSON_ENV = "CLIPGOD_GDRIVE_TOKEN_JSON"
DEFAULT_FOLDERS = {
    "jobs": "jobs",
    "sources": "sources",
    "moments": "moments",
    "clips": "clips",
    "jobs_done": "jobs_done",
}
_API_FIELDS = "id, name, size, modifiedTime, mimeType"
_MISSING: Any = object()

ProgressCallback = Callable[[int, int], None]  # (octets_transférés, total)


class DriveError(Exception):
    """Erreur Drive (fichier absent, Drive non monté, échec API…) avec message lisible."""


@dataclass(frozen=True)
class DriveFile:
    path: str                 # relatif à ClipGod/, ex. "clips/abc_0.mp4"
    name: str
    is_folder: bool = False
    size: int | None = None
    modified: str | None = None
    id: str | None = None     # id Drive (backend API uniquement)

    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------
class DriveBackend(ABC):
    kind = "abstract"

    def __init__(self, root_name: str = "ClipGod", folders: Mapping[str, str] | None = None):
        self.root_name = root_name
        self.folders = {**DEFAULT_FOLDERS, **dict(folders or {})}

    # -- chemins -----------------------------------------------------------
    def norm(self, remote: str | os.PathLike) -> str:
        """Normalise un chemin distant : POSIX, relatif à ClipGod/, sans ``..``."""
        parts = [p for p in str(remote).replace("\\", "/").split("/") if p not in ("", ".")]
        if ".." in parts:
            raise DriveError(f"Chemin distant invalide (..) : {remote}")
        if parts and parts[0] == self.root_name:
            parts = parts[1:]
        return "/".join(parts)

    def remote(self, folder_key: str, *names: str) -> str:
        """``drive.remote("clips", "abc_0.mp4")`` -> ``"clips/abc_0.mp4"``."""
        if folder_key not in self.folders:
            raise DriveError(f"Dossier logique inconnu : {folder_key} (connus : {', '.join(self.folders)})")
        return "/".join([self.folders[folder_key], *names])

    def display(self, remote: str) -> str:
        """Chemin lisible pour l'utilisateur : ``ClipGod/clips/abc_0.mp4``."""
        rel = self.norm(remote)
        return f"{self.root_name}/{rel}" if rel else self.root_name

    # -- primitives (à implémenter) ---------------------------------------
    @abstractmethod
    def makedirs(self, remote: str) -> None: ...

    @abstractmethod
    def stat(self, remote: str) -> DriveFile | None: ...

    @abstractmethod
    def list_dir(self, remote: str = "", pattern: str | None = None) -> list[DriveFile]: ...

    @abstractmethod
    def upload_file(self, local: str | os.PathLike, remote: str, *,
                    progress: ProgressCallback | None = None) -> DriveFile: ...

    @abstractmethod
    def download_file(self, remote: str, local: str | os.PathLike, *,
                      progress: ProgressCallback | None = None) -> Path: ...

    @abstractmethod
    def read_bytes(self, remote: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, remote: str, data: bytes, mimetype: str = "application/octet-stream") -> DriveFile: ...

    @abstractmethod
    def delete(self, remote: str, *, missing_ok: bool = True) -> None: ...

    # -- dérivées ------------------------------------------------------------
    def exists(self, remote: str) -> bool:
        return self.stat(remote) is not None

    def read_json(self, remote: str, default: Any = _MISSING) -> Any:
        try:
            raw = self.read_bytes(remote)
        except DriveError:
            if default is _MISSING:
                raise
            return default
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if default is _MISSING:
                raise DriveError(f"JSON invalide sur Drive : {self.display(remote)} ({exc})") from exc
            log.warning("JSON invalide ignoré sur Drive : %s", self.display(remote))
            return default

    def write_json(self, remote: str, data: Any) -> DriveFile:
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
        return self.write_bytes(remote, payload, mimetype="application/json")

    def ensure_structure(self) -> list[str]:
        """Crée ``ClipGod/{jobs,sources,moments,clips,jobs_done}`` si besoin."""
        created = []
        for key in DEFAULT_FOLDERS:
            self.makedirs(self.folders[key])
            created.append(self.display(self.folders[key]))
        return created

    def __repr__(self) -> str:
        return f"<{type(self).__name__} root={self.root_name!r}>"


# ---------------------------------------------------------------------------
# Backend local (montage Colab / Drive for Desktop / tests)
# ---------------------------------------------------------------------------
def _atomic_copy(src: Path, dst: Path, progress: ProgressCallback | None = None,
                 chunk_size: int = 8 * 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    total = src.stat().st_size
    done = 0
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while chunk := fin.read(chunk_size):
                fout.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
            fout.flush()
            os.fsync(fout.fileno())
        shutil.copystat(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class LocalDriveBackend(DriveBackend):
    kind = "local"

    def __init__(self, root: str | os.PathLike, root_name: str = "ClipGod",
                 folders: Mapping[str, str] | None = None):
        super().__init__(root_name, folders)
        self.root = Path(root)

    def _p(self, remote: str) -> Path:
        rel = self.norm(remote)
        return self.root / rel if rel else self.root

    def _to_file(self, p: Path) -> DriveFile:
        st = p.stat()
        rel = p.relative_to(self.root).as_posix()
        from datetime import datetime, timezone

        return DriveFile(
            path=rel,
            name=p.name,
            is_folder=p.is_dir(),
            size=None if p.is_dir() else st.st_size,
            modified=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        )

    def makedirs(self, remote: str) -> None:
        self._p(remote).mkdir(parents=True, exist_ok=True)

    def stat(self, remote: str) -> DriveFile | None:
        p = self._p(remote)
        return self._to_file(p) if p.exists() else None

    def list_dir(self, remote: str = "", pattern: str | None = None) -> list[DriveFile]:
        d = self._p(remote)
        if not d.is_dir():
            return []
        items = [
            self._to_file(p)
            for p in d.iterdir()
            if not p.name.startswith(".") and (pattern is None or fnmatch.fnmatch(p.name, pattern))
        ]
        return sorted(items, key=lambda f: f.name)

    def upload_file(self, local, remote, *, progress=None) -> DriveFile:
        src = Path(local)
        if not src.is_file():
            raise DriveError(f"Fichier local introuvable : {src}")
        dst = self._p(remote)
        if dst.exists() and dst.resolve() == src.resolve():
            return self._to_file(dst)
        _atomic_copy(src, dst, progress)
        log.info("↑ %s -> %s", src.name, self.display(remote), extra={"bytes": src.stat().st_size})
        return self._to_file(dst)

    def download_file(self, remote, local, *, progress=None) -> Path:
        src = self._p(remote)
        if not src.is_file():
            raise DriveError(f"Absent sur Drive : {self.display(remote)}")
        dst = Path(local)
        if not (dst.exists() and dst.resolve() == src.resolve()):
            _atomic_copy(src, dst, progress)
        log.info("↓ %s -> %s", self.display(remote), dst, extra={"bytes": src.stat().st_size})
        return dst

    def read_bytes(self, remote: str) -> bytes:
        p = self._p(remote)
        if not p.is_file():
            raise DriveError(f"Absent sur Drive : {self.display(remote)}")
        return p.read_bytes()

    def write_bytes(self, remote: str, data: bytes, mimetype: str = "application/octet-stream") -> DriveFile:
        p = self._p(remote)
        atomic_write_bytes(p, data)
        return self._to_file(p)

    def delete(self, remote: str, *, missing_ok: bool = True) -> None:
        p = self._p(remote)
        if not p.exists():
            if missing_ok:
                return
            raise DriveError(f"Absent sur Drive : {self.display(remote)}")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    def __repr__(self) -> str:
        return f"<LocalDriveBackend root={str(self.root)!r}>"


# ---------------------------------------------------------------------------
# Backend API Drive v3
# ---------------------------------------------------------------------------
def _translate_http_error(exc: BaseException, action: str) -> DriveError | None:
    try:
        from googleapiclient.errors import HttpError
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(exc, HttpError):
        return None
    status = getattr(getattr(exc, "resp", None), "status", "?")
    try:
        reason = exc._get_reason()  # seule façon fiable d'obtenir le message
    except Exception:  # noqa: BLE001 — API privée : repli sur str(exc)
        reason = str(exc)
    hint = ""
    if status in (401, 403):
        hint = " — vérifiez le token OAuth (supprimez-le pour relancer l'autorisation) et le scope 'drive'."
    elif status == 404:
        hint = " — fichier/dossier introuvable ou non partagé avec ce compte."
    return DriveError(f"API Drive, {action} : HTTP {status} {reason}{hint}")


@contextmanager
def _api_call(action: str) -> Iterator[None]:
    try:
        yield
    except DriveError:
        raise
    except Exception as exc:
        translated = _translate_http_error(exc, action)
        if translated is not None:
            raise translated from exc
        raise


class ApiDriveBackend(DriveBackend):
    kind = "api"

    def __init__(self, service: Any, root_name: str = "ClipGod", folders: Mapping[str, str] | None = None, *,
                 parent_id: str = "root", chunk_mb: int = 32, max_retries: int = 5):
        super().__init__(root_name, folders)
        self._service = service
        self._parent_id = parent_id
        # Les uploads résumables exigent un multiple de 256 Kio.
        self.chunk_size = max(1, int(chunk_mb)) * 1024 * 1024
        self.max_retries = int(max_retries)
        self._folder_ids: dict[str, str] = {}  # chemin dossier relatif ("" = racine ClipGod) -> id

    # -- bas niveau --------------------------------------------------------
    def _files(self) -> Any:
        return self._service.files()

    def _exec(self, request: Any) -> dict[str, Any]:
        return request.execute(num_retries=self.max_retries)

    @staticmethod
    def _q(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _query(self, parent_id: str, name: str | None = None, folder: bool | None = None) -> list[dict[str, Any]]:
        clauses = [f"'{self._q(parent_id)}' in parents", "trashed = false"]
        if name is not None:
            clauses.append(f"name = '{self._q(name)}'")
        if folder is True:
            clauses.append(f"mimeType = '{FOLDER_MIME}'")
        elif folder is False:
            clauses.append(f"mimeType != '{FOLDER_MIME}'")
        results: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            resp = self._exec(self._files().list(
                q=" and ".join(clauses), spaces="drive", pageSize=1000, pageToken=token,
                orderBy="modifiedTime desc", fields=f"nextPageToken, files({_API_FIELDS})",
            ))
            results.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return results

    def _create_folder(self, name: str, parent_id: str) -> str:
        resp = self._exec(self._files().create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}, fields="id"
        ))
        log.debug("dossier Drive créé : %s (%s)", name, resp["id"])
        return resp["id"]

    def _folder_id(self, remote_dir: str, *, create: bool = False) -> str | None:
        rel = self.norm(remote_dir)
        if rel in self._folder_ids:
            return self._folder_ids[rel]
        if "" not in self._folder_ids:
            found = self._query(self._parent_id, self.root_name, folder=True)
            if found:
                if len(found) > 1:
                    log.warning("Plusieurs dossiers %r sur Drive : utilisation du plus récent.", self.root_name)
                self._folder_ids[""] = found[0]["id"]
            elif create:
                self._folder_ids[""] = self._create_folder(self.root_name, self._parent_id)
            else:
                return None
        parent = self._folder_ids[""]
        current = ""
        for part in [p for p in rel.split("/") if p]:
            current = f"{current}/{part}" if current else part
            if current in self._folder_ids:
                parent = self._folder_ids[current]
                continue
            found = self._query(parent, part, folder=True)
            if found:
                fid = found[0]["id"]
            elif create:
                fid = self._create_folder(part, parent)
            else:
                return None
            self._folder_ids[current] = parent = fid
        return parent

    def _split(self, remote: str) -> tuple[str, str]:
        rel = PurePosixPath(self.norm(remote))
        if not rel.name:
            raise DriveError("Chemin de fichier distant vide.")
        parent = rel.parent.as_posix()
        return ("" if parent == "." else parent), rel.name

    def _find(self, remote: str) -> dict[str, Any] | None:
        directory, name = self._split(remote)
        parent = self._folder_id(directory)
        if parent is None:
            return None
        found = self._query(parent, name)
        return found[0] if found else None

    def _to_file(self, meta: Mapping[str, Any], rel: str) -> DriveFile:
        size = meta.get("size")
        return DriveFile(
            path=rel,
            name=meta.get("name", PurePosixPath(rel).name),
            is_folder=meta.get("mimeType") == FOLDER_MIME,
            size=int(size) if size is not None else None,
            modified=meta.get("modifiedTime"),
            id=meta.get("id"),
        )

    def _upload_media(self, media: Any, remote: str) -> tuple[Any, list[dict[str, Any]]]:
        """Prépare la requête create (nouveau fichier) ou update (écrase le contenu, garde l'id)."""
        directory, name = self._split(remote)
        parent = self._folder_id(directory, create=True)
        existing = self._query(parent, name, folder=False)
        if existing:
            request = self._files().update(fileId=existing[0]["id"], media_body=media, fields=_API_FIELDS)
        else:
            request = self._files().create(body={"name": name, "parents": [parent]},
                                           media_body=media, fields=_API_FIELDS)
        return request, existing

    def _download_media(self, file_id: str, fh: Any, progress: ProgressCallback | None = None) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        downloader = MediaIoBaseDownload(fh, self._files().get_media(fileId=file_id), chunksize=self.chunk_size)
        done = False
        while not done:
            status, done = downloader.next_chunk(num_retries=self.max_retries)
            if status and progress:
                progress(int(status.resumable_progress), int(status.total_size or 0))

    # -- interface -----------------------------------------------------------
    def makedirs(self, remote: str) -> None:
        with _api_call(f"création de {self.display(remote)}"):
            self._folder_id(remote, create=True)

    def stat(self, remote: str) -> DriveFile | None:
        rel = self.norm(remote)
        with _api_call(f"lecture de {self.display(rel)}"):
            if not rel:
                fid = self._folder_id("")
                return None if fid is None else DriveFile(path="", name=self.root_name, is_folder=True, id=fid)
            meta = self._find(rel)
            return None if meta is None else self._to_file(meta, rel)

    def list_dir(self, remote: str = "", pattern: str | None = None) -> list[DriveFile]:
        rel = self.norm(remote)
        with _api_call(f"listage de {self.display(rel)}"):
            parent = self._folder_id(rel)
            if parent is None:
                return []
            seen: set[str] = set()
            items: list[DriveFile] = []
            for meta in self._query(parent):  # triés du plus récent au plus ancien : on garde le plus récent
                name = meta["name"]
                if name in seen or (pattern is not None and not fnmatch.fnmatch(name, pattern)):
                    continue
                seen.add(name)
                items.append(self._to_file(meta, f"{rel}/{name}" if rel else name))
        return sorted(items, key=lambda f: f.name)

    def upload_file(self, local, remote, *, progress=None) -> DriveFile:
        from googleapiclient.http import MediaFileUpload

        src = Path(local)
        if not src.is_file():
            raise DriveError(f"Fichier local introuvable : {src}")
        mimetype = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        with _api_call(f"upload de {src.name}"):
            media = MediaFileUpload(str(src), mimetype=mimetype, resumable=True, chunksize=self.chunk_size)
            request, _ = self._upload_media(media, remote)
            response = None
            while response is None:
                status, response = request.next_chunk(num_retries=self.max_retries)
                if status and progress:
                    progress(int(status.resumable_progress), int(status.total_size or 0))
        log.info("↑ %s -> %s", src.name, self.display(remote), extra={"bytes": src.stat().st_size})
        return self._to_file(response, self.norm(remote))

    def download_file(self, remote, local, *, progress=None) -> Path:
        dst = Path(local)
        with _api_call(f"téléchargement de {self.display(remote)}"):
            meta = self._find(remote)
            if meta is None or meta.get("mimeType") == FOLDER_MIME:
                raise DriveError(f"Absent sur Drive : {self.display(remote)}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.part")
            try:
                with open(tmp, "wb") as fh:
                    self._download_media(meta["id"], fh, progress)
                os.replace(tmp, dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        log.info("↓ %s -> %s", self.display(remote), dst, extra={"bytes": dst.stat().st_size})
        return dst

    def read_bytes(self, remote: str) -> bytes:
        with _api_call(f"lecture de {self.display(remote)}"):
            meta = self._find(remote)
            if meta is None or meta.get("mimeType") == FOLDER_MIME:
                raise DriveError(f"Absent sur Drive : {self.display(remote)}")
            buffer = io.BytesIO()
            self._download_media(meta["id"], buffer)
            return buffer.getvalue()

    def write_bytes(self, remote: str, data: bytes, mimetype: str = "application/octet-stream") -> DriveFile:
        from googleapiclient.http import MediaIoBaseUpload

        with _api_call(f"écriture de {self.display(remote)}"):
            media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
            request, _ = self._upload_media(media, remote)
            response = self._exec(request)
        return self._to_file(response, self.norm(remote))

    def delete(self, remote: str, *, missing_ok: bool = True) -> None:
        """Met à la corbeille Drive (récupérable 30 jours) plutôt que supprimer définitivement."""
        rel = self.norm(remote)
        with _api_call(f"suppression de {self.display(rel)}"):
            meta = self._find(rel)
            if meta is None:
                if missing_ok:
                    return
                raise DriveError(f"Absent sur Drive : {self.display(rel)}")
            self._exec(self._files().update(fileId=meta["id"], body={"trashed": True}, fields="id"))
        for key in [k for k in self._folder_ids if k == rel or k.startswith(rel + "/")]:
            del self._folder_ids[key]


# ---------------------------------------------------------------------------
# Authentification (backend API)
# ---------------------------------------------------------------------------
def _token_path(cfg: Mapping[str, Any]) -> Path:
    env_name = get(cfg, "drive.token_env", "CLIPGOD_GDRIVE_TOKEN")
    return resolve_path(cfg, get_env(env_name) or "secrets/drive_token.json")


def _save_token(path: Path, creds: Any) -> None:
    atomic_write_text(path, creds.to_json())
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover — Windows
        pass
    log.info("Token Drive enregistré : %s", path)


def load_credentials(cfg: Mapping[str, Any], *, interactive: bool = True) -> Any:
    """Retourne des credentials OAuth valides pour l'API Drive.

    Ordre : ``$CLIPGOD_GDRIVE_TOKEN_JSON`` (contenu) > fichier token > flux OAuth navigateur (si ``interactive``).
    """
    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:  # pragma: no cover
        raise DriveError("google-auth manquant : pip install google-api-python-client google-auth-oauthlib") from exc

    token_path = _token_path(cfg)
    token_json = get_env(TOKEN_JSON_ENV)
    creds = None
    from_file = False
    try:
        if token_json:
            creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)
        elif token_path.is_file():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            from_file = True
    except (ValueError, json.JSONDecodeError) as exc:
        log.warning("Token Drive illisible (%s) : nouvelle autorisation nécessaire.", exc)
        creds = None

    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            if from_file:
                _save_token(token_path, creds)
            return creds
        except RefreshError as exc:
            log.warning("Rafraîchissement du token Drive impossible (%s).", exc)

    if not interactive:
        raise DriveError(
            f"Aucun token Drive valide. Fournissez {TOKEN_JSON_ENV} (contenu du token généré sur le laptop "
            f"par `clipgod sync --status`) ou exécutez le flux OAuth sur une machine avec navigateur."
        )

    secrets_env = get(cfg, "drive.credentials_env", "CLIPGOD_GDRIVE_CLIENT_SECRETS")
    try:
        secrets_file = resolve_path(cfg, get_env(
            secrets_env, required=True,
            hint="Créez un ID client OAuth « Application de bureau » dans Google Cloud Console (API Drive activée).",
        ))
    except ConfigError as exc:
        raise DriveError(str(exc)) from exc
    if not secrets_file.is_file():
        raise DriveError(f"Fichier client_secrets introuvable : {secrets_file} (variable {secrets_env}).")

    from google_auth_oauthlib.flow import InstalledAppFlow

    log.info("Autorisation Google Drive : une fenêtre de navigateur va s'ouvrir…")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    _save_token(token_path, creds)
    return creds


def build_service(cfg: Mapping[str, Any], *, interactive: bool = True) -> Any:
    from googleapiclient.discovery import build

    creds = load_credentials(cfg, interactive=interactive)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Fabrique
# ---------------------------------------------------------------------------
def _local_root(cfg: Mapping[str, Any], runtime: str) -> Path:
    configured = get(cfg, "drive.local_root", None)
    if configured:
        root = resolve_path(cfg, configured)
        if not root.parent.is_dir():
            raise DriveError(
                f"drive.local_root introuvable : {root.parent} n'existe pas. "
                "Google Drive pour ordinateur est-il lancé et synchronisé ?"
            )
        return root
    if runtime == "colab":
        mount = Path(get(cfg, "drive.colab_mount", "/content/drive"))
        if not (mount / "MyDrive").is_dir():
            raise DriveError(
                f"Google Drive n'est pas monté sur {mount}. Exécutez d'abord : "
                "from google.colab import drive; drive.mount('/content/drive')"
            )
        return Path(get(cfg, "drive.colab_root", str(mount / "MyDrive" / get(cfg, "drive.root_folder"))))
    if runtime == "kaggle":
        return Path(get(cfg, "drive.kaggle_root", "/kaggle/working/ClipGod"))
    raise DriveError("drive.backend=local nécessite drive.local_root (dossier Google Drive pour ordinateur).")


def get_drive(cfg: Mapping[str, Any], backend: str | None = None, *,
              interactive: bool | None = None) -> DriveBackend:
    """Construit le backend Drive adapté à l'environnement.

    ``auto`` : Colab -> montage local ; laptop avec ``drive.local_root`` -> dossier synchronisé ;
    sinon (laptop, Kaggle) -> API Drive.
    """
    runtime = (cfg.get("_meta") or {}).get("runtime", "local")
    choice = backend or get(cfg, "drive.backend", "auto")
    root_name = get(cfg, "drive.root_folder", "ClipGod")
    folders = get(cfg, "drive.folders", DEFAULT_FOLDERS)

    if choice == "auto":
        choice = "local" if (runtime == "colab" or get(cfg, "drive.local_root", None)) else "api"
    if choice == "local":
        drive: DriveBackend = LocalDriveBackend(_local_root(cfg, runtime), root_name, folders)
    elif choice == "api":
        service = build_service(cfg, interactive=(runtime == "local") if interactive is None else interactive)
        drive = ApiDriveBackend(
            service, root_name, folders,
            chunk_mb=int(get(cfg, "drive.upload_chunk_mb", 32)),
            max_retries=int(get(cfg, "drive.max_retries", 5)),
        )
    else:
        raise DriveError(f"drive.backend inconnu : {choice!r} (auto | api | local).")
    log.debug("backend Drive : %r", drive)
    return drive


# ---------------------------------------------------------------------------
# File de jobs (partagée laptop / notebook)
# ---------------------------------------------------------------------------
def done_job_ids(drive: DriveBackend) -> set[str]:
    return {f.stem for f in drive.list_dir(drive.folders["jobs_done"], "*.json")}


def pending_jobs(drive: DriveBackend) -> list[dict[str, Any]]:
    """Jobs de ``jobs/`` sans marqueur dans ``jobs_done/``, du plus ancien au plus récent.

    Chaque job reçoit ``_drive_path``. Un job JSON illisible est ignoré (avec un warning).
    """
    done = done_job_ids(drive)
    jobs: list[dict[str, Any]] = []
    for f in drive.list_dir(drive.folders["jobs"], "*.json"):
        if f.stem in done:
            continue
        try:
            job = drive.read_json(f.path)
        except DriveError as exc:
            log.warning("Job ignoré (%s) : %s", f.name, exc)
            continue
        if not isinstance(job, dict):
            log.warning("Job ignoré (%s) : contenu non objet JSON.", f.name)
            continue
        job.setdefault("job_id", f.stem)
        job["_drive_path"] = f.path
        jobs.append(job)
    return sorted(jobs, key=lambda j: (str(j.get("created_at", "")), j["job_id"]))


def mark_job_done(drive: DriveBackend, job_id: str, payload: Mapping[str, Any] | None = None) -> DriveFile:
    """Écrit ``jobs_done/<job_id>.json`` (status, finished_at, + payload : moments, clips, erreurs…)."""
    doc = {"job_id": job_id, "status": "done", "finished_at": utc_now_iso(), **dict(payload or {})}
    return drive.write_json(drive.remote("jobs_done", f"{job_id}.json"), doc)
