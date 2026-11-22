import itertools

import pytest

from core import drive as D

FOLDER = D.FOLDER_MIME


# ---------------------------------------------------------------- fake API Drive v3
class _Req:
    def __init__(self, fn):
        self._fn = fn

    def execute(self, num_retries=0):
        return self._fn()

    def next_chunk(self, num_retries=0):
        return None, self._fn()


class FakeFiles:
    """Émule files().list/create/update/get_media avec un parsing minimal des requêtes q."""

    def __init__(self):
        self.items = {}   # id -> meta (+ "parents", "data", "trashed")
        self._ids = itertools.count(1)
        self.queries = []

    def _match(self, q):
        self.queries.append(q)
        clauses = q.split(" and ")
        out = []
        for meta in self.items.values():
            ok = not meta.get("trashed")
            for c in clauses:
                if c.endswith("in parents"):
                    ok &= c.split("'")[1] in meta["parents"]
                elif c.startswith("name = "):
                    ok &= meta["name"] == c[len("name = '"):-1].replace("\\'", "'").replace("\\\\", "\\")
                elif c.startswith("mimeType = "):
                    ok &= meta["mimeType"] == FOLDER
                elif c.startswith("mimeType != "):
                    ok &= meta["mimeType"] != FOLDER
            if ok:
                out.append(meta)
        return sorted(out, key=lambda m: m["modifiedTime"], reverse=True)

    def list(self, q, **kw):
        return _Req(lambda: {"files": [dict(m) for m in self._match(q)]})

    def _media_bytes(self, media):
        if media is None:
            return b""
        if hasattr(media, "_filename"):
            with open(media._filename, "rb") as fh:
                return fh.read()
        return media.getbytes(0, media.size())

    def create(self, body, media_body=None, fields=None):
        def run():
            fid = f"id{next(self._ids)}"
            data = self._media_bytes(media_body)
            self.items[fid] = {"id": fid, "name": body["name"], "parents": body.get("parents", ["root"]),
                               "mimeType": body.get("mimeType", "application/octet-stream"),
                               "modifiedTime": f"2026-11-22T00:00:{len(self.items):02d}Z",
                               "size": str(len(data)), "data": data}
            return dict(self.items[fid])
        return _Req(run)

    def update(self, fileId, media_body=None, body=None, fields=None):
        def run():
            meta = self.items[fileId]
            if body:
                meta.update(body)
            if media_body is not None:
                meta["data"] = self._media_bytes(media_body)
                meta["size"] = str(len(meta["data"]))
            return dict(meta)
        return _Req(run)


class FakeService:
    def __init__(self):
        self.f = FakeFiles()

    def files(self):
        return self.f


@pytest.fixture
def api_drive(monkeypatch):
    svc = FakeService()
    drv = D.ApiDriveBackend(svc, chunk_mb=1)
    monkeypatch.setattr(drv, "_download_media",
                        lambda fid, fh, progress=None: fh.write(svc.f.items[fid]["data"]))
    return drv


@pytest.fixture
def local_drive(tmp_path):
    return D.LocalDriveBackend(tmp_path / "MyDrive" / "ClipGod")


@pytest.fixture(params=["local", "api"])
def drive(request, local_drive, api_drive):
    return local_drive if request.param == "local" else api_drive


# ---------------------------------------------------------------- comportement commun
def test_norm(local_drive):
    assert local_drive.norm("ClipGod/sources/a.mp4") == "sources/a.mp4"
    assert local_drive.norm("/jobs//x.json") == "jobs/x.json"
    with pytest.raises(D.DriveError):
        local_drive.norm("jobs/../../etc")


def test_structure_upload_download_list(drive, tmp_path):
    drive.ensure_structure()
    assert {f.name for f in drive.list_dir("")} >= {"jobs", "sources", "moments", "clips", "jobs_done"}
    src = tmp_path / "v.mp4"
    src.write_bytes(b"\x00" * 1000)
    seen = []
    f = drive.upload_file(src, "ClipGod/sources/v.mp4", progress=lambda d, t: seen.append(d))
    assert f.path == "sources/v.mp4" and f.size == 1000
    out = drive.download_file("sources/v.mp4", tmp_path / "dl" / "v.mp4")
    assert out.read_bytes() == src.read_bytes()
    assert [x.name for x in drive.list_dir("sources", "*.mp4")] == ["v.mp4"]
    # ré-upload = écrasement, pas de doublon
    src.write_bytes(b"\x01" * 10)
    drive.upload_file(src, "sources/v.mp4")
    assert len(drive.list_dir("sources")) == 1 and drive.stat("sources/v.mp4").size == 10


def test_json_and_delete(drive):
    drive.write_json("moments/x.json", {"a": [1, 2]})
    assert drive.read_json("moments/x.json") == {"a": [1, 2]}
    assert drive.read_json("moments/none.json", default=None) is None
    with pytest.raises(D.DriveError):
        drive.read_bytes("moments/none.json")
    drive.delete("moments/x.json")
    assert not drive.exists("moments/x.json")
    drive.delete("moments/x.json")  # missing_ok
    with pytest.raises(D.DriveError):
        drive.delete("moments/x.json", missing_ok=False)


def test_job_queue(drive):
    drive.ensure_structure()
    drive.write_json("jobs/20261122-100000-aaaaaa.json", {"job_id": "20261122-100000-aaaaaa",
                                                          "created_at": "2026-11-22T10:00:00Z"})
    drive.write_json("jobs/20261122-090000-bbbbbb.json", {"job_id": "20261122-090000-bbbbbb",
                                                          "created_at": "2026-11-22T09:00:00Z"})
    drive.write_bytes("jobs/broken.json", b"{nope")
    pending = D.pending_jobs(drive)
    assert [j["job_id"] for j in pending] == ["20261122-090000-bbbbbb", "20261122-100000-aaaaaa"]
    D.mark_job_done(drive, "20261122-090000-bbbbbb", {"clips": ["clips/x_0.mp4"]})
    assert [j["job_id"] for j in D.pending_jobs(drive)] == ["20261122-100000-aaaaaa"]
    done = drive.read_json("jobs_done/20261122-090000-bbbbbb.json")
    assert done["status"] == "done" and done["clips"] == ["clips/x_0.mp4"]


# ---------------------------------------------------------------- spécifique API
def test_api_escapes_quotes_and_caches_folders(api_drive):
    api_drive.write_json("clips/l'arrestation.json", {})
    assert api_drive.exists("clips/l'arrestation.json")
    n = len(api_drive._service.f.queries)
    api_drive.exists("clips/l'arrestation.json")
    assert len(api_drive._service.f.queries) == n + 1      # dossiers en cache : 1 seule requête
    assert any("\\'" in q for q in api_drive._service.f.queries)


def test_api_missing_root_does_not_create(api_drive):
    assert api_drive.list_dir("jobs") == []
    assert api_drive._service.f.items == {}


# ---------------------------------------------------------------- fabrique
def test_get_drive_local_root(project):
    from core.config import load_config

    (project / "gdrive").mkdir()
    cfg = load_config(project / "config.yaml", overrides={"drive.local_root": "gdrive/ClipGod"})
    drv = D.get_drive(cfg)
    assert isinstance(drv, D.LocalDriveBackend) and drv.root == project / "gdrive" / "ClipGod"


def test_get_drive_colab_requires_mount(project, monkeypatch):
    from core.config import load_config

    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    cfg = load_config(project / "config.yaml", overrides={"drive.colab_mount": str(project / "nomount")})
    with pytest.raises(D.DriveError, match="pas monté"):
        D.get_drive(cfg)


def test_api_backend_non_interactive_without_token(project):
    from core.config import load_config

    cfg = load_config(project / "config.yaml")
    with pytest.raises(D.DriveError, match="Aucun token"):
        D.get_drive(cfg, backend="api", interactive=False)
