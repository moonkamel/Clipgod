"""Worker de bout en bout sur le backend API Drive (cas Kaggle), avec le faux service Drive des tests."""

import json
from pathlib import Path

import pytest

from core import worker as W
from core.config import load_config
from core.drive import ApiDriveBackend, pending_jobs
from tests.test_drive import FakeService


@pytest.fixture
def api_env(project, tmp_path, monkeypatch):
    cfg = load_config(project / "config.yaml")
    svc = FakeService()
    drive = ApiDriveBackend(svc, chunk_mb=1)
    monkeypatch.setattr(drive, "_download_media", lambda fid, fh, progress=None: fh.write(svc.f.items[fid]["data"]))
    drive.ensure_structure()
    src = tmp_path / "vod.mp4"
    src.write_bytes(b"v" * 2048)
    drive.upload_file(src, "sources/twitch_1.mp4")
    drive.write_json("jobs/20261122-100000-aaaaaa.json", {
        "job_id": "20261122-100000-aaaaaa", "video_id": "twitch_1", "streamer": "kaicenat",
        "source_drive_path": "ClipGod/sources/twitch_1.mp4", "created_at": "2026-11-22T10:00:00Z"})
    return cfg, drive, tmp_path


def _scan(calls):
    def scan_fn(video, out, cfg):
        calls.append(cfg["scan"])
        Path(out, "twitch_1.json").write_text(json.dumps({"video_id": "twitch_1", "moments": [{"idx": 0}]}))
        Path(out, "twitch_1.transcript.json").write_text(json.dumps({"segments": []}))
        return [{"idx": 0, "score": 80}]
    return scan_fn


def _render(moments, video, out, cfg):
    p = Path(out, "twitch_1_0.mp4")
    p.write_bytes(b"clip")
    p.with_suffix(".json").write_text(json.dumps({"hook": "h"}))
    return [str(p)]


def test_api_backend_full_job(api_env):
    cfg, drive, tmp = api_env
    calls = []
    res = W.run_queue(drive, cfg, workdir=tmp / "w", scan_fn=_scan(calls), render_fn=_render)
    assert res[0]["status"] == "done"
    assert calls[0]["work_dir"].startswith(str(tmp / "w"))           # pas de Drive monté : checkpoints locaux
    for remote in ("moments/twitch_1.json", "moments/twitch_1.transcript.json", "clips/twitch_1_0.mp4",
                   "clips/twitch_1_0.json", "jobs_done/20261122-100000-aaaaaa.json"):
        assert drive.exists(remote), remote
    assert drive.read_bytes("clips/twitch_1_0.mp4") == b"clip"
    assert not drive.exists("jobs/20261122-100000-aaaaaa.lock") and pending_jobs(drive) == []


def test_existing_transcript_is_reused_after_interruption(api_env):
    cfg, drive, tmp = api_env
    drive.write_json("moments/twitch_1.transcript.json", {"segments": [{"start": 0, "end": 1, "text": "hi"}]})
    calls = []
    W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=_scan(calls))
    reused = calls[0]["transcript_path"]
    assert reused and json.loads(Path(reused).read_text())["segments"][0]["text"] == "hi"
