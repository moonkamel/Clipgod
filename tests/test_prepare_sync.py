import json

import pytest

from core import worker as W
from core.config import load_config
from core.drive import LocalDriveBackend, pending_jobs
from core.storage import StateStore
from modules import prepare as P
from modules import sync as SY


@pytest.fixture
def env(project, tmp_path):
    cfg = load_config(project / "config.yaml")
    drive = LocalDriveBackend(tmp_path / "gdrive" / "ClipGod")
    return cfg, drive, tmp_path


def _probe(url):
    return {"id": "2345678", "title": "GTA 6 RP chaos", "uploader_id": "kaicenat", "duration": 7200.0,
            "webpage_url": url, "heatmap": [{"start_time": 0, "end_time": 10, "value": 0.4}]}


def _fake_download(url, out_dir, video_id, **kw):
    from pathlib import Path

    p = Path(out_dir) / f"{video_id}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"vod" * 100)
    return p


# ---------------------------------------------------------------- prepare
def test_detect_platform_and_mode():
    assert P.detect_platform("https://www.twitch.tv/videos/1") == "twitch"
    assert P.detect_platform("https://youtu.be/abc") == "youtube"
    assert P.detect_platform("https://kick.com/video/x") == "kick"
    yt = P.Source("url", "youtube", "youtube_a")
    tw = P.Source("url", "twitch", "twitch_a")
    assert P.choose_download_mode(yt, "auto") == "local" and P.choose_download_mode(tw, "auto") == "cloud"
    assert P.choose_download_mode(tw, "local") == "local"


def test_prepare_twitch_url_cloud_mode(env):
    cfg, drive, _ = env
    job = P.prepare("https://www.twitch.tv/videos/2345678", cfg, drive, probe=_probe,
                    downloader=lambda *a, **k: pytest.fail("aucun téléchargement en mode cloud"))
    assert job["video_id"] == "twitch_2345678" and job["download"] == "cloud"
    assert job["source_drive_path"] is None and job["streamer"] == "kaicenat"
    assert job["heatmap"] and job["config"]["scan"] and "publish" not in job["config"]
    assert drive.read_json(f"jobs/{job['job_id']}.json")["video_id"] == "twitch_2345678"
    state = StateStore(cfg["_meta"]["root"] + "/data/state.json")
    assert state.get_job(job["job_id"])["status"] == "queued"


def test_prepare_youtube_downloads_locally_and_uploads(env):
    cfg, drive, tmp = env
    job = P.prepare("https://www.youtube.com/watch?v=2345678", cfg, drive, probe=_probe, downloader=_fake_download,
                    streamer="@other", voiceover=True)
    assert job["source_drive_path"] == "ClipGod/sources/youtube_2345678.mp4"
    assert drive.exists("sources/youtube_2345678.mp4") and job["options"]["voiceover"] is True
    assert job["streamer"] == "@other"
    assert not list((tmp / "cache" / "downloads").glob("*.mp4"))        # VOD locale supprimée après upload


def test_prepare_local_file_dedupe_and_force(env):
    cfg, drive, tmp = env
    vod = tmp / "inbox" / "raw_20261122_2130.mp4"
    vod.parent.mkdir()
    vod.write_bytes(b"x" * 5000)
    first = P.prepare(str(vod), cfg, drive, streamer="kaicenat")
    assert first["platform"] == "local" and first["video_id"].startswith("local_raw_20261122_2130_")
    again = P.prepare(str(vod), cfg, drive)
    assert again["_existing"] == "en attente" and again["job_id"] == first["job_id"]
    forced = P.prepare(str(vod), cfg, drive, force=True)
    assert forced["job_id"] != first["job_id"] and len(pending_jobs(drive)) == 2
    with pytest.raises(P.PrepareError):
        P.prepare(str(tmp / "missing.mp4"), cfg, drive)


def test_prepare_pushes_model_only_when_changed(env):
    cfg, drive, tmp = env
    model = tmp / "data" / "model.pkl"
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"v1")
    state = StateStore(tmp / "data" / "state.json")
    assert P.push_model(cfg, drive, state) is True and drive.read_bytes("model.pkl") == b"v1"
    assert P.push_model(cfg, drive, state) is False
    model.write_bytes(b"v2")
    assert P.push_model(cfg, drive, state) is True and drive.read_bytes("model.pkl") == b"v2"


def test_prepare_rejects_live_and_playlists(env):
    cfg, drive, _ = env
    with pytest.raises(P.PrepareError, match="direct"):
        P.prepare("https://twitch.tv/kaicenat", cfg, drive, probe=lambda u: {"id": "x", "is_live": True})


def test_worker_downloads_cloud_jobs(env, monkeypatch):
    cfg, drive, tmp = env
    P.prepare("https://www.twitch.tv/videos/2345678", cfg, drive, probe=_probe)
    monkeypatch.setattr(P, "download_url", _fake_download)
    seen = []

    def scan_fn(video, out, c):
        seen.append((video, c["job"]["heatmap"]))
        return []

    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=scan_fn)
    assert res[0]["status"] == "done" and seen[0][0].endswith("twitch_2345678.mp4") and seen[0][1]


# ---------------------------------------------------------------- sync
def _done_job(drive, tmp, cfg, *, failed=False):
    vod = tmp / "v.mp4"
    vod.write_bytes(b"x" * 3000)
    job = P.prepare(str(vod), cfg, drive, streamer="kaicenat")
    jid, vid = job["job_id"], job["video_id"]
    if failed:
        drive.write_json(f"jobs_done/{jid}.json", {"status": "failed", "error": "CUDA OOM", "video_id": vid})
        return jid, vid
    drive.write_json(f"moments/{vid}.json", {"video_id": vid, "moments": []})
    clips = []
    for k, suffix in enumerate(["", "_h1"]):
        name = f"{vid}_0{suffix}"
        drive.write_bytes(f"clips/{name}.mp4", b"clip" * (10 + k))
        drive.write_json(f"clips/{name}.json", {"hook": f"hook {k}", "variant": k, "moment_idx": 0,
                                                "duration": 15.0, "score": 90, "streamer": "kaicenat"})
        clips.append(f"clips/{name}.mp4")
    drive.write_json(f"jobs_done/{jid}.json", {"status": "done", "video_id": vid, "clips": clips,
                                               "moments": f"moments/{vid}.json", "n_moments": 1})
    return jid, vid


def test_sync_downloads_updates_state_and_cleans_source(env):
    cfg, drive, tmp = env
    jid, vid = _done_job(drive, tmp, cfg)
    state = StateStore(tmp / "data" / "state.json")
    assert [r.status for r in SY.queue_overview(drive, state)] == ["done"]

    rep = SY.sync(cfg, drive)
    assert rep.synced_jobs == [jid] and not rep.errors
    out = tmp / "output" / "clips"
    assert (out / f"{vid}_0.mp4").read_bytes() == b"clip" * 10 and (out / f"{vid}_0_h1.json").is_file()
    assert (tmp / "output" / "moments" / f"{vid}.json").is_file()
    clip = state.get_clip(f"{vid}_0_h1")
    assert clip["status"] == "synced" and clip["hook"] == "hook 1" and clip["job_id"] == jid
    assert state.get_job(jid)["status"] == "synced"
    assert rep.cleaned_sources and not drive.exists(f"sources/{vid}.mp4")

    # idempotent : 2e passage ne fait rien, et n'écrase pas un statut avancé
    state.upsert_clip(f"{vid}_0", status="published")
    rep2 = SY.sync(cfg, drive)
    assert rep2.synced_jobs == [] and state.get_clip(f"{vid}_0")["status"] == "published"
    rep3 = SY.sync(cfg, drive, redownload=True)
    assert rep3.synced_jobs == [jid] and rep3.skipped == 4 and state.get_clip(f"{vid}_0")["status"] == "published"


def test_sync_reports_failed_and_pending(env):
    cfg, drive, tmp = env
    jid, _ = _done_job(drive, tmp, cfg, failed=True)
    other = P.prepare("https://www.twitch.tv/videos/2345678", cfg, drive, probe=_probe)
    W.claim_job(drive, other["job_id"], "colab-x")
    state = StateStore(tmp / "data" / "state.json")
    statuses = {r.job_id: r.status for r in SY.queue_overview(drive, state)}
    assert statuses == {jid: "failed", other["job_id"]: "running"}
    rep = SY.sync(cfg, drive)
    assert rep.failed_jobs and rep.failed_jobs[0][0] == jid and "CUDA OOM" in rep.failed_jobs[0][1]
    assert state.get_job(jid)["status"] == "failed"


# ---------------------------------------------------------------- CLI
def test_cli_prepare_and_sync_status(env, monkeypatch):
    from typer.testing import CliRunner

    import clipgod

    _cfg, drive, tmp = env
    monkeypatch.setattr("core.drive.get_drive", lambda c, **k: drive)
    vod = tmp / "clip.mp4"
    vod.write_bytes(b"y" * 100)
    runner = CliRunner()
    res = runner.invoke(clipgod.app, ["--config", str(tmp / "config.yaml"), "prepare", "--source", str(vod),
                                      "--streamer", "kaicenat"])
    assert res.exit_code == 0, res.output
    assert "Job" in res.output and "clipgod sync" in res.output and "colab.research.google.com" in res.output
    res = runner.invoke(clipgod.app, ["--config", str(tmp / "config.yaml"), "sync", "--status"])
    assert res.exit_code == 0 and "queued" in res.output
    res = runner.invoke(clipgod.app, ["--config", str(tmp / "config.yaml"), "run"])
    assert res.exit_code == 1 and "discover" in res.output              # aucune clé API -> aucune VOD
    assert json.loads((tmp / "data" / "state.json").read_text())["jobs"]
