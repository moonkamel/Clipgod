import json

import pytest

from core import worker as W
from core.config import load_config
from core.drive import LocalDriveBackend, pending_jobs


@pytest.fixture
def env(project, tmp_path):
    cfg = load_config(project / "config.yaml")
    drive = LocalDriveBackend(tmp_path / "MyDrive" / "ClipGod")
    drive.ensure_structure()
    return cfg, drive, tmp_path


def add_job(drive, tmp_path, job_id="20261122-100000-aaaaaa", **extra):
    src = tmp_path / f"{job_id}.mp4"
    src.write_bytes(b"video")
    drive.upload_file(src, f"sources/{job_id}.mp4")
    job = {"job_id": job_id, "video_id": f"vid_{job_id[-6:]}", "source_drive_path": f"ClipGod/sources/{job_id}.mp4",
           "streamer": "kaicenat", "created_at": "2026-11-22T10:00:00Z",
           "config": {"scan": {"top_k_moments": 2}}, **extra}
    drive.write_json(f"jobs/{job_id}.json", job)
    return job


def fake_scan(calls):
    def _scan(video, out, cfg):
        calls.append(cfg)
        vid = cfg["job"]["video_id"]
        moments = [{"rank": 1, "start": 1.0, "end": 20.0, "score": 88.0, "hooks": ["h"], "reason": "r"}]
        (tmp := __import__("pathlib").Path(out)).mkdir(parents=True, exist_ok=True)
        (tmp / f"{vid}.json").write_text(json.dumps({"video_id": vid, "moments": moments}))
        (tmp / f"{vid}.transcript.json").write_text("{}")
        return moments
    return _scan


def fake_render(moments, video, out, cfg):
    from pathlib import Path

    paths = []
    for i, _ in enumerate(moments):
        p = Path(out) / f"{cfg['job']['video_id']}_{i}.mp4"
        p.write_bytes(b"clip")
        paths.append(str(p))
    return paths


def test_run_queue_scan_and_render(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    calls = []
    res = W.run_queue(drive, cfg, workdir=tmp / "work", scan_fn=fake_scan(calls), render_fn=fake_render)
    assert [r["status"] for r in res] == ["done"]
    # config : job > repo, + work_dir sur Drive (checkpoints persistants)
    assert calls[0]["scan"]["top_k_moments"] == 2 and calls[0]["job"]["streamer"] == "kaicenat"
    assert calls[0]["scan"]["work_dir"].startswith(str(drive.root))
    assert "config" not in calls[0]["job"]
    assert drive.exists("moments/vid_aaaaaa.json") and drive.exists("moments/vid_aaaaaa.transcript.json")
    assert drive.exists("clips/vid_aaaaaa_0.mp4")
    done = drive.read_json("jobs_done/20261122-100000-aaaaaa.json")
    assert done["status"] == "done" and done["clips"] == ["clips/vid_aaaaaa_0.mp4"] and done["n_moments"] == 1
    assert not drive.exists("jobs/20261122-100000-aaaaaa.lock")
    assert pending_jobs(drive) == []
    assert not (tmp / "work" / "sources" / "20261122-100000-aaaaaa.mp4").exists()     # source locale nettoyée


def test_scan_only_then_render_later(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]))
    done = drive.read_json("jobs_done/20261122-100000-aaaaaa.json")
    assert done["stages"] == ["scan"] and done["clips"] == []


def test_model_pkl_is_picked_up(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    drive.write_bytes("model.pkl", b"weights")
    calls = []
    W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan(calls))
    assert calls[0]["scan"]["model_path"].endswith("model.pkl")


def test_failure_retries_then_marks_failed(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)

    def boom(*a):
        raise RuntimeError("CUDA out of memory")

    r1 = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=boom, max_attempts=2)
    assert r1[0]["status"] == "retry" and len(pending_jobs(drive)) == 1
    assert "out of memory" in W.read_lock(drive, "20261122-100000-aaaaaa")["last_error"]
    r2 = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=boom, max_attempts=2)
    assert r2[0]["status"] == "failed" and pending_jobs(drive) == []
    done = drive.read_json("jobs_done/20261122-100000-aaaaaa.json")
    assert done["status"] == "failed" and done["attempts"] == 2


def test_invalid_job_fails_immediately_and_queue_continues(env):
    cfg, drive, tmp = env
    drive.write_json("jobs/20261122-090000-bad000.json", {"job_id": "20261122-090000-bad000"})
    add_job(drive, tmp)
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]))
    assert [r["status"] for r in res] == ["failed", "done"]


def test_missing_source_is_final_error(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    drive.delete("sources/20261122-100000-aaaaaa.mp4")
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]))
    assert res[0]["status"] == "failed" and "Source absente" in res[0]["error"]


def test_lock_held_by_other_worker_is_respected(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    assert W.claim_job(drive, "20261122-100000-aaaaaa", "kaggle-other-1") is not None
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]))
    assert res == [] and len(pending_jobs(drive)) == 1
    assert W.queue_status(drive)[0]["locked_by"] == "kaggle-other-1"


def test_not_implemented_stage_stops_without_burning_attempts(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)

    def not_yet(*a):
        raise NotImplementedError("render : étape 5")

    with pytest.raises(NotImplementedError):
        W.run_queue(drive, cfg, workdir=tmp / "w", scan_fn=fake_scan([]), render_fn=not_yet)
    assert W.read_lock(drive, "20261122-100000-aaaaaa")["attempts"] == 0
    assert len(pending_jobs(drive)) == 1


def test_max_jobs_and_only_job(env):
    cfg, drive, tmp = env
    add_job(drive, tmp, "20261122-100000-aaaaaa")
    add_job(drive, tmp, "20261122-110000-bbbbbb", created_at="2026-11-22T11:00:00Z")
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]),
                      only_job="20261122-110000-bbbbbb")
    assert [r["job_id"] for r in res] == ["20261122-110000-bbbbbb"]
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]), max_jobs=1)
    assert len(res) == 1 and pending_jobs(drive) == []


def test_voiceover_option_maps_to_render_config(env):
    cfg, _, _ = env
    out = W.build_job_config(cfg, {"job_id": "j", "options": {"voiceover": True}})
    assert out["render"]["voiceover"]["enabled"] is True


def test_lease_expires_quickly_and_only_job_forces(env):
    cfg, drive, tmp = env
    add_job(drive, tmp)
    jid = "20261122-100000-aaaaaa"
    lock = W.claim_job(drive, jid, "colab-dead-1")                # session qui va « planter »
    exp = W._parse_ts(lock["expires_at"])
    assert 19 <= (exp - W.utc_now()).total_seconds() / 60 <= 21    # bail de 20 min, plus 3 h
    assert W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([])) == []
    res = W.run_queue(drive, cfg, workdir=tmp / "w", stages=("scan",), scan_fn=fake_scan([]), only_job=jid)
    assert res[0]["status"] == "done"                              # ONLY_JOB reprend le bail


def test_lease_renewal(env):
    _cfg, drive, tmp = env
    add_job(drive, tmp)
    jid = "20261122-100000-aaaaaa"
    W.claim_job(drive, jid, "me", lease_minutes=1)
    assert W.renew_lease(drive, jid, "me", lease_minutes=30)
    exp = W._parse_ts(W.read_lock(drive, jid)["expires_at"])
    assert (exp - W.utc_now()).total_seconds() > 25 * 60
    assert not W.renew_lease(drive, jid, "someone-else")
    with W.LeaseKeeper(drive, jid, "me", lease_minutes=0.05) as keeper:   # renouvelle toutes les 5 s
        assert keeper._thread.is_alive()
    assert not keeper._thread.is_alive()
