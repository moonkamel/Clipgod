import json

import pytest

from core import storage as S


def test_atomic_json_roundtrip(tmp_path):
    p = tmp_path / "a" / "b.json"
    S.write_json(p, {"x": "é"})
    assert S.read_json(p) == {"x": "é"}
    assert not list(p.parent.glob("*.tmp"))


def test_read_json_corrupt_with_default(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{oops")
    assert S.read_json(p, default={}) == {}
    assert list(tmp_path.glob("bad.json.corrupt-*"))
    with pytest.raises(FileNotFoundError):
        S.read_json(tmp_path / "missing.json")


def test_state_store(tmp_path):
    st = S.StateStore(tmp_path / "state.json")
    st.upsert_job("j1", status="queued", video_id="v")
    st.upsert_job("j1", status="done")
    st.upsert_clip("c1", status="synced")
    assert st.get_job("j1")["status"] == "done" and st.get_job("j1")["video_id"] == "v"
    assert list(st.jobs("done")) == ["j1"] and st.jobs("queued") == {}
    assert json.loads((tmp_path / "state.json").read_text())["version"] == 1


def test_checkpoint_skips_done_steps(tmp_path):
    calls = []

    def work(x):
        calls.append(x)
        return {"v": x * 2}

    ck = S.Checkpoint(tmp_path, "job/1")
    assert ck.run("step", work, 2) == {"v": 4}
    assert S.Checkpoint(tmp_path, "job/1").run("step", work, 2) == {"v": 4}   # rechargé depuis disque
    assert calls == [2]


def test_checkpoint_records_failures(tmp_path):
    ck = S.Checkpoint(tmp_path, "r")

    def boom():
        raise RuntimeError("cuda OOM")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            ck.run("transcribe", boom)
    assert ck.attempts("transcribe") == 2 and not ck.is_done("transcribe")
    ck.reset()
    assert ck.summary() == {}


def test_ids_and_slug():
    assert S.slugify("GTA 6 — Course poursuite !") == "gta-6-course-poursuite"
    jid = S.new_job_id()
    assert len(jid.split("-")) == 3
