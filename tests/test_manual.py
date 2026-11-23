import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from core.config import load_config
from core.storage import StateStore
from modules import manual as MN

PARIS = ZoneInfo("Europe/Paris")
NOW = datetime(2026, 11, 22, 7, 0, tzinfo=timezone.utc)          # 08:00 à Paris


def meta(**kw):
    base = {"duration": 22.0, "hook": "Wait for it… 😳", "hook_at_seconds": 0.0, "width": 1080, "height": 1920,
            "transformations": ["vertical_reframe", "word_subtitles", "hook_overlay", "streamer_tag"],
            "video_id": "yt_1", "moment_idx": 0, "variant": 0, "score": 90, "streamer": "theradbrad",
            "source_url": "https://youtube.com/watch?v=1"}
    base.update(kw)
    return base


@pytest.fixture
def env(project):
    cfg = load_config(project / "config.yaml")
    state = StateStore(project / "data" / "state.json")
    clips_dir = project / "output" / "clips"
    clips_dir.mkdir(parents=True)

    def add(clip_id, status="synced", **kw):
        mp4 = clips_dir / f"{clip_id}.mp4"
        mp4.write_bytes(b"mp4")
        mp4.with_suffix(".json").write_text(json.dumps(meta(clip_id=clip_id, **kw)))
        state.upsert_clip(clip_id, status=status, path=str(mp4))
        return mp4

    return cfg, state, add


def test_caption_has_hook_credit_and_tiktok_hashtags(env):
    cfg, _, _ = env
    clip = MN.ClipInfo("c", MN.Path("c.mp4"), meta())
    text = MN.build_caption(clip, cfg, "tiktok")
    assert text.startswith("Wait for it… 😳") and "Credit: @theradbrad" in text
    assert "#fyp" in text and "#shorts" not in text and "\n\n\n" not in text


def test_plan_one_variant_per_moment_best_first_with_daily_cap(env):
    cfg, _, add = env
    add("m0", moment_idx=0, score=70)
    add("m0_h1", moment_idx=0, variant=1, score=70)
    add("m1", moment_idx=1, score=95)
    add("m2", moment_idx=2, score=80, status="published")          # déjà sur YouTube : reste candidat TikTok
    add("bad", moment_idx=3, duration=45)                           # refusé par la validation
    cfg["publish"]["manual"]["max_per_day"] = 2
    rejected = []
    posts = MN.plan_manual(cfg, limit=10, now=NOW, rejected=rejected)
    assert [p.clip.clip_id for p in posts] == ["m1", "m2", "m0"]
    assert rejected and rejected[0][0] == "bad"
    days = [p.slot.astimezone(PARIS).date() for p in posts]
    assert max(days.count(d) for d in set(days)) <= 2
    assert all(p.slot > NOW for p in posts)


def test_recorded_post_is_never_proposed_again(env):
    cfg, state, add = env
    add("m0")
    add("m0_h1", variant=1)
    post = MN.plan_manual(cfg, limit=1, now=NOW)[0]
    MN.record_manual(cfg, post, url="https://www.tiktok.com/@vicereplay/video/1")
    saved = state.load()["posts"]["m0@tiktok"]
    assert saved["backend"] == "manual" and saved["status"] == "scheduled" and saved["url"].endswith("/1")
    assert MN.plan_manual(cfg, limit=5, now=NOW) == []               # ni m0 ni sa variante
    assert state.get_clip("m0")["status"] == "scheduled"


def test_cli_tiktok_plan_and_interactive(env, monkeypatch):
    import clipgod

    cfg, state, add = env
    add("m0")
    add("m1", moment_idx=1, score=50)
    monkeypatch.setattr(MN, "copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(MN, "reveal_file", lambda p: True)
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    conf = ["--config", str(cfg["_meta"]["config_path"])] if cfg.get("_meta", {}).get("config_path") else []
    runner = CliRunner()
    res = runner.invoke(clipgod.app, [*conf, "tiktok", "--plan"])
    assert res.exit_code == 0, res.output
    assert "Credit: @theradbrad" in res.output and "posts" not in state.load()
    res = runner.invoke(clipgod.app, [*conf, "tiktok"], input="https://www.tiktok.com/@x/video/9\ns\n")
    assert res.exit_code == 0, res.output
    posts = state.load()["posts"]
    assert list(posts) == ["m0@tiktok"] and posts["m0@tiktok"]["url"].endswith("/9")


def test_parse_selection():
    assert MN.parse_selection("1,3 5-7", 6) == [1, 3, 5, 6]
    assert MN.parse_selection("2;2, 9", 4) == [2]
    with pytest.raises(ValueError):
        MN.parse_selection("abc", 3)


def test_plan_full_listing_and_only_selection(env):
    cfg, _, add = env
    add("m0", moment_idx=0, score=70)
    add("m0_h1", moment_idx=0, variant=1, score=70)
    add("m1", moment_idx=1, score=95)
    listing = MN.plan_manual(cfg, schedule=False, now=NOW, one_per_moment=False)
    assert [p.clip.clip_id for p in listing] == ["m1", "m0", "m0_h1"]
    chosen = MN.plan_manual(cfg, now=NOW, only=["m0_h1", "m1"])
    assert [p.clip.clip_id for p in chosen] == ["m0_h1", "m1"] and all(p.slot for p in chosen)


def test_cli_tiktok_choose(env, monkeypatch):
    import clipgod

    _cfg, state, add = env
    add("m0", moment_idx=0, score=70)
    add("m0_h1", moment_idx=0, variant=1, score=70)
    add("m1", moment_idx=1, score=95)
    played = []
    monkeypatch.setattr(MN, "copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(MN, "reveal_file", lambda p: True)
    monkeypatch.setattr(MN, "play_file", lambda p: played.append(p.name) or True)
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    # liste : 1=m1, 2=m0, 3=m0_h1 ; on regarde le 3, on choisit 3, puis on confirme
    res = CliRunner().invoke(clipgod.app, ["tiktok", "--choose"], input="v3\n3\n\n")
    assert res.exit_code == 0, res.output
    assert played == ["m0_h1.mp4"]
    assert list(state.load()["posts"]) == ["m0_h1@tiktok"]


def test_hook_quality_and_safe_title():
    assert MN.hook_quality("theRadBrad instantly regretted this") > MN.hook_quality('"Oh, no!" 💀')
    assert MN.hook_quality('"Oh, no!" 💀') > MN.hook_quality("The loudest moment of the stream 🔊")
    assert MN.hook_quality('ShehzadGaming4K: "भाई अरे भाई"') == 0
    assert MN.safe_title('"LET\'S GO." 💀') == "LET'S GO"
    assert MN.safe_title("Wait for it… 😳") == "Wait for it"
    assert MN.safe_title("Who? <did> this/now") == "Who did thisnow"


def test_export_one_video_per_moment_with_captions(env):
    cfg, state, add = env
    root = Path(cfg["_meta"]["root"])
    add("yt_1_0", moment_idx=0, score=80, hook='"LET\'S GO." 💀')
    add("yt_1_0_h1", moment_idx=0, variant=1, score=80, hook="Nobody expected this 😳")
    add("yt_1_0_h2", moment_idx=0, variant=2, score=80, hook="The loudest moment of the stream 🔊")
    add("yt_1_1", moment_idx=1, score=60, hook="theRadBrad completely lost it 😱")
    add("hi_0", video_id="hi", score=99, hook="Wait for it… 😳")                  # source en hindi
    mdir = root / "output" / "moments"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "hi.json").write_text(json.dumps({"video_id": "hi", "language": "hi"}))
    (mdir / "yt_1.json").write_text(json.dumps({"video_id": "yt_1", "language": "en"}))
    rep = MN.export_ready(cfg, now=NOW)
    names = [t.name for t, _ in rep.files]
    assert names == ["01 - Nobody expected this.mp4", "02 - theRadBrad completely lost it.mp4"]
    assert rep.skipped_language == ["hi_0"]
    text = (rep.folder / "legendes.txt").read_text(encoding="utf-8-sig")
    assert "===== 01 - Nobody expected this.mp4 =====" in text and "Credit: @theradbrad" in text
    assert "Programmer :" in text and "😳" in text
    assert state.load()["posts"]["yt_1_0_h1@tiktok"]["status"] == "exported"
    rep2 = MN.export_ready(cfg, now=NOW)                                          # rien de nouveau
    assert rep2.files == [] and not list(rep.folder.glob("*.mp4"))
