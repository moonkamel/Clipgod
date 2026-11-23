"""Best-of quotidien : sélection (unitaire) + vrai montage ffmpeg sur de petits clips."""

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from core.config import load_config
from core.storage import StateStore
from modules import bestof as B

NOW = datetime(2026, 11, 23, 18, 0, tzinfo=timezone.utc)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")


def _meta(clip_id, video_id, idx, variant=0, score=80, streamer="kai", hook="He crashed the plane", hours=2,
          duration=20.0):
    return {"clip_id": clip_id, "video_id": video_id, "moment_idx": idx, "variant": variant, "score": score,
            "streamer": streamer, "hook": hook, "duration": duration,
            "rendered_at": (NOW - timedelta(hours=hours)).isoformat()}


@pytest.fixture
def env(project):
    cfg = load_config(project / "config.yaml")
    cfg["render"].update({"width": 180, "height": 320, "fps": 15, "encoder": "libx264", "preset": "ultrafast",
                          "subtitles": {**cfg["render"]["subtitles"], "font_url": None}})
    state = StateStore(project / "data" / "state.json")
    clips = project / "output" / "clips"
    clips.mkdir(parents=True)
    (project / "output" / "moments").mkdir(parents=True)

    def add(clip_id, *, media=None, **kw):
        mp4 = clips / f"{clip_id}.mp4"
        if media:
            shutil.copy(media, mp4)
        else:
            mp4.write_bytes(b"x")
        meta = _meta(clip_id, **kw)
        mp4.with_suffix(".json").write_text(json.dumps(meta))
        state.upsert_clip(clip_id, status="synced", path=str(mp4), duration=meta["duration"])

    def moments(video_id, language="en", cutscene=None):
        (project / "output" / "moments" / f"{video_id}.json").write_text(json.dumps(
            {"video_id": video_id, "language": language,
             "moments": [{"idx": i, "cutscene": (cutscene or {}).get(i, 0.0)} for i in range(5)]}))

    return cfg, state, add, moments


def test_selection_rules(env):
    cfg, state, add, moments = env
    moments("a")
    moments("b", cutscene={1: 0.9})
    moments("hi", language="hi")
    add("a_0", video_id="a", idx=0, score=90, hook="The loudest moment of the stream 🔊")
    add("a_0_h1", video_id="a", idx=0, variant=1, score=90, hook="kai instantly regretted this")
    add("a_1", video_id="a", idx=1, score=85)
    add("a_2", video_id="a", idx=2, score=84)                        # 3e moment de kai : max 2 par streamer
    add("b_0", video_id="b", idx=0, score=70, streamer="xqc")
    add("b_1", video_id="b", idx=1, score=99, streamer="xqc")        # cinématique
    add("hi_0", video_id="hi", idx=0, score=99, streamer="desi")     # autre langue
    add("old_0", video_id="old", idx=0, score=99, streamer="old", hours=100)   # trop ancien
    picks = B.select_picks(cfg, now=NOW)
    assert [(p.clip_id, p.rank) for p in picks] == [("a_0_h1", 1), ("a_1", 2), ("b_0", 3)]

    state.save({**state.load(), "bestof": {"TOP3_x": {"groups": ["a_0"]}}})   # déjà utilisé hier
    assert [p.clip_id for p in B.select_picks(cfg, now=NOW)] == ["a_1", "a_2", "b_0"]


def test_plan_durations_respects_target():
    picks = [B.Pick(str(i), None, {}, d) for i, d in enumerate([29, 25, 22, 20, 12])]
    B.plan_durations(picks, max_clip=18, target_total=80, intro=1.6, card=0.9)
    assert all(p.use_seconds <= 18 for p in picks)
    assert sum(p.use_seconds for p in picks) + 1.6 + 0.9 * 5 <= 80.5
    assert picks[4].use_seconds == 12                                # un clip court n'est pas rallongé


def test_overlay_and_caption(env):
    cfg, *_ = env
    picks = [B.Pick(f"c{i}", None, {"streamer": s, "hook": h, "score": 90 - i}, 15, rank=i + 1, use_seconds=10)
             for i, (s, h) in enumerate([("kai", "No way 💀"), ("xqc", "Wait for it"), ("caseoh", "LOL")])]
    ass = B.build_overlay(picks, W=1080, H=1920, intro=1.6, card=0.9, lang="en", date_label="23 NOV 2026",
                          font="Montserrat ExtraBold")
    assert "TOP 3" in ass and "GTA 6 MOMENTS" in ass and "23 NOV 2026" in ass
    order = [line.split(",,0,0,0,,")[1].split("}")[-1] for line in ass.splitlines() if ",Big," in line][1:]
    assert order == ["#3", "#2", "#1"]                               # compte à rebours
    caption = B.build_caption(picks, cfg, lang="en", date_label="23 NOV 2026")
    assert caption.splitlines()[0].startswith("Top 3 GTA 6 Moments Of The Day")
    assert caption.index("#3 @caseoh") < caption.index("#1 @kai — No way 💀")
    assert "Credits: @kai @xqc @caseoh" in caption and "#gta6" in caption


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg requis")
    out = []
    for i, color in enumerate(("red", "green", "blue")):
        f = tmp_path_factory.mktemp("bo") / f"m{i}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", f"color=c={color}:s=180x320:r=15:d=4",
                        "-f", "lavfi", "-i", f"sine=f={440 + 100 * i}:d=4:sample_rate=48000",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(f)], check=True)
        out.append(f)
    return out


@needs_ffmpeg
def test_make_bestof_real_ffmpeg(env, media):
    cfg, state, add, moments = env
    moments("v")
    for i, f in enumerate(media):
        add(f"v_{i}", media=f, video_id="v", idx=i, score=90 - i, streamer=f"s{i}", duration=4.0)
    cfg["bestof"].update({"intro_seconds": 1.0, "card_seconds": 0.5})
    result = B.make_bestof(cfg, now=NOW)
    assert result.video.is_file() and result.caption_file.is_file()
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
         str(result.video)], capture_output=True, text=True, check=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (180, 320)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(1.0 + 3 * (0.5 + 4.0), abs=0.4)
    assert result.video.name.startswith("TOP3_2026-11-23")
    assert set(state.load()["bestof"][result.video.stem]["groups"]) == {"v_0", "v_1", "v_2"}
    # Le lendemain, ces moments ne reviennent pas.
    with pytest.raises(B.BestOfError):
        B.make_bestof(cfg, now=NOW + timedelta(hours=1))


def test_cli_bestof_plan_and_not_enough(env):
    from typer.testing import CliRunner

    import clipgod

    _cfg, _state, add, moments = env
    moments("a")
    add("a_0", video_id="a", idx=0, score=90, hours=1)
    runner = CliRunner()
    res = runner.invoke(clipgod.app, ["bestof", "--plan", "--hours", "1000"])
    assert res.exit_code == 0 and "a_0.mp4" in res.output
    res = runner.invoke(clipgod.app, ["bestof", "--no-open", "--hours", "1000"])
    assert res.exit_code == 1 and "minimum 3" in res.output


@needs_ffmpeg
def test_bestof_from_folder(env, tmp_path):
    cfg, _state, _add, _moments = env
    folder = tmp_path / "bestof"
    folder.mkdir()
    specs = [("1 - @KaiCenat crash.mp4", "red", "180x320", True), ("2 - wtf.mp4", "green", "320x180", False),
             ("10 - last.mp4", "blue", "180x320", True)]
    for name, color, size, sound in specs:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=15:d=3"]
        if sound:
            cmd += ["-f", "lavfi", "-i", "sine=f=500:d=3:sample_rate=48000", "-c:a", "aac", "-shortest"]
        subprocess.run([*cmd, "-c:v", "libx264", "-preset", "ultrafast", str(folder / name)], check=True)
    (folder / "notes.txt").write_text("ignoré")

    picks = B.picks_from_folder(folder)
    assert [p.path.name for p in picks] == ["1 - @KaiCenat crash.mp4", "2 - wtf.mp4", "10 - last.mp4"]  # tri naturel
    assert picks[0].streamer == "KaiCenat" and not picks[1].meta["has_audio"]

    cfg["bestof"].update({"intro_seconds": 1.0, "card_seconds": 0.5})
    result = B.make_bestof(cfg, now=NOW, folder=folder)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
         str(result.video)], capture_output=True, text=True, check=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (180, 320)
    assert float(info["format"]["duration"]) == pytest.approx(1.0 + 3 * (0.5 + 3.0), abs=0.5)
    assert "#1 @KaiCenat" in result.caption and "#2" in result.caption
    archived = folder / "publies" / "2026-11-23"
    assert sorted(p.name for p in archived.iterdir()) == sorted(n for n, *_ in specs)
    assert not list(folder.glob("*.mp4")) and (folder / "notes.txt").is_file()     # dossier prêt pour demain
