"""Tests de modules.render : logique pure + rendu ffmpeg réel (petite résolution pour la vitesse)."""

import json
import shutil
import subprocess

import pytest

from modules import render as R

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")


# ---------------------------------------------------------------- logique pure
def _sample(t, faces):
    return R.FaceSample(t, faces)


def test_stable_small_face_is_facecam():
    samples = [_sample(i * 0.5, [(0.85 + 0.003 * (i % 2), 0.8, 0.08, 0.14, 0.95)]) for i in range(20)]
    layout = R.decide_layout(samples, {}, src_aspect=16 / 9)
    assert layout.kind == "facecam"
    x, y, w, h = layout.facecam
    assert 0 <= x and x + w <= 1 and 0 <= y and y + h <= 1
    assert x <= 0.85 <= x + w and y <= 0.8 <= y + h
    # ratio du rectangle en pixels = ratio du panneau 1080x576
    assert w * 16 / 9 / h == pytest.approx(1080 / 576, rel=0.02)


def test_moving_or_big_faces_are_not_facecam():
    moving = [_sample(i, [(0.1 + 0.04 * i, 0.5, 0.08, 0.14, 0.9)]) for i in range(20)]
    assert R.decide_layout(moving, {}, src_aspect=16 / 9).kind == "center"
    big = [_sample(i, [(0.5, 0.4, 0.4, 0.6, 0.9)]) for i in range(20)]
    lay = R.decide_layout(big, {}, src_aspect=16 / 9)
    assert lay.kind == "center" and lay.track[-1][1] == pytest.approx(0.5, abs=0.01)
    rare = [_sample(i, [(0.85, 0.8, 0.08, 0.14, 0.9)] if i < 5 else []) for i in range(20)]
    assert R.decide_layout(rare, {}, src_aspect=16 / 9).kind == "center"


def test_manual_region_and_vertical_source():
    assert R.decide_layout([], {}, src_aspect=16 / 9, manual_region=[0.7, 0.7, 0.3, 0.3]).kind == "facecam"
    assert R.decide_layout([], {}, src_aspect=9 / 16).kind == "center"


def test_track_is_smoothed():
    samples = [_sample(i * 0.5, [(0.9, 0.5, 0.3, 0.4, 0.9)]) for i in range(10)]
    xs = [x for _, x in R._track(samples, {"smoothing_alpha": 0.2})]
    assert xs == sorted(xs) and xs[0] < 0.6 and xs[-1] < 0.9          # converge progressivement, sans saut


def test_piecewise_expr():
    assert R.piecewise_expr([(0, 0.5)], scale=1920, offset=-300, lo=0, hi=1300) == "660.0"
    expr = R.piecewise_expr([(0, 0.2), (1, 0.8)], scale=1000, lo=0, hi=1000)
    assert expr.startswith("if(lt(t,0.000)") and "800.0" in expr


def test_chunk_words():
    words = [{"start": i * 0.3, "end": i * 0.3 + 0.25, "word": w} for i, w in enumerate(
        ["oh", "my", "god", "no", "way.", "this", "is", "insane"])]
    chunks = [[w["word"] for w in c] for c in R.chunk_words(words, max_words=3, max_chars=18)]
    assert chunks[0] == ["oh", "my", "god"] and chunks[1] == ["no", "way."]
    assert all(len(c) <= 3 for c in chunks)


def test_build_ass_contents():
    words = [{"start": 10.0, "end": 10.4, "word": "no"}, {"start": 10.4, "end": 10.9, "word": "way!"},
             {"start": 50.0, "end": 51.0, "word": "hors"}]
    rcfg = {"width": 1080, "height": 1920, "subtitles": {"y_position_ratio": 0.64}}
    ass = R.build_ass(words=words, clip_start=9.5, clip_duration=20, rcfg=rcfg, hook="Wait for it… 😳 {x}",
                      streamer="@kaicenat")
    assert "PlayResY: 1920" in ass and r"\pos(540,1228)" in ass         # 64 % de 1920
    assert "0:00:00.50" in ass and "NO" in ass and "WAY!" in ass and "HORS" not in ass
    assert "@kaicenat" in ass and "Wait for it… (x)" in ass and "😳" not in ass
    assert ass.count("Style: Hook") == 1 and "0:00:02.00" in ass       # hook 2 s


def test_ass_color_and_piper_urls():
    assert R._ass_color("#FFE600", "") == "&H0000E6FF" and R._ass_color("&H00FFFFFF", "") == "&H00FFFFFF"
    onnx, meta = R.piper_voice_urls("en_US-ryan-high", "https://hf.co/x")
    assert onnx == "https://hf.co/x/en/en_US/ryan/high/en_US-ryan-high.onnx" and meta.endswith(".onnx.json")


# ---------------------------------------------------------------- rendu réel
@pytest.fixture(scope="module")
def source(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg requis")
    out = tmp_path_factory.mktemp("rsrc") / "src.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=12",
                    "-f", "lavfi", "-i", "color=c=orange:s=120x90:r=30:d=12",
                    "-f", "lavfi", "-i", "sine=f=440:d=12:sample_rate=48000",
                    "-filter_complex", "[0:v][1:v]overlay=510:260[v]", "-map", "[v]", "-map", "2:a",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


def _cfg(tmp_path, **render):
    rcfg = {"width": 360, "height": 640, "fps": 30, "encoder": "libx264", "preset": "ultrafast",
            "cache_dir": str(tmp_path / "cache"),
            "layout": {"mode": "auto", "yunet_model_url": None, "gameplay_ratio": 0.7, "facecam_ratio": 0.3},
            "subtitles": {"font_url": None, "font_size": 30}, "hook": {"font_size": 26},
            "streamer_overlay": {"font_size": 16}, **render}
    return {"render": rcfg, "job": {"video_id": "vid", "streamer": "kaicenat", "job_id": "j1"}}


def _moment():
    return {"idx": 0, "rank": 1, "start": 2.0, "end": 7.0, "score": 91.0, "hooks": ["Wait for it… 😳", "No way 💀"],
            "words": [{"start": 2.5, "end": 2.9, "word": "no"}, {"start": 2.9, "end": 3.4, "word": "way!"}],
            "signals": {"audio_loudness_peak": 1.0}, "reason": "pic audio"}


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height",
                          "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@needs_ffmpeg
def test_render_variants_center_and_resume(tmp_path, source, monkeypatch):
    cfg = _cfg(tmp_path)
    clips = R.render([_moment()], str(source), str(tmp_path / "out"), cfg)
    assert [p.split("/")[-1] for p in clips] == ["vid_0.mp4", "vid_0_h1.mp4"]
    info = _probe(clips[0])
    v = next(s for s in info["streams"] if s["codec_name"] == "h264")
    assert (v["width"], v["height"]) == (360, 640)
    assert float(info["format"]["duration"]) == pytest.approx(5.0, abs=0.15)
    meta = json.loads((tmp_path / "out" / "vid_0_h1.json").read_text())
    assert meta["hook"] == "No way 💀" and meta["variant"] == 1 and meta["layout"] == "center"
    assert {"vertical_reframe", "word_subtitles", "hook_overlay", "streamer_tag"} <= set(meta["transformations"])

    # Reprise : rien n'est re-rendu…
    monkeypatch.setattr(R, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-rendu")))
    assert R.render([_moment()], str(source), str(tmp_path / "out"), cfg) == clips
    # …sauf si la config de rendu change.
    monkeypatch.undo()
    cfg["render"]["hook"]["font_size"] = 30
    assert len(R.render([_moment()], str(source), str(tmp_path / "out"), cfg)) == 2


@needs_ffmpeg
def test_render_facecam_single_variant_loudness(tmp_path, source):
    cfg = _cfg(tmp_path, ab_variants=False)
    cfg["render"]["layout"]["facecam_regions"] = {"kaicenat": [0.79, 0.72, 0.19, 0.25]}
    clips = R.render([_moment()], str(source), str(tmp_path / "fc"), cfg)
    assert len(clips) == 1
    meta = json.loads((tmp_path / "fc" / "vid_0.json").read_text())
    assert meta["layout"] == "facecam" and "facecam_split" in meta["transformations"]
    ebur = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", clips[0], "-af", "ebur128", "-f", "null", "-"],
                          capture_output=True, text=True, check=False).stderr
    integrated = float(ebur.rsplit("I:", 1)[1].split("LUFS")[0])
    assert integrated == pytest.approx(-14, abs=1.0)


@needs_ffmpeg
def test_render_voiceover_degrades_gracefully(tmp_path, source, monkeypatch):
    cfg = _cfg(tmp_path, voiceover={"enabled": True})
    monkeypatch.setattr(R, "synthesize_voice", lambda *a, **k: None)      # piper indisponible
    clips = R.render([_moment()], str(source), str(tmp_path / "vo"), cfg)
    assert len(clips) == 2
    assert json.loads((tmp_path / "vo" / "vid_0.json").read_text())["voiceover"] is False


@needs_ffmpeg
def test_one_bad_moment_does_not_block_others(tmp_path, source):
    bad = {**_moment(), "idx": 1, "start": 5.0, "end": 5.2}
    clips = R.render([bad, _moment()], str(source), str(tmp_path / "mix"), _cfg(tmp_path, ab_variants=False))
    assert [p.split("/")[-1] for p in clips] == ["vid_0.mp4"]
    with pytest.raises(R.RenderError):
        R.render([bad], str(source), str(tmp_path / "mix2"), _cfg(tmp_path))


def test_render_missing_video_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.render([_moment()], str(tmp_path / "nope.mp4"), str(tmp_path), {})
