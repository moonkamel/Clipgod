"""Moments viraux : empreinte audio réelle (ffmpeg), VODs d'origine, orchestration avec faux YouTube."""

import json
import shutil
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from core.config import load_config
from core.storage import StateStore
from modules import viral as V

NOW = datetime(2026, 11, 23, 18, 0, tzinfo=timezone.utc)
SR = 16000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def speechlike(seconds, seed):
    """Bruit dont l'enveloppe spectrale change toutes les 200 ms (comme de la parole / un jeu)."""
    rng = np.random.default_rng(seed)
    n, seg = int(seconds * SR), int(0.2 * SR)
    out = np.zeros(n, np.float32)
    f = np.fft.rfftfreq(seg, 1 / SR)
    for i in range(0, n, seg):
        env = np.exp(-((f - rng.uniform(200, 3000)) / rng.uniform(150, 800)) ** 2)
        chunk = np.fft.irfft(np.fft.rfft(rng.standard_normal(seg)) * env, seg) * rng.uniform(0.05, 1) * (rng.random() > .15)
        out[i:i + seg] = chunk[: n - i]
    return out / (np.abs(out).max() + 1e-9) * 0.8


def save(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    """VOD de 6 min ; Short = intro étrangère de 3 s + VOD[150:175] moins fort + musique."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg requis")
    d = tmp_path_factory.mktemp("viral")
    vod = speechlike(360, 1)
    t = np.arange(25 * SR) / SR
    music = 0.12 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))
    short = np.concatenate([speechlike(3, 2) * 0.7, vod[150 * SR:175 * SR] * 0.6 + music])
    return {"vod": save(d / "vod.wav", vod), "short": save(d / "short.wav", short),
            "other": save(d / "other.wav", speechlike(25, 3)), "dir": d}


@needs_ffmpeg
def test_locate_finds_the_moment_and_rejects_unrelated(audio):
    vf = V.features_from_source(audio["vod"])
    assert vf.shape[1] == V.N_BANDS and abs(vf.shape[0] / V.FPS - 360) < 1
    off, conf = V.locate(V.features_from_source(audio["short"]), vf)
    assert off == pytest.approx(147.0, abs=0.3) and conf > 0.3          # 150 s - 3 s d'intro
    assert V.locate(V.features_from_source(audio["other"]), vf) is None


@needs_ffmpeg
def test_streaming_features_match_in_memory(audio):
    with wave.open(str(audio["vod"])) as w:
        pcm16 = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    pcm8 = pcm16[::2].astype(np.float32) / 32768.0                        # 16 kHz -> 8 kHz (grossier)
    streamed = V.features_from_source(audio["vod"], block_frames=500)       # nombreux blocs
    assert abs(streamed.shape[0] - V.features_from_pcm(pcm8).shape[0]) <= 2


def test_clip_bounds():
    assert V.clip_bounds(100.0, 20.0, {}) == (97.0, 122.5)
    a, b = V.clip_bounds(2.0, 58.0, {})
    assert a == 0.0 and b - a <= 29.0


# ---------------------------------------------------------------- faux YouTube
class _Req:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self.data


def _video(i, title, ch, chid, published, dur, views, desc=""):
    return {"id": i, "snippet": {"title": title, "channelTitle": ch, "channelId": chid, "description": desc,
                                 "publishedAt": iso(published), "defaultAudioLanguage": "en",
                                 "liveBroadcastContent": "none"},
            "contentDetails": {"duration": dur}, "statistics": {"viewCount": str(views)}}


class FakeYT:
    def __init__(self):
        self.catalog = {
            # Short d'une chaîne de clips qui renvoie à la VOD dans sa description
            "SHORT1": _video("SHORT1", "Kai CRASHES the plane in GTA 6 💀", "ClipsDaily", "UCclips00000000000000000",
                             NOW - timedelta(hours=5), "PT28S", 3_000_000,
                             "Full stream: https://www.youtube.com/watch?v=VODKAI00001 @KaiCenat"),
            # Short du streamer lui-même : VOD trouvée dans ses dernières vidéos
            "SHORT2": _video("SHORT2", "GTA 6 cops chase gone wrong", "Streamy", "UCstreamy000000000000000",
                             NOW - timedelta(hours=8), "PT25S", 900_000),
            "SHORT3": _video("SHORT3", "Minecraft lol", "Other", "UCother00000000000000000", NOW - timedelta(hours=2), "PT20S", 9_000_000),
            "VODKAI00001": _video("VODKAI00001", "KAI PLAYS GTA 6", "KaiCenat", "UCkai0000000000000000000",
                                  NOW - timedelta(days=1), "PT3H", 2_000_000),
            "VODSTREAMY1": _video("VODSTREAMY1", "GTA 6 stream day 4", "Streamy", "UCstreamy000000000000000",
                                  NOW - timedelta(days=2), "PT2H", 100_000),
            "OLDVOD": _video("OLDVOD", "GTA 6 first stream", "Streamy", "UCstreamy000000000000000",
                             NOW - timedelta(days=30), "PT2H", 500_000),
        }
        self.calls = []

    def search(self):
        outer = self

        class S:
            def list(self, **kw):
                outer.calls.append(("search", kw))
                return _Req({"items": [{"id": {"videoId": i}} for i in ("SHORT1", "SHORT2", "SHORT3")]})
        return S()

    def videos(self):
        outer = self

        class Vd:
            def list(self, **kw):
                outer.calls.append(("videos", kw))
                return _Req({"items": [outer.catalog[i] for i in kw["id"].split(",") if i in outer.catalog]})
        return Vd()

    def channels(self):
        outer = self

        class C:
            def list(self, **kw):
                outer.calls.append(("channels", kw))
                key = kw.get("id") or kw.get("forHandle")
                return _Req({"items": [{"contentDetails": {"relatedPlaylists": {"uploads": f"UU-{key}"}}}]})
        return C()

    def playlistItems(self):
        uploads = {"UU-UCstreamy000000000000000": ["SHORT2", "VODSTREAMY1", "OLDVOD"], "UU-UCclips00000000000000000": ["SHORT1"],
                   "UU-@KaiCenat": ["VODKAI00001"]}

        class P:
            def list(self, **kw):
                return _Req({"items": [{"contentDetails": {"videoId": v}} for v in uploads.get(kw["playlistId"], [])]})
        return P()


@pytest.fixture
def cfg(project, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy")
    (project / "data").mkdir(exist_ok=True)
    return load_config(project / "config.yaml")


def _yt(cfg, service):
    from core.quota import YouTubeQuota
    from modules.discover import YouTubeDiscover

    return YouTubeDiscover(service, YouTubeQuota.from_config(cfg), cfg, budget=1000)


def test_viral_shorts_sorted_by_views_and_on_topic(cfg):
    shorts = V.viral_shorts(cfg, _yt(cfg, FakeYT()), now=NOW)
    assert [s.id for s in shorts] == ["SHORT1", "SHORT2"]                  # Minecraft écarté
    assert shorts[0].views == 3_000_000 and shorts[0].url.endswith("/shorts/SHORT1")


def test_candidate_vods(cfg):
    yt = _yt(cfg, FakeYT())
    shorts = {s.id: s for s in V.viral_shorts(cfg, yt, now=NOW)}
    c1 = V.candidate_vods(shorts["SHORT1"], yt)
    assert c1[0][0].endswith("VODKAI00001") and c1[0][2] == "KaiCenat"       # lien explicite d'abord
    c2 = V.candidate_vods(shorts["SHORT2"], yt)
    assert [c[0].split("=")[-1] for c in c2] == ["VODSTREAMY1"]              # VOD de 30 jours écartée


@needs_ffmpeg
def test_run_viral_end_to_end(cfg, audio, tmp_path):
    files = {"SHORT1": audio["short"], "SHORT2": audio["other"], "VODKAI00001": audio["vod"],
             "VODSTREAMY1": audio["vod"]}

    def audio_fn(url, out_dir, name):
        key = url.rstrip("/").split("/")[-1].split("=")[-1]
        dst = Path(out_dir) / f"{name}.wav"
        shutil.copy(files[key], dst)
        return dst

    sections, jobs = [], []

    def section_fn(url, start, end, out_dir, name, **kw):
        sections.append((url, start, end))
        f = Path(out_dir) / f"{name}.mp4"
        f.write_bytes(b"x")
        return f

    def prepare_fn(source, cfg_, drive, **kw):
        jobs.append(kw)
        return {"job_id": f"job{len(jobs)}"}

    found = V.run_viral(cfg, None, now=NOW, service=FakeYT(), audio_fn=audio_fn, section_fn=section_fn,
                        prepare_fn=prepare_fn, on_event=lambda m: None)
    assert [h.short.id for h in found] == ["SHORT1"]                       # SHORT2 : moment introuvable
    hit = found[0]
    assert hit.offset == pytest.approx(147.0, abs=0.3) and hit.streamer == "KaiCenat"
    url, s, _e = sections[0]
    assert url.endswith("VODKAI00001") and s == pytest.approx(hit.clip_start - 8, abs=0.01)
    job = jobs[0]
    assert job["video_id"] == "viral_SHORT1" and job["streamer"] == "KaiCenat"
    fm = job["extra_job"]["fixed_moment"]
    assert fm["start"] == pytest.approx(8.0, abs=0.01) and 20 <= fm["end"] - fm["start"] <= 29
    assert job["extra_job"]["viral"]["views"] == 3_000_000
    state = StateStore(Path(cfg["_meta"]["root"]) / "data" / "state.json").load()["viral"]
    assert state["SHORT1"]["status"] == "prepared" and state["SHORT2"]["status"] == "failed"
    # 2e run : rien n'est refait (prepared / échec récent), l'empreinte de la VOD est en cache.
    again = V.run_viral(cfg, None, now=NOW, service=FakeYT(), audio_fn=audio_fn, section_fn=section_fn,
                        prepare_fn=prepare_fn, on_event=lambda m: None)
    assert again == [] and len(jobs) == 1
    assert list((Path(cfg["_meta"]["root"]) / "cache" / "viral").glob("*.npy"))


def test_scan_fixed_moment_and_bestof_viral_order(project):
    from modules import bestof as B
    from modules.scan import fixed_moment

    transcript = {"language": "en", "segments": [{"start": 7.0, "end": 12.0, "text": "no way he did that", "words": [
        {"start": 7.0 + i, "end": 7.9 + i, "word": w} for i, w in enumerate(["no", "way", "he", "did", "that"])]}]}
    m = fixed_moment({"fixed_moment": {"start": 8.5, "end": 30.0}, "streamer": "kai",
                      "viral": {"views": 10}}, transcript, 40.0, {}, "viral_X")
    assert m["start"] == 7.9 and m["end"] == 30.0 and m["viral"] == {"views": 10}   # début hors mot
    assert len(m["hooks"]) == 3 and m["words"]

    cfg = load_config(project / "config.yaml")
    state = StateStore(project / "data" / "state.json")
    clips = project / "output" / "clips"
    clips.mkdir(parents=True)
    for cid, views, score in (("viral_a_0", 100, 99), ("viral_b_0", 5000, 50), ("viral_c_0", 900, 70),
                              ("normal_0", None, 100)):
        mp4 = clips / f"{cid}.mp4"
        mp4.write_bytes(b"x")
        meta = {"video_id": cid[:-2], "moment_idx": 0, "variant": 0, "score": score, "streamer": cid,
                "hook": "No way", "duration": 20, "rendered_at": iso(NOW - timedelta(hours=1)),
                "viral": {"views": views} if views else None}
        mp4.with_suffix(".json").write_text(json.dumps(meta))
        state.upsert_clip(cid, status="synced", path=str(mp4))
    picks = B.select_picks(cfg, now=NOW, viral=True)
    assert [p.clip_id for p in picks] == ["viral_b_0", "viral_c_0", "viral_a_0"]   # par vues, sans « normal »


@pytest.mark.parametrize("title,expected", [
    ("Bro Vs Me INSANE Mission Cards 😱🔥 | GTA VI", True),
    ("GTA 6 - Helicopter Crash & Damage Physics", True),
    ("GTA 6 Devs Knew Players Would Try This", True),
    ("Lock in lucia😭 - GTA 6", True),
    ("Is GTA 5 Better Than GTA 6?", False),
    ("GTA 5 vs GTA 6 graphics", False),
    ("How Will GTA 6 Online Be Like?", False),
    ("Mythpat Cried Live on His GTA 6 Stream", False),
    ("GTA 6 Jason & Lucia DIY Paper Puzzle", False),
    ("GTA 6's Ending Will Leave You in Tears", False),
    ("About GTA 6…", False),                                             # indéterminé -> écarté
])
def test_gameplay_verdict(title, expected):
    ok, why = V.gameplay_verdict(V.Short("x", title, "ch", "", 1, 30, NOW))
    assert ok is expected, why


def test_top_shorts_gameplay_file(cfg, tmp_path):
    yt = _yt(cfg, FakeYT())
    kept, dropped = V.top_shorts(cfg, yt, now=NOW, hours=48, count=10)
    assert [s.id for s, _ in kept] == ["SHORT1", "SHORT2"]               # crash / cops chase : gameplay
    out = V.write_top_shorts(tmp_path / "top.txt", kept, dropped, hours=48, gameplay_only=True)
    text = out.read_text(encoding="utf-8-sig")
    assert text.startswith("Top 2 Shorts GTA 6 gameplay les plus vus (48 h)")
    assert "https://www.youtube.com/shorts/SHORT1" in text and "3 000 000 vues" in text
