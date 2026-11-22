import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from core.config import load_config
from core.quota import YouTubeQuota
from core.storage import StateStore
from modules import monetize as MZ
from modules import publish as PB

PARIS = ZoneInfo("Europe/Paris")
NOW = datetime(2026, 11, 22, 7, 0, tzinfo=timezone.utc)          # 08:00 à Paris


def meta(**kw):
    base = {"duration": 22.0, "hook": "Wait for it… 😳", "hook_at_seconds": 0.0, "width": 1080, "height": 1920,
            "transformations": ["vertical_reframe", "word_subtitles", "hook_overlay", "streamer_tag"],
            "video_id": "twitch_1", "moment_idx": 0, "variant": 0, "score": 90, "streamer": "kaicenat",
            "source_url": "https://www.twitch.tv/videos/1"}
    base.update(kw)
    return base


@pytest.fixture
def env(project):
    cfg = load_config(project / "config.yaml")
    state = StateStore(project / "data" / "state.json")
    clips_dir = project / "output" / "clips"
    clips_dir.mkdir(parents=True)

    def add(clip_id, **kw):
        mp4 = clips_dir / f"{clip_id}.mp4"
        mp4.write_bytes(b"mp4")
        mp4.with_suffix(".json").write_text(json.dumps(meta(clip_id=clip_id, **kw)))
        state.upsert_clip(clip_id, status="synced", path=str(mp4))
        return mp4

    return cfg, state, add


class FakeBackend:
    def __init__(self, name="fake", fail_after=None):
        self.name, self.calls, self.fail_after = name, [], fail_after

    def publish(self, post, *, test):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise PB.QuotaExceeded("quota")
        self.calls.append((post.clip.clip_id, post.platform, post.slot, test, post.title))
        return PB.PublishResult(f"id{len(self.calls)}", f"https://x/{len(self.calls)}", "private" if test else
                                ("scheduled" if post.slot else "published"), "private")


# ---------------------------------------------------------------- validation monetize
def test_validate_clip_rules(tmp_path):
    rules = MZ.Rules()
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    assert MZ.validate_clip(meta(), rules, clip).ok
    bad = MZ.validate_clip(meta(duration=31, hook=None, transformations=["vertical_reframe"], width=1920,
                                height=1080), rules, tmp_path / "missing.mp4")
    text = " ".join(bad.reasons)
    assert not bad.ok and "trop long" in text and "aucun hook" in text and "re-upload" in text
    assert "non vertical" in text and "absent" in text
    assert "hook après" in " ".join(MZ.validate_clip(meta(hook_at_seconds=1.5), rules).reasons)


# ---------------------------------------------------------------- créneaux & plan
def test_next_slots_respects_daily_cap_and_taken():
    times = PB.slot_times({"slots": {"eu": ["08:30", "11:00"], "us": ["00:30", "01:30", "02:30", "03:30"]}})
    taken = [datetime(2026, 11, 22, 11, 0, tzinfo=PARIS)]
    slots = PB.next_slots(5, now=NOW, tz=PARIS, times=times, max_per_day=2, taken=taken)
    local = [s.astimezone(PARIS) for s in slots]
    assert local[0] == datetime(2026, 11, 22, 8, 30, tzinfo=PARIS)          # 08:30 > 08:00 + 20 min
    per_day = {}
    for s in local + [t for t in taken]:
        per_day[s.date()] = per_day.get(s.date(), 0) + 1
    assert max(per_day.values()) <= 2 and len(set(local)) == 5 and taken[0] not in local


def test_plan_interleaves_variants_and_rotates_platforms():
    def c(cid, group, variant, score):
        return PB.ClipInfo(cid, Path(cid), {"video_id": group, "moment_idx": 0, "variant": variant, "score": score})

    clips = [c("a0", "A", 0, 70), c("a1", "A", 1, 70), c("b0", "B", 0, 95), c("b1", "B", 1, 95), c("b2", "B", 2, 95)]
    plan = PB.plan_posts(clips, ["youtube_shorts", "tiktok"], already={"b1@tiktok"})
    assert [(x.clip_id, p) for x, p in plan] == [
        ("b0", "youtube_shorts"), ("a0", "youtube_shorts"),        # meilleurs moments d'abord, variantes 0
        ("b1", "youtube_shorts"), ("a1", "tiktok"),                # b1@tiktok déjà publié -> autre plateforme
        ("b2", "youtube_shorts")]


def test_metadata_has_credit_and_limits():
    clip = PB.ClipInfo("x", Path("x"), meta(hook="H" * 120))
    title, desc, tags = PB.build_metadata(clip, {"hashtags": ["#gta6", "#shorts"]})
    assert len(title) <= 100 and title.endswith("#gta6 #shorts")
    assert "Credit: @kaicenat" in desc and "twitch.tv/videos/1" in desc and "kaicenat" in tags


# ---------------------------------------------------------------- publish
def test_publish_test_mode_private_youtube_only(env):
    cfg, state, add = env
    add("twitch_1_0")
    add("twitch_1_0_h1", variant=1, hook="No way 💀")
    add("twitch_1_1", moment_idx=1, duration=45)                      # trop long -> refusé
    yt = FakeBackend("youtube_api")
    rep = PB.publish(cfg, backends={"youtube_shorts": yt}, now=NOW)
    assert [c[0] for c in yt.calls] == ["twitch_1_0", "twitch_1_0_h1"]
    assert all(test and slot is None for _, _, slot, test, _ in yt.calls)   # test : immédiat, privé
    assert rep.rejected[0][0] == "twitch_1_1"
    rejected = json.loads((Path(cfg["_meta"]["root"]) / "data" / "rejected.json").read_text())
    assert "trop long" in rejected["rejected"]["twitch_1_1"]["reasons"][0]
    assert state.get_clip("twitch_1_0")["status"] == "published" and state.get_clip("twitch_1_1")["status"] == "rejected"
    # 2e run : rien n'est republié
    yt2 = FakeBackend("youtube_api")
    PB.publish(cfg, backends={"youtube_shorts": yt2}, now=NOW)
    assert yt2.calls == []


def test_publish_live_schedules_on_slots_with_daily_cap(env):
    cfg, state, add = env
    for i in range(4):
        for k in range(2):
            add(f"v_{i}{'_h1' if k else ''}", moment_idx=i, variant=k, score=90 - i)
    cfg["publish"]["max_posts_per_day"] = 3
    yt, tt = FakeBackend("youtube_api"), FakeBackend("upload_post")
    rep = PB.publish(cfg, test=False, backends={"youtube_shorts": yt, "tiktok": tt}, now=NOW,
                     platforms=["youtube_shorts", "tiktok"])
    slots = sorted(p["slot"] for p in rep.posted)
    assert len(rep.posted) == 8 and len(set(slots)) == 8                # chaque post a son créneau
    days = {}
    for s in slots:
        d = datetime.fromisoformat(s).date()
        days[d] = days.get(d, 0) + 1
    assert max(days.values()) <= 3
    assert {c[1] for c in tt.calls} == {"tiktok"} and len(tt.calls) == 4   # variantes _h1 -> TikTok
    assert state.get_clip("v_0")["status"] == "scheduled"
    assert len(state.load()["posts"]) == 8


def test_quota_exhaustion_blocks_platform_but_not_run(env):
    cfg, _, add = env
    for i in range(3):
        add(f"q_{i}", moment_idx=i)
    yt = FakeBackend("youtube_api", fail_after=1)
    rep = PB.publish(cfg, backends={"youtube_shorts": yt}, now=NOW)
    assert len(rep.posted) == 1 and len(rep.skipped) == 2 and "quota" in rep.skipped[0][1]


def test_dry_run_writes_nothing(env):
    cfg, state, add = env
    add("d_0", duration=50)
    add("d_1", moment_idx=1)
    rep = PB.publish(cfg, dry_run=True, backends={"youtube_shorts": PB.DryRunBackend()}, now=NOW)
    assert [p["status"] for p in rep.posted] == ["planned"] and rep.rejected
    assert "posts" not in state.load() and state.get_clip("d_0")["status"] == "synced"
    assert not (Path(cfg["_meta"]["root"]) / "data" / "rejected.json").exists()


# ---------------------------------------------------------------- backends réels (mockés)
class _InsertRequest:
    def __init__(self, body):
        self.body = body

    def next_chunk(self, num_retries=0):
        return None, {"id": "VID123", "status": {"privacyStatus": self.body["status"]["privacyStatus"]}}


class FakeYTService:
    def __init__(self):
        self.bodies = []

    def videos(self):
        outer = self

        class V:
            def insert(self, part, body, media_body):
                outer.bodies.append(body)
                return _InsertRequest(body)
        return V()


def test_youtube_backend_body_and_quota(env, tmp_path):
    cfg, _, add = env
    mp4 = add("y_0")
    svc = FakeYTService()
    quota = YouTubeQuota(tmp_path / "q.json", daily_limit=10000, safety_margin=200)
    be = PB.YouTubeBackend(cfg, service=svc, quota=quota)
    clip = PB.ClipInfo("y_0", mp4, meta())
    slot = datetime(2026, 11, 23, 1, 30, tzinfo=PARIS)
    r = be.publish(PB.Post(clip, "youtube_shorts", slot, "T #gta6", "D", ["gta6"]), test=False)
    assert r.status == "scheduled" and r.url == "https://youtube.com/shorts/VID123"
    st = svc.bodies[0]["status"]
    assert st["privacyStatus"] == "private" and st["publishAt"] == "2026-11-23T00:30:00Z"
    assert svc.bodies[0]["snippet"]["categoryId"] == "20" and quota.used() == 1600
    be.publish(PB.Post(clip, "youtube_shorts", None, "T", "D", []), test=True)
    assert svc.bodies[1]["status"] == {"selfDeclaredMadeForKids": False, "privacyStatus": "private"}
    quota.spend(10000 - 200 - quota.used() - 100, "videos.insert")
    with pytest.raises(PB.QuotaExceeded):
        be.publish(PB.Post(clip, "youtube_shorts", None, "T", "D", []), test=True)


def test_upload_post_backend_request(env, monkeypatch):
    cfg, _, add = env
    mp4 = add("u_0")
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["Authorization"]
        seen["body"] = req.content
        return httpx.Response(200, json={"request_id": "r1"})

    monkeypatch.setenv("UPLOAD_POST_API_KEY", "k")
    monkeypatch.setenv("UPLOAD_POST_USER", "me")
    be = PB.UploadPostBackend(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    slot = datetime(2026, 11, 23, 8, 30, tzinfo=PARIS)
    r = be.publish(PB.Post(PB.ClipInfo("u_0", mp4, meta()), "tiktok", slot, "Title", "D"), test=False)
    assert r.status == "scheduled" and r.remote_id == "r1" and seen["auth"] == "Apikey k"
    assert b'name="platform[]"' in seen["body"] and b"tiktok" in seen["body"] and b"scheduled_date" in seen["body"]
    assert be.publish(PB.Post(PB.ClipInfo("u_0", mp4, meta()), "tiktok", None), test=True).status == "skipped"
    monkeypatch.delenv("UPLOAD_POST_API_KEY")
    assert "tiktok" not in PB.build_backends(cfg, ["tiktok"], dry_run=False)
