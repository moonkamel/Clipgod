import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from core.config import load_config
from core.quota import QuotaExceeded, YouTubeQuota
from modules import discover as D

NOW = datetime(2026, 11, 22, 20, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


@pytest.fixture
def cfg(project, monkeypatch):
    for k in ("YOUTUBE_API_KEY", "TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    (project / "data").mkdir(exist_ok=True)
    (project / "data" / "campaigns.txt").write_text(
        "# commentaire\ntwitch:KaiCenat gta6-launch\nyoutube:@caseoh\nkick:xqc\nwhop:ignored\n")
    return load_config(project / "config.yaml")


# ---------------------------------------------------------------- utilitaires
def test_parsers_and_relevance():
    assert D.parse_iso8601_duration("PT3H2M1S") == 10921 and D.parse_iso8601_duration("P1DT1H") == 90000
    assert D.parse_twitch_duration("3h2m10s") == 10930 and D.parse_twitch_duration("45m") == 2700
    kw = ["GTA 6", "GTA VI", "Grand Theft Auto VI"]
    assert D.title_matches("FIRST DAY IN GTA VI!!", kw) and D.title_matches("gta 6 rp | cops", kw)
    assert not D.title_matches("GTA V RP chill", kw) and not D.title_matches("GTA 60fps", kw)


def test_campaigns_parsing(cfg):
    t = D.load_campaigns(Path(cfg["_meta"]["root"]) / "data/campaigns.txt")
    assert t.twitch == {"kaicenat": "gta6-launch"} and "caseoh" in t.youtube and "xqc" in t.kick
    assert t.campaign_for("twitch", "KaiCenat") == "gta6-launch" and t.campaign_for("twitch", "other") is None


def test_quota_tracker_resets_daily(tmp_path):
    day = [datetime(2026, 11, 22, 10, tzinfo=timezone.utc)]
    q = YouTubeQuota(tmp_path / "s.json", daily_limit=1000, safety_margin=100, now=lambda: day[0])
    q.spend(800, "search.list")
    assert q.remaining() == 100 and not q.can_spend(101)
    with pytest.raises(QuotaExceeded):
        q.spend(200, "videos.insert")
    day[0] += timedelta(days=1)
    assert q.remaining() == 900


# ---------------------------------------------------------------- Twitch (httpx mocké)
def twitch_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.startswith(D.TWITCH_TOKEN_URL):
        return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
    assert request.headers["Authorization"] == "Bearer tok" and request.headers["Client-Id"] == "cid"
    path, q = request.url.path, request.url.params
    if path.endswith("/games"):
        return httpx.Response(200, json={"data": [{"id": "999", "name": q["name"]}]})
    if path.endswith("/streams") and q.get("game_id"):
        return httpx.Response(200, json={"data": [
            {"id": "s1", "user_login": "bigstreamer", "title": "GTA 6 heists", "viewer_count": 30000,
             "started_at": iso(NOW - timedelta(hours=2)), "game_name": "Grand Theft Auto VI"},
            {"id": "s2", "user_login": "fresh", "title": "just started", "viewer_count": 9000,
             "started_at": iso(NOW - timedelta(minutes=10)), "game_name": "Grand Theft Auto VI"}]})
    if path.endswith("/streams"):
        return httpx.Response(200, json={"data": []})
    if path.endswith("/users"):
        return httpx.Response(200, json={"data": [{"id": "u1", "login": "kaicenat"}]})
    if path.endswith("/videos") and q.get("game_id"):
        return httpx.Response(200, json={"data": [
            {"id": "v1", "user_login": "midstreamer", "title": "day 3 chaos", "view_count": 50000,
             "created_at": iso(NOW - timedelta(hours=8)), "duration": "4h0m0s", "url": "https://www.twitch.tv/videos/v1"},
            {"id": "v2", "user_login": "old", "title": "GTA 6", "view_count": 90000,
             "created_at": iso(NOW - timedelta(hours=40)), "duration": "5h0m0s"}]})
    if path.endswith("/videos") and q.get("user_id") == "u1":
        return httpx.Response(200, json={"data": [
            {"id": "v3", "user_login": "kaicenat", "title": "GTA 6 w/ friends", "view_count": 800,
             "created_at": iso(NOW - timedelta(hours=5)), "duration": "1h30m0s"},
            {"id": "v4", "user_login": "kaicenat", "title": "GTA 6 short", "view_count": 800,
             "created_at": iso(NOW - timedelta(hours=3)), "duration": "20m0s"}]})
    return httpx.Response(404)


def test_twitch_discover_filters_and_ranks(cfg, monkeypatch):
    monkeypatch.setenv("TWITCH_CLIENT_ID", "cid")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "secret")
    client = httpx.Client(transport=httpx.MockTransport(twitch_handler))
    sources = D.discover(cfg, platforms=["twitch"], now=NOW,
                         fetchers={"twitch": lambda c, camp, **k: D.discover_twitch(c, camp, client=client, **k)})
    urls = [s["url"] for s in sources]
    # campagne d'abord (même sous min_viewers), puis vues/heure ; live trop récent, VOD trop vieille et trop courte exclus
    assert urls == ["https://www.twitch.tv/videos/v3", "https://www.twitch.tv/bigstreamer",
                    "https://www.twitch.tv/videos/v1"]
    assert sources[0]["campaign"] == "gta6-launch" and sources[1]["status"] == "live"
    assert sources[2]["duration"] == 14400 and sources[2]["views_per_hour"] == pytest.approx(12500, rel=0.01)
    doc = json.loads((Path(cfg["_meta"]["root"]) / "data/sources.json").read_text())
    assert doc["count"] == 3 and set(doc["sources"][0]) >= {"url", "title", "viewers", "duration", "streamer", "platform"}


def test_twitch_token_is_cached(cfg, monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return twitch_handler(req)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tw = D.TwitchDiscover(client, "cid", "secret", tmp_path / "tok.json")
    tw.token()
    D.TwitchDiscover(client, "cid", "secret", tmp_path / "tok.json").token()
    assert sum(D.TWITCH_TOKEN_URL in c for c in calls) == 1


def test_cache_avoids_second_call(cfg, monkeypatch):
    n = []

    def fake(c, camp, **k):
        n.append(1)
        return [D.SourceItem("https://www.twitch.tv/videos/9", "GTA 6", 5000, 4000, "x", "twitch", "vod",
                             iso(NOW - timedelta(hours=2)), 2500.0)]

    D.discover(cfg, platforms=["twitch"], now=NOW, fetchers={"twitch": fake})
    D.discover(cfg, platforms=["twitch"], now=NOW, fetchers={"twitch": fake})
    assert len(n) == 1
    D.discover(cfg, platforms=["twitch"], now=NOW, use_cache=False, fetchers={"twitch": fake})
    assert len(n) == 2


# ---------------------------------------------------------------- YouTube (service factice)
class _Req:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self.data


class FakeYouTube:
    def __init__(self):
        self.calls = []

    def search(self):
        outer = self

        class S:
            def list(self, **kw):
                outer.calls.append(("search", kw))
                vid = "LIVE1" if kw.get("eventType") == "live" else "VOD1"
                return _Req({"items": [{"id": {"videoId": vid}}]})
        return S()

    def channels(self):
        outer = self

        class C:
            def list(self, **kw):
                outer.calls.append(("channels", kw))
                return _Req({"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUx"}}}]})
        return C()

    def playlistItems(self):
        class P:
            def list(self, **kw):
                return _Req({"items": [{"contentDetails": {"videoId": "CAMP1"}}]})
        return P()

    def videos(self):
        outer = self

        class V:
            def list(self, **kw):
                outer.calls.append(("videos", kw))
                return _Req({"items": [
                    {"id": "LIVE1", "snippet": {"title": "GTA 6 LIVE", "channelTitle": "Streamy",
                                                "liveBroadcastContent": "live"},
                     "liveStreamingDetails": {"concurrentViewers": "12000",
                                              "actualStartTime": iso(NOW - timedelta(hours=3))}},
                    {"id": "VOD1", "snippet": {"title": "GTA VI full playthrough", "channelTitle": "Vids",
                                               "liveBroadcastContent": "none", "publishedAt": iso(NOW - timedelta(hours=10))},
                     "contentDetails": {"duration": "PT2H"}, "statistics": {"viewCount": "100000"}},
                    {"id": "CAMP1", "snippet": {"title": "GTA 6 stream VOD", "channelTitle": "caseoh",
                                                "liveBroadcastContent": "none", "publishedAt": iso(NOW - timedelta(hours=20))},
                     "contentDetails": {"duration": "PT3H"}, "statistics": {"viewCount": "300"}},
                    {"id": "UP1", "snippet": {"title": "GTA 6 premiere", "liveBroadcastContent": "upcoming"}},
                ]})
        return V()


def test_youtube_discover_uses_little_quota(cfg):
    yt = FakeYouTube()
    sources = D.discover(cfg, platforms=["youtube"], now=NOW, fetchers={
        "youtube": lambda c, camp, **k: D.discover_youtube(c, camp, service=yt, **k)})
    assert [s["url"].split("=")[-1] for s in sources] == ["CAMP1", "LIVE1", "VOD1"]
    assert sources[0]["campaign"] == "caseoh" and sources[1]["viewers"] == 12000
    assert sum(1 for c in yt.calls if c[0] == "search") == 2
    assert YouTubeQuota.from_config(cfg).used() == 100 * 2 + 1 + 1 + 1        # 2 search + channels + playlist + videos
    vod_search = next(kw for name, kw in yt.calls if name == "search" and "eventType" not in kw)
    assert vod_search["videoDuration"] == "long" and vod_search["publishedAfter"].startswith("2026-11-21T20:00:00")


def test_youtube_respects_daily_budget(cfg):
    cfg["discover"]["youtube"]["daily_budget"] = 150          # une seule recherche possible
    yt = FakeYouTube()
    D.discover_youtube(cfg, D.CampaignTargets(), now=NOW, service=yt)
    assert sum(1 for c in yt.calls if c[0] == "search") == 1


# ---------------------------------------------------------------- Kick (httpx mocké) + erreurs
def test_kick_and_platform_failures(cfg, monkeypatch):
    def handler(req):
        if req.url.path == "/api/v2/channels/xqc":
            return httpx.Response(200, json={"slug": "xqc", "user": {"username": "xQc"}, "livestream": None})
        if req.url.path == "/api/v2/channels/xqc/videos":
            return httpx.Response(200, json=[{"session_title": "GTA 6 RP day 2", "created_at": iso(NOW - timedelta(hours=6)),
                                              "duration": 3 * 3600 * 1000, "views": 70000,
                                              "video": {"uuid": "abc-123"},
                                              "categories": [{"name": "Grand Theft Auto VI"}]}])
        return httpx.Response(403)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    sources = D.discover(cfg, platforms=["kick", "twitch"], now=NOW, fetchers={
        "kick": lambda c, camp, **k: D.discover_kick(c, camp, client=client, **k)})
    assert [s["url"] for s in sources] == ["https://kick.com/xqc/videos/abc-123"]
    doc = json.loads((Path(cfg["_meta"]["root"]) / "data/sources.json").read_text())
    assert "TWITCH_CLIENT_ID" in doc["errors"]["twitch"]                   # Twitch sans clé : ignoré, pas fatal
    with pytest.raises(D.DiscoverError):
        D.discover(cfg, platforms=["twitch"], now=NOW)


def test_best_source_prefers_vod(cfg, monkeypatch):
    items = [D.SourceItem("https://www.twitch.tv/live", "GTA 6", 50000, 9000, "a", "twitch", "live",
                          iso(NOW - timedelta(hours=3)), 50000.0),
             D.SourceItem("https://www.twitch.tv/videos/7", "GTA 6", 9000, 9000, "b", "twitch", "vod",
                          iso(NOW - timedelta(hours=3)), 3000.0)]
    monkeypatch.setattr(D, "FETCHERS", {"twitch": lambda c, camp, **k: items})
    monkeypatch.setattr(D, "utc_now", lambda: NOW)
    cfg["discover"]["platforms"] = ["twitch"]
    assert D.best_source(cfg) == "https://www.twitch.tv/videos/7"


def _item(title, lang=None, duration=4 * 3600, status="vod"):
    return D.SourceItem("https://x/" + title, title, 9000, duration, "s", "twitch", status,
                        iso(NOW - timedelta(hours=2)), 4000.0, extra={"language": lang} if lang else {})


def test_language_filter_and_max_duration():
    dcfg = {"languages": ["en", "fr"], "keywords": ["GTA 6"], "max_duration_minutes": 300, "min_viewers": 0}
    assert D.keep(_item("GTA 6 heist", "en"), dcfg, now=NOW)
    assert D.keep(_item("GTA 6 braquage", "fr"), dcfg, now=NOW)
    assert not D.keep(_item("GTA 6 | DAY 5 | GABBAR SINGH IS BACK", "hi"), dcfg, now=NOW)   # langue déclarée
    assert not D.keep(_item("GTA 6 | قراند ثفت أوتو مباشر بث"), dcfg, now=NOW)              # écriture non latine
    assert D.keep(_item("GTA 6 first day"), dcfg, now=NOW)                                   # langue inconnue
    assert not D.keep(_item("GTA 6 marathon", "en", duration=10 * 3600), dcfg, now=NOW)       # > 5 h
    assert D.keep(_item("GTA 6 live", "en", duration=10 * 3600, status="live"), dcfg, now=NOW)  # live : pas de max
    assert D.keep(_item("GTA 6 | DAY 5", "hi"), {**dcfg, "languages": []}, now=NOW)


def test_youtube_language_and_relevance_param(cfg):
    assert D._yt_language({"defaultAudioLanguage": "en-US"}) == "en" and D._yt_language({}) is None
    yt = FakeYouTube()
    D.discover_youtube(cfg, D.CampaignTargets(), now=NOW, service=yt)
    assert all(kw.get("relevanceLanguage") == "en" for name, kw in yt.calls if name == "search")
