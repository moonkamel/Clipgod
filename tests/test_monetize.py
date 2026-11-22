import json
from pathlib import Path

import httpx
import pytest

from core.config import load_config
from core.storage import StateStore
from modules import monetize as M

NEXT_DATA_PAGE = """<html><body><div>Content Rewards</div>
<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"campaigns":[
 {"id":"c1","slug":"kai-gta6","title":"Kai Cenat GTA 6","ratePerThousandViewsCents":150,"budgetCents":2000000,
  "paidOutCents":750000,"platforms":["TikTok","YouTube Shorts"],"status":"active",
  "requirements":"Clips must be under 60 seconds. Captions required. Use #kaicenat and tag @kaicenat"},
 {"id":"c2","slug":"low-cpm","title":"Low","cpm":"$0.20","budget":"$500","status":"active"},
 {"id":"c3","slug":"ended","title":"Old","cpm":2.0,"budget":1000,"status":"ended"}]}}}</script></body></html>"""

FLIGHT_PAGE = """<html><script>self.__next_f.push([1,"{\\"slug\\":\\"xqc-clips\\",\\"name\\":\\"xQc clips\\",\\"cpm\\":1.1,\\"budgetRemaining\\":3200}"])</script></html>"""

TEXT_PAGE = """<html><head><style>.x{}</style></head><body><h1>GTA VI Clipping</h1>
<p>Earn $2.50 per 1,000 views. $15,000 budget &middot; $4,200 remaining.</p>
<p>Requirements: between 10 and 45 seconds, post on TikTok or Instagram Reels, no reuploads, include #gta6</p>
</body></html>"""


def test_money_parsing():
    assert M.parse_money("$1,234.50") == 1234.5 and M.parse_money("2.5k") == 2500
    assert M.parse_money(150, cents=True) == 1.5 and M.parse_money(None) is None


def test_parse_next_data_listing():
    found = {c.slug: c for c in M.parse_page(NEXT_DATA_PAGE)}
    kai = found["kai-gta6"]
    assert kai.cpm_usd == 1.5 and kai.budget_total_usd == 20000 and kai.budget_remaining_usd == 12500
    assert set(kai.platforms) == {"tiktok", "youtube_shorts"}
    assert kai.rules["max_duration_seconds"] == 60 and kai.rules["require_subtitles"] is True
    assert kai.rules["required_hashtags"] == ["#kaicenat"] and kai.rules["required_mentions"] == ["@kaicenat"]
    assert found["low-cpm"].cpm_usd == 0.2 and found["ended"].active is False


def test_parse_flight_and_text_pages():
    flight = M.parse_page(FLIGHT_PAGE)
    assert flight[0].slug == "xqc-clips" and flight[0].cpm_usd == 1.1 and flight[0].budget_remaining_usd == 3200
    c = M.parse_page(TEXT_PAGE, slug="gta-vi", url="https://whop.com/gta-vi/")[0]
    assert c.cpm_usd == 2.5 and c.budget_total_usd == 15000 and c.budget_remaining_usd == 4200
    assert c.rules["min_duration_seconds"] == 10 and c.rules["max_duration_seconds"] == 45
    assert c.rules["require_transformation"] is True and c.rules["required_hashtags"] == ["#gta6"]
    assert set(c.platforms) == {"tiktok", "instagram_reels"}


@pytest.fixture
def cfg(project):
    c = load_config(project / "config.yaml")
    (project / "data").mkdir(exist_ok=True)
    return c


def _write_manual(cfg, text):
    (Path(cfg["_meta"]["root"]) / "data" / "whop_campaigns.yaml").write_text(text)


def test_scrape_listing_and_followed_pages_with_manual_override(cfg, monkeypatch):
    root = Path(cfg["_meta"]["root"])
    (root / "data" / "campaigns.txt").write_text("whop:gta-vi  # suivie\ntwitch:kaicenat\n")
    _write_manual(cfg, "- slug: kai-gta6\n  cpm_usd: 1.8\n  streamers: ['@KaiCenat']\n")
    pages = {"/discover/content-rewards/": NEXT_DATA_PAGE, "/gta-vi/": TEXT_PAGE}
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, text=pages.get(req.url.path, "")) if req.url.path in pages else httpx.Response(404)

    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = {c.slug: c for c in M.fetch_campaigns(cfg, client=client)}
    assert calls == ["/discover/content-rewards/", "/gta-vi/"]
    assert found["kai-gta6"].cpm_usd == 1.8 and found["kai-gta6"].budget_remaining_usd == 12500   # manuel prime
    assert found["kai-gta6"].streamers == ["kaicenat"] and found["gta-vi"].cpm_usd == 2.5
    # cache : pas de 2e appel réseau ; allow_scrape=False n'appelle jamais le réseau
    M.fetch_campaigns(cfg, client=client)
    M.fetch_campaigns(cfg, client=client, refresh=True, allow_scrape=False)
    assert len(calls) == 2


def test_rank_uses_history_and_budget(cfg):
    root = Path(cfg["_meta"]["root"])
    (root / "data" / "historical_scores.csv").write_text(
        "clip_id,signals,hook,platform,views_1h,views_6h,views_24h,views_72h,campaign\n"
        "a,{},h,tiktok,1,2,3,10000,kai-gta6\nb,{},h,tiktok,1,2,3,30000,kai-gta6\nc,{},h,tiktok,1,2,3,1000,other\n")
    camps = [M.Campaign("kai-gta6", "Kai", cpm_usd=1.5, budget_remaining_usd=12500),
             M.Campaign("big-cpm", "Big", cpm_usd=5.0, budget_remaining_usd=20),
             M.Campaign("unknown", "?", cpm_usd=None)]
    rows = M.rank_campaigns(cfg, camps)
    kai = next(r for r in rows if r["slug"] == "kai-gta6")
    assert kai["expected_views"] == 20000 and kai["usd_per_clip"] == 30.0 and kai["usd_per_hour"] == 90.0
    assert rows[0]["slug"] == "kai-gta6" and kai["eligible"]
    big = next(r for r in rows if r["slug"] == "big-cpm")
    assert not big["eligible"] and big["usd_per_hour"] <= 20 and big["views_source"] == "global"
    assert "CPM inconnu" in next(r for r in rows if r["slug"] == "unknown")["excluded_because"]


def test_rules_for_campaign_and_matching(cfg):
    c = M.Campaign("kai", platforms=["tiktok"], streamers=["kaicenat"],
                   rules={"max_duration_seconds": 60, "require_subtitles": True, "required_hashtags": ["#kai"]})
    rules = M.rules_for(cfg, c)
    assert rules.max_duration_seconds == 60 and rules.platforms == ["tiktok"] and rules.campaign == "kai"
    assert M.rules_for(cfg, None).max_duration_seconds == 30
    assert M.match_campaign([c], streamer="@KaiCenat") is c and M.match_campaign([c], slug="nope") is None
    meta = {"duration": 45, "hook": "h", "hook_at_seconds": 0, "width": 1080, "height": 1920,
            "transformations": ["vertical_reframe", "hook_overlay", "facecam_split"]}
    v = M.validate_clip(meta, rules)
    assert not v.ok and v.reasons == ["sous-titres exigés par la campagne"]       # 45 s OK pour cette campagne


def test_publish_applies_campaign_rules(cfg):
    from modules import publish as PB
    from tests.test_publish import FakeBackend, meta

    _write_manual(cfg, "- slug: kai\n  cpm_usd: 1.5\n  platforms: [tiktok]\n  streamers: [kaicenat]\n"
                       "  required_hashtags: ['#kaicenat']\n  required_mentions: ['@kaicenat']\n")
    root = Path(cfg["_meta"]["root"])
    clips = root / "output" / "clips"
    clips.mkdir(parents=True)
    state = StateStore(root / "data" / "state.json")
    for cid in ("k_0", "k_0_h1"):
        (clips / f"{cid}.mp4").write_bytes(b"x")
        (clips / f"{cid}.json").write_text(json.dumps(meta(variant=1 if "h1" in cid else 0)))
        state.upsert_clip(cid, status="synced", path=str(clips / f"{cid}.mp4"))
    yt, tt = FakeBackend("youtube_api"), FakeBackend("upload_post")
    rep = PB.publish(cfg, test=False, platforms=["youtube_shorts", "tiktok"],
                     backends={"youtube_shorts": yt, "tiktok": tt})
    assert yt.calls == [] and len(tt.calls) == 2                       # campagne : TikTok uniquement
    title = tt.calls[0][4]
    assert "#kaicenat" in title and "@kaicenat" in title and len(title) <= 100
    assert all(p["campaign"] == "kai" for p in rep.posted)


def test_cli_monetize_table_and_validate(cfg, tmp_path):
    from typer.testing import CliRunner

    import clipgod

    _write_manual(cfg, "- slug: kai\n  name: Kai GTA 6\n  cpm_usd: 1.5\n  budget_remaining_usd: 9000\n")
    runner = CliRunner()
    conf = str(Path(cfg["_meta"]["root"]) / "config.yaml")
    res = runner.invoke(clipgod.app, ["--config", conf, "--dry-run", "monetize"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "Kai GTA 6" in res.output and "--campaign kai" in res.output
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    clip.with_suffix(".json").write_text(json.dumps({"duration": 40, "hook": "h", "hook_at_seconds": 0,
                                                     "width": 1080, "height": 1920}))
    res = runner.invoke(clipgod.app, ["--config", conf, "monetize", "--validate", str(clip)], env={"COLUMNS": "200"})
    assert res.exit_code == 1 and "trop long" in res.output
    rejected = json.loads((Path(cfg["_meta"]["root"]) / "data" / "rejected.json").read_text())
    assert "c" in rejected["rejected"]
