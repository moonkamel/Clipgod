import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.config import load_config
from core.scoring import FEATURES, ModelScorer, load_scorer
from core.storage import StateStore
from modules import learn as L

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(project):
    c = load_config(project / "config.yaml")
    (project / "data").mkdir(exist_ok=True)
    return c


def seed_posts(cfg, n=30, *, hours_ago=80, seed=0):
    """n posts YouTube (+ variantes A/B) dont les vues dépendent du signal text_hype_keywords."""
    rng = random.Random(seed)
    root = Path(cfg["_meta"]["root"])
    clips_dir = root / "output" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(root / "data" / "state.json")
    with state.transaction() as doc:
        for i in range(n):
            hype = rng.random()
            signals = {f: round(rng.random(), 3) for f in FEATURES} | {"text_hype_keywords": round(hype, 3)}
            cid = f"vid_{i}"
            mp4 = clips_dir / f"{cid}.mp4"
            mp4.write_bytes(b"x")
            mp4.with_suffix(".json").write_text(json.dumps({"signals": signals, "score": 50 + i}))
            doc.setdefault("clips", {})[cid] = {"status": "published", "path": str(mp4), "campaign": "kai"}
            pub = NOW - timedelta(hours=hours_ago)
            pid = f"{cid}@youtube_shorts"
            hook = '"NO WAY" 💀' if i % 2 else "Wait for it… 😳"
            doc.setdefault("posts", {})[pid] = {
                "clip_id": cid, "platform": "youtube_shorts", "remote_id": f"YT{i}", "url": f"https://youtube.com/shorts/YT{i}",
                "posted_at": pub.isoformat(), "slot": None, "status": "published", "hook": hook,
                "variant": i % 2, "group": f"g{i // 2}", "campaign": "kai"}
            final = int(500 + 20000 * hype)
            doc.setdefault("observations", {})[pid] = [
                {"t": (pub + timedelta(hours=2)).isoformat(), "views": final // 10},
                {"t": (pub + timedelta(hours=30)).isoformat(), "views": final},
                {"t": (pub + timedelta(hours=75)).isoformat(), "views": int(final * 1.3)}]
    return state


def test_views_at_interpolates_without_extrapolating():
    pub = NOW
    series = [{"t": (pub + timedelta(hours=2)).isoformat(), "views": 200},
              {"t": (pub + timedelta(hours=26)).isoformat(), "views": 2600}]
    assert L.views_at(series, pub, 1) == pytest.approx(100)               # entre 0 (publication) et 2 h
    assert L.views_at(series, pub, 24) == pytest.approx(2400)
    assert L.views_at(series, pub, 72) is None


class FakeYT:
    def __init__(self, views):
        self.views, self.calls = views, 0

    def videos(self):
        outer = self

        class V:
            def list(self, **kw):
                outer.calls += 1
                ids = kw["id"].split(",")

                class R:
                    def execute(self_inner):
                        return {"items": [{"id": i, "statistics": {"viewCount": str(outer.views[i])}}
                                          for i in ids if i in outer.views]}
                return R()
        return V()


def test_collect_youtube_and_manual(cfg):
    root = Path(cfg["_meta"]["root"])
    state = StateStore(root / "data" / "state.json")
    with state.transaction() as doc:
        doc["posts"] = {
            "a@youtube_shorts": {"clip_id": "a", "platform": "youtube_shorts", "remote_id": "YA",
                                 "posted_at": (NOW - timedelta(hours=3)).isoformat()},
            "a@tiktok": {"clip_id": "a", "platform": "tiktok", "remote_id": "r1", "url": "https://tiktok.com/@x/1",
                         "posted_at": (NOW - timedelta(hours=3)).isoformat()},
            "old@youtube_shorts": {"clip_id": "o", "platform": "youtube_shorts", "remote_id": "YO",
                                   "posted_at": (NOW - timedelta(days=10)).isoformat()}}
    (root / "data" / "manual_views.csv").write_text(
        f"url,views,measured_at\nhttps://tiktok.com/@x/1,\"12,500\",{(NOW - timedelta(hours=1)).isoformat()}\n")
    yt = FakeYT({"YA": 4200, "YO": 99})
    rep = L.collect(cfg, now=NOW, youtube_service=yt)
    assert rep.observed == 2 and rep.by_platform == {"youtube_shorts": 1, "tiktok": 1}
    obs = state.load()["observations"]
    assert obs["a@youtube_shorts"][0]["views"] == 4200 and obs["a@tiktok"][0]["views"] == 12500
    assert "old@youtube_shorts" not in obs                                    # > 72 h : plus suivi
    L.collect(cfg, now=NOW, youtube_service=yt)                               # même instant : pas de doublon
    assert len(state.load()["observations"]["a@youtube_shorts"]) == 1


def test_history_csv(cfg):
    seed_posts(cfg, n=4)
    rows = L.build_history(cfg)
    path = L.write_history(cfg, rows)
    with open(path) as fh:
        data = list(csv.DictReader(fh))
    assert list(data[0])[:8] == ["clip_id", "signals", "hook", "platform", "views_1h", "views_6h", "views_24h", "views_72h"]
    assert all(r["views_24h"] and r["views_72h"] and r["campaign"] == "kai" for r in data)
    assert json.loads(data[0]["signals"])["text_hype_keywords"] >= 0


def test_should_retrain_cadence(cfg):
    assert L.should_retrain(cfg, 5, now=NOW)[0] is False
    assert L.should_retrain(cfg, 12, now=NOW) == (True, "premier entraînement")
    state = StateStore(Path(cfg["_meta"]["root"]) / "data" / "state.json")
    with state.transaction() as d:
        d["model_meta"] = {"trained_at": (NOW - timedelta(days=2)).isoformat(), "n_samples": 12}
    assert L.should_retrain(cfg, 20, now=NOW)[0] is False
    assert L.should_retrain(cfg, 32, now=NOW) == (True, "20 nouveaux clips")
    assert L.should_retrain(cfg, 13, now=NOW + timedelta(days=6))[1] == "réentraînement hebdomadaire"


def test_learn_trains_model_used_by_next_scan(cfg):
    seed_posts(cfg, n=40)
    res = L.learn(cfg, collect_views=False, push=False, now=NOW)
    assert res.trained and res.model_meta["n_samples"] == 40 and res.model_meta["cv_r2"] > 0.3
    assert max(res.model_meta["importances"], key=res.model_meta["importances"].get) == "text_hype_keywords"
    # Le scan charge data/model.pkl automatiquement (paths.model) et suit ce qu'il a appris.
    scorer = load_scorer(cfg)
    assert isinstance(scorer, ModelScorer) and scorer.n_samples == 40
    import numpy as np

    X = np.zeros((2, len(FEATURES)))
    X[0, FEATURES.index("text_hype_keywords")] = 1.0
    X[1, FEATURES.index("visual_motion")] = 1.0
    s = scorer.score(X, FEATURES)
    assert s[0] > s[1]
    # 2e run : pas de nouveau clip -> pas de réentraînement
    assert L.learn(cfg, collect_views=False, push=False, now=NOW).trained is False


def test_report_content(cfg):
    seed_posts(cfg, n=20)
    manual = Path(cfg["_meta"]["root"]) / "data" / "whop_campaigns.yaml"
    manual.write_text("- slug: kai\n  cpm_usd: 2.0\n")
    res = L.learn(cfg, collect_views=False, push=False, now=NOW, retrain=True)
    doc = res.report_path.read_text()
    for section in ("Recommandations", "Top clips", "Flop clips", "Hooks gagnants", "Styles de hook", "Créneaux",
                    "Campagnes les plus rentables", "Modèle de viralité"):
        assert section in doc
    stats = L.analyze(cfg, L.build_history(cfg))
    assert stats["n_measured"] == 20 and stats["ab"] and stats["campaigns"][0]["campaign"] == "kai"
    assert stats["campaigns"][0]["earned"] > 0 and {h["key"] for h in stats["hook_styles"]} == {"citation", "suspense"}
    assert "<script" not in doc                                            # rapport autonome, sans JS


def test_hook_style():
    assert L.hook_style('"NO WAY" 💀') == "citation" and L.hook_style('kai: "bro"') == "citation"
    assert L.hook_style("Wait for it… 😳") == "suspense" and L.hook_style("Who did this?") == "question"
    assert L.hook_style("Nobody expected this") == "superlatif" and L.hook_style(None) == "aucun"


def test_cli_learn(cfg):
    from typer.testing import CliRunner

    import clipgod

    seed_posts(cfg, n=12)
    res = CliRunner().invoke(clipgod.app, ["--config", str(Path(cfg["_meta"]["root"]) / "config.yaml"), "learn",
                                           "--no-collect", "--no-push"], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "model.pkl réentraîné" in res.output and "report.html" in res.output
