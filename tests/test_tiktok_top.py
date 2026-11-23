"""Veille TikTok : parsing pur + vrai navigateur (Chromium) sur une fausse page de hashtag servie en local."""

import http.server
import json
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone

import pytest

from core.config import load_config
from modules import tiktok_top as T

NOW = datetime(2026, 11, 23, 18, 0, tzinfo=timezone.utc)


def vid_at(dt, low=12345):
    """Identifiant TikTok publié à ``dt`` (32 bits de poids fort = secondes Unix)."""
    return str((int(dt.timestamp()) << 32) | low)


@pytest.mark.parametrize("text,expected", [
    ("1.2M", 1_200_000), ("845.3K", 845_300), ("1,2 M", 1_200_000), ("12 345", 12345), ("987", 987),
    ("2B", 2_000_000_000), ("", 0), ("n/a", 0), ("3,4 k", 3400),
])
def test_parse_count(text, expected):
    assert T.parse_count(text) == expected


def test_id_timestamp_and_parse_items():
    t = NOW - timedelta(hours=5)
    vid = vid_at(t)
    assert T.id_timestamp(vid) == t
    items = T.parse_items([
        {"href": f"https://www.tiktok.com/@kai/video/{vid}?is_from_webapp=1", "views": "1.1M", "text": ""},
        {"href": f"https://www.tiktok.com/@kai/video/{vid}", "views": "1.2M", "text": "GTA 6 crash #gta6"},
        {"href": "https://www.tiktok.com/@kai", "views": "9M", "text": "profil"},
    ])
    assert len(items) == 1 and items[0].views == 1_200_000 and items[0].text == "GTA 6 crash #gta6"
    assert items[0].url == f"https://www.tiktok.com/@kai/video/{vid}"


def test_rank_filters_age_and_gameplay(project):
    cfg = load_config(project / "config.yaml")
    mk = lambda h, views, text: T.TikTokVideo(vid_at(NOW - timedelta(hours=h)), "a", "u", views, text,
                                              NOW - timedelta(hours=h))
    videos = [mk(5, 900, "GTA 6 helicopter crash physics"), mk(10, 5000, "GTA 5 vs GTA 6 graphics"),
              mk(3, 3000, "police chase GTA 6 #gta6"), mk(80, 99999, "GTA 6 crash old")]
    kept, dropped = T.rank(videos, now=NOW, hours=48, count=10, cfg=cfg)
    assert [v.views for v, _ in kept] == [3000, 900]                  # trié par vues, trop vieux exclu
    assert [v.views for v, _ in dropped] == [5000]                    # comparaison écartée


PAGE = """<!doctype html><html><body><div id="grid"></div><script>
const all = %s; let shown = 0;
function more(){ const g = document.getElementById('grid');
  for (const v of all.slice(shown, shown + 4)) {
    const d = document.createElement('div'); d.setAttribute('data-e2e', 'challenge-item');
    d.innerHTML = `<a href="/@${v.a}/video/${v.id}"><img alt="${v.t}"><strong data-e2e="video-views">${v.v}</strong></a>`;
    d.style.height = '600px'; g.appendChild(d); }
  shown += 4; }
setTimeout(more, 300);                        // chargement différé, comme TikTok
window.addEventListener('scroll', () => { if (innerHeight + scrollY >= document.body.scrollHeight - 50) more(); });
</script></body></html>"""


@pytest.fixture
def fake_tiktok(tmp_path):
    now = datetime.now(timezone.utc)
    data = [{"a": f"user{i}", "id": vid_at(now - timedelta(hours=i + 1)), "v": f"{(i + 1) * 100}K",
             "t": ("GTA 6 crash moment #gta6" if i % 2 else "GTA 6 news leak")} for i in range(10)]
    (tmp_path / "tag").mkdir()
    (tmp_path / "tag" / "gta6").write_text(PAGE % json.dumps(data), encoding="utf-8")

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path), **k)

        def guess_type(self, path):
            return "text/html"

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", data
    srv.shutdown()


CHROMIUM = os.environ.get("CLIPGOD_TEST_CHROMIUM", "/opt/pw-browsers/chromium")


@pytest.mark.skipif(not shutil.which(CHROMIUM) and not os.path.exists(CHROMIUM), reason="Chromium requis")
def test_collect_with_real_browser(fake_tiktok, tmp_path, project):
    pytest.importorskip("playwright")
    base, data = fake_tiktok
    raw = T.collect(["gta6"], profile_dir=tmp_path / "profile", scrolls=3, headless=True, base_url=base,
                    executable_path=CHROMIUM, scroll_pause_ms=300, wait_seconds=10, say=lambda m: None)
    videos = T.parse_items(raw)
    assert len(videos) >= 8                                           # le défilement charge la suite
    by_id = {v.id: v for v in videos}
    first = by_id[data[0]["id"]]
    assert first.views == 100_000 and first.author == "user0" and first.text == "GTA 6 news leak"

    cfg = load_config(project / "config.yaml")
    out, kept = T.run_toptiktok(cfg, hours=48, count=3, output=tmp_path / "top.txt",
                                collect_fn=lambda tags, **kw: raw, say=lambda m: None)
    assert [v.views for v, _ in kept] == [1_000_000, 800_000, 600_000]   # gameplay seulement, par vues
    assert "news leak" in out.read_text(encoding="utf-8-sig").split("Écartées")[1]
