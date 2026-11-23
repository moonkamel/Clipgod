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
    kept, dropped = T.rank(videos, now=NOW, hours=48, count=10, cfg=cfg, languages=[])
    assert [v.views for v, _ in kept] == [3000, 900]                  # trié par vues, trop vieux exclu
    assert [v.views for v, _ in dropped] == [5000]                    # comparaison écartée


PAGE = """<!doctype html><html><body><div id="grid"></div><script>
// Comme TikTok : la liste arrive en JSON (/api/challenge/item_list), les vignettes n'affichent pas les vues.
let cursor = 0;
async function more(){ const r = await fetch('/api/challenge/item_list/?cursor=' + cursor); const d = await r.json();
  cursor += d.itemList.length; const g = document.getElementById('grid');
  for (const v of d.itemList) { const el = document.createElement('div'); el.setAttribute('data-e2e', 'challenge-item');
    el.innerHTML = `<a href="/@${v.author.uniqueId}/video/${v.id}"><img alt="${v.desc} created by X with Y"></a>`;
    el.style.height = '600px'; g.appendChild(el); } }
setTimeout(more, 300);
window.addEventListener('scroll', () => { if (innerHeight + scrollY >= document.body.scrollHeight - 50) more(); });
</script></body></html>"""

DETAIL = """<!doctype html><html><body><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">%s
</script></body></html>"""


def _items(now):
    """10 vidéos : paires = anglais gameplay, 3 = français, 5 = sans vues ni langue dans la liste (page vidéo)."""
    out = []
    for i in range(10):
        text = {3: "Où trouver la Bentley en or dans GTA 6 #gta6"}.get(i, f"Insane police chase in GTA 6 #{i}")
        it = {"id": vid_at(now - timedelta(hours=i + 1)), "desc": text, "author": {"uniqueId": f"user{i}"},
              "stats": {"playCount": (i + 1) * 100_000}, "textLanguage": "fr" if i == 3 else "en"}
        if i == 5:
            it.pop("stats"), it.pop("textLanguage")
            it["statsV2"] = {}
        out.append(it)
    return out


@pytest.fixture
def fake_tiktok(tmp_path):
    now = datetime.now(timezone.utc)
    items = _items(now)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/challenge/item_list"):
                cursor = int(self.path.split("cursor=")[1])
                listed = [dict(it) for it in items[cursor:cursor + 4]]
                body, ctype = json.dumps({"itemList": listed}), "application/json"
            elif self.path.startswith("/tag/"):
                body, ctype = PAGE, "text/html"
            elif "/video/" in self.path:
                vid = self.path.rsplit("/", 1)[1]
                it = next(i for i in items if i["id"] == vid)
                full = {**it, "stats": {"playCount": 2_500_000}, "textLanguage": "en"} if it is items[5] else it
                body = DETAIL % json.dumps({"__DEFAULT_SCOPE__": {"webapp.video-detail": {"itemInfo": {
                    "itemStruct": full}}}})
                ctype = "text/html"
            else:
                self.send_response(404)
                self.end_headers()
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", items
    srv.shutdown()


CHROMIUM = os.environ.get("CLIPGOD_TEST_CHROMIUM", "/opt/pw-browsers/chromium")


@pytest.mark.skipif(not shutil.which(CHROMIUM) and not os.path.exists(CHROMIUM), reason="Chromium requis")
def test_collect_reads_hidden_list_then_details(fake_tiktok, tmp_path, project):
    pytest.importorskip("playwright")
    base, items = fake_tiktok
    raw = T.collect(["gta6"], profile_dir=tmp_path / "profile", scrolls=3, headless=True, base_url=base,
                    executable_path=CHROMIUM, scroll_pause_ms=400, wait_seconds=10, say=lambda m: None)
    videos = {v.id: v for v in T.parse_items(raw)}
    assert len(videos) >= 8
    first = videos[items[0]["id"]]
    assert first.views == 100_000 and first.lang == "en" and first.text == "Insane police chase in GTA 6 #0"
    assert videos[items[5]["id"]].views == 0                          # pas de vues dans la liste…

    def details(urls, **kw):                                         # … lues sur la page de la vidéo
        local = [u.replace("https://www.tiktok.com", base) for u in urls]
        kw["headless"] = True
        return T.fetch_details(local, executable_path=CHROMIUM, pause_ms=200, **kw)

    cfg = load_config(project / "config.yaml")
    out, kept = T.run_toptiktok(cfg, hours=48, count=3, output=tmp_path / "top.txt",
                                collect_fn=lambda tags, **kw: raw, details_fn=details, say=lambda m: None)
    assert [v.views for v, _ in kept] == [2_500_000, 1_000_000, 900_000]   # n°5 récupéré via sa page
    text = out.read_text(encoding="utf-8-sig")
    assert "2 500 000 vues" in text and "langue : fr" in text            # la vidéo française est écartée


def test_guess_language():
    assert T.guess_language("Ou trouver la BENTLEY en OR dans GTA 6") == "fr"
    assert T.guess_language("THE MOST INSANE LOOT LOCATIONS IN GTA 6! Find a Rare Mercedes") == "en"
    assert T.guess_language("yanlış karar vermek deyince de #gta6") == "tr"
    assert T.guess_language("GTA 6 #gta6 #fyp") is None
    assert T.guess_language("جي تي اي 6 مذهل") == "other"


def test_extract_items_and_rank_by_language(project):
    payload = {"itemList": [{"id": vid_at(NOW - timedelta(hours=2)), "desc": "GTA 6 crash physics",
                             "author": {"uniqueId": "kai"}, "statsV2": {"playCount": "1234567"},
                             "textLanguage": "en"},
                            {"id": vid_at(NOW - timedelta(hours=3)), "desc": "GTA 6 crash physique dans la ville",
                             "author": {"uniqueId": "fr"}, "stats": {"playCount": 9_999_999}}]}
    videos = T.parse_items(T.extract_items(payload))
    assert {v.author: v.views for v in videos} == {"kai": 1_234_567, "fr": 9_999_999}
    cfg = load_config(project / "config.yaml")
    kept, dropped = T.rank(videos, now=NOW, hours=48, count=10, cfg=cfg, languages=["en"])
    assert [v.author for v, _ in kept] == ["kai"] and dropped[0][1] == "langue : fr"
    kept_all, _ = T.rank(videos, now=NOW, hours=48, count=10, cfg=cfg, languages=[])
    assert [v.author for v, _ in kept_all] == ["fr", "kai"]
