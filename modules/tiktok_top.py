"""Veille TikTok (laptop) : les vidéos GTA 6 gameplay les plus vues des dernières heures -> fichier .txt.

TikTok n'a pas d'API gratuite de recherche et bloque yt-dlp sur les pages de hashtags. ``clipgod toptiktok`` ouvre
donc les pages publiques ``tiktok.com/tag/<hashtag>`` dans **le Chrome (ou Edge) déjà installé**, via Playwright,
les fait défiler comme un visiteur, puis lit pour chaque vidéo : le lien, les vues affichées et la description.

    - date de publication : déduite de l'identifiant de la vidéo (ses 32 premiers bits = horodatage Unix) ;
    - filtre « gameplay » : le même que ``clipgod topshorts`` (titre / hashtags / description) ;
    - profil de navigateur dédié (``cache/tiktok_profile``) : si TikTok affiche une vérification (captcha), on la
      résout une fois dans la fenêtre, les cookies sont gardés pour les fois suivantes.

Lecture légère : quelques pages, défilement espacé ; rien n'est téléchargé ni republié.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import get, resolve_path
from core.logger import get_logger

log = get_logger(__name__)

VIDEO_RE = re.compile(r"/@([^/?#]+)/video/(\d{15,22})")
COUNT_RE = re.compile(r"^\s*([\d.,\s]+)\s*([kmb]|mille|k|m|md|mrd)?\s*$", re.IGNORECASE)

# Lien, vues affichées et description de chaque vignette vidéo de la page.
COLLECT_JS = """
() => [...document.querySelectorAll('a[href*="/video/"]')].map(a => {
  const box = a.closest('[data-e2e="challenge-item"], [data-e2e="search_top-item"], div') || a;
  const views = a.querySelector('[data-e2e="video-views"]') || box.querySelector('[data-e2e="video-views"]');
  const img = a.querySelector('img') || box.querySelector('img');
  const desc = box.querySelector('[data-e2e="challenge-item-desc"], [data-e2e="search-card-desc"]');
  return {href: a.href, views: views ? views.textContent : '',
          text: (desc ? desc.textContent : '') || (img ? img.alt : '') || ''};
})
"""


class TikTokError(RuntimeError):
    """Playwright absent, aucun navigateur, page jamais chargée…"""


@dataclass
class TikTokVideo:
    id: str
    author: str
    url: str
    views: int
    text: str
    published_at: datetime
    lang: str | None = None              # langue détectée par TikTok (textLanguage), sinon None


# ---------------------------------------------------------------------------
# Données structurées de TikTok (ce que la page télécharge pour s'afficher)
# ---------------------------------------------------------------------------
def item_to_raw(it: Mapping[str, Any]) -> dict[str, Any] | None:
    """Vidéo au format de l'API web TikTok (``itemList`` / ``itemStruct``) -> entrée brute."""
    vid = str(it.get("id") or "")
    author = it.get("author")
    author = author.get("uniqueId") if isinstance(author, Mapping) else author
    if not vid.isdigit() or not author:
        return None
    stats = it.get("statsV2") or it.get("stats") or {}
    views = stats.get("playCount") if isinstance(stats, Mapping) else None
    return {"href": f"https://www.tiktok.com/@{author}/video/{vid}", "views": views if views is not None else "",
            "text": it.get("desc") or "", "lang": it.get("textLanguage") or None, "source": "api"}


def extract_items(obj: Any, out: list[dict[str, Any]] | None = None, depth: int = 0) -> list[dict[str, Any]]:
    """Cherche récursivement les vidéos dans une réponse JSON de TikTok (listes ``itemList``, ``itemStruct``…)."""
    out = [] if out is None else out
    if depth > 12:
        return out
    if isinstance(obj, Mapping):
        if "id" in obj and ("stats" in obj or "statsV2" in obj) and "author" in obj:
            raw = item_to_raw(obj)
            if raw:
                out.append(raw)
            return out
        for v in obj.values():
            extract_items(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            extract_items(v, out, depth + 1)
    return out


# Langue d'une description quand TikTok ne la donne pas : mots-outils les plus fréquents par langue.
STOPWORDS = {
    "en": {"the", "a", "an", "and", "you", "your", "is", "in", "of", "to", "this", "that", "how", "what", "with",
           "my", "it", "on", "for", "when", "me", "he", "she", "they", "can", "be", "was", "are", "do", "don't",
           "best", "never", "find", "where", "get", "insane", "most", "new", "why", "who"},
    "fr": {"le", "la", "les", "un", "une", "des", "et", "est", "dans", "pour", "avec", "sur", "en", "du", "de",
           "ou", "trouver", "avoir", "quelques", "voix", "mon", "ma", "pas", "qui", "que", "c'est"},
    "es": {"el", "los", "las", "y", "es", "en", "con", "para", "por", "una", "del", "que", "cómo", "dónde"},
    "it": {"il", "lo", "gli", "e", "è", "di", "con", "per", "una", "dei", "che", "contanti", "leggendario"},
    "pt": {"o", "os", "as", "e", "é", "em", "com", "para", "uma", "do", "da", "que", "não", "você"},
    "de": {"der", "die", "das", "und", "ist", "mit", "für", "ein", "eine", "nicht", "wie", "auf"},
    "tr": {"ve", "bir", "bu", "için", "ile", "deyince", "de", "da", "çok", "ne", "yanlış", "karar"},
}


def guess_language(text: str) -> str | None:
    """Langue probable d'une description (``en``, ``fr``…), ``other`` hors alphabet latin, None si indécidable."""
    from core.scoring import is_latin_text

    clean = re.sub(r"(?:[#@]\S+|https?://\S+|created by .*$)", " ", text, flags=re.IGNORECASE)
    if not is_latin_text(clean, threshold=0.6):
        return "other"
    words = re.findall(r"[a-zà-öø-ÿçğışü']+", clean.lower())
    scores = {lang: sum(w in sw for w in words) for lang, sw in STOPWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    top = scores[best]
    if top == 0 or sum(1 for v in scores.values() if v == top) > 1:
        return None
    return best


def parse_count(text: str | None) -> int:
    """« 1.2M » / « 1,2 M » / « 845.3K » / « 12 345 » -> entier (0 si illisible)."""
    if not text:
        return 0
    m = COUNT_RE.match(str(text).strip().lower().replace(" ", " ").replace("\xa0", " "))
    if not m:
        return 0
    number = m.group(1).replace(" ", "")
    unit = (m.group(2) or "").lower()
    if unit:
        number = number.replace(",", ".")
    else:
        number = number.replace(",", "").replace(".", "")
    try:
        value = float(number)
    except ValueError:
        return 0
    factor = {"k": 1e3, "mille": 1e3, "m": 1e6, "b": 1e9, "md": 1e9, "mrd": 1e9}.get(unit, 1)
    return round(value * factor)


def id_timestamp(video_id: str) -> datetime:
    """Les 32 premiers bits d'un identifiant TikTok sont l'heure de publication (secondes Unix)."""
    return datetime.fromtimestamp(int(video_id) >> 32, timezone.utc)


def parse_items(raw: Iterable[Mapping[str, Any]]) -> list[TikTokVideo]:
    """Résultats bruts (API + page) -> vidéos uniques : vues, langue et description les plus complètes gardées."""
    best: dict[str, TikTokVideo] = {}
    for item in raw:
        m = VIDEO_RE.search(str(item.get("href") or ""))
        if not m:
            continue
        author, vid = m.group(1), m.group(2)
        views = item.get("views")
        count = int(views) if isinstance(views, (int, float)) or str(views).isdigit() else parse_count(views)
        text = re.sub(r"\s+created by .*$", "", " ".join(str(item.get("text") or "").split()), flags=re.IGNORECASE)
        video = TikTokVideo(vid, author, f"https://www.tiktok.com/@{author}/video/{vid}", count, text,
                            id_timestamp(vid), item.get("lang") or None)
        cur = best.get(vid)
        if cur is None:
            best[vid] = video
            continue
        cur.views = max(cur.views, video.views)
        cur.lang = cur.lang or video.lang
        if len(video.text) > len(cur.text):
            cur.text = video.text
    return list(best.values())


def rank(videos: Sequence[TikTokVideo], *, now: datetime, hours: float, count: int, gameplay_only: bool = True,
         cfg: Mapping[str, Any] | None = None, languages: Sequence[str] | None = ("en",)
         ) -> tuple[list[tuple[TikTokVideo, str]], list[tuple[TikTokVideo, str]]]:
    """(retenues, écartées) : publiées depuis ``hours`` h, langue (TikTok, sinon devinée), gameplay ; par vues."""
    from modules.viral import Short, gameplay_verdict

    since = now - timedelta(hours=hours)
    wanted = {x.lower() for x in languages or []}
    kept: list[tuple[TikTokVideo, str]] = []
    dropped: list[tuple[TikTokVideo, str]] = []
    for v in sorted(videos, key=lambda v: -v.views):
        if v.published_at < since:
            continue
        if wanted:
            lang = (v.lang or guess_language(v.text) or "").split("-")[0].lower()
            if lang not in wanted:
                dropped.append((v, f"langue : {lang or 'inconnue'}"))
                continue
        if not gameplay_only:
            kept.append((v, ""))
            continue
        short = Short(v.id, v.text, v.author, "", v.views, 0, v.published_at,
                      tags=[t.lstrip("#") for t in v.text.split() if t.startswith("#")])
        ok, why = gameplay_verdict(short, cfg)
        (kept if ok else dropped).append((v, why))
    return kept[:count], dropped


def write_report(path_: str | Path, kept: Sequence[tuple[TikTokVideo, str]],
                 dropped: Sequence[tuple[TikTokVideo, str]], *, hours: float, seen: int, gameplay_only: bool) -> Path:
    def fmt(n: int) -> str:
        return f"{n:,}".replace(",", " ")

    kind = " gameplay" if gameplay_only else ""
    lines = [f"Top {len(kept)} TikTok GTA 6{kind} les plus vus ({hours:g} h) — {seen} vidéos examinées\n"]
    for i, (v, why) in enumerate(kept, 1):
        extra = f" - {why}" if why else ""
        lines.append(f"{i}. {v.text[:100] or '(sans description)'}\n   @{v.author} - {fmt(v.views)} vues - "
                     f"publié le {v.published_at:%d/%m à %H:%M} (UTC){extra}\n   {v.url}\n")
    if gameplay_only and dropped:
        lines.append(f"\n--- Écartées ({len(dropped)}) : vérifie qu'aucun gameplay n'a été oublié ---\n")
        lines += [f"- {v.text[:80] or '(sans description)'} (@{v.author}, {fmt(v.views)} vues) : {why}\n   {v.url}"
                  for v, why in dropped]
    out = Path(path_)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------------------
# Navigateur
# ---------------------------------------------------------------------------
def collect(tags: Sequence[str], *, profile_dir: Path, scrolls: int = 8, headless: bool = False,
            wait_seconds: float = 90, say: Callable[[str], None] = print, base_url: str = "https://www.tiktok.com",
            executable_path: str | None = None, scroll_pause_ms: int = 1800) -> list[dict[str, Any]]:
    """Ouvre chaque page de hashtag dans Chrome/Edge, fait défiler et renvoie les vignettes lues."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise TikTokError("Playwright n'est pas installé : pip install playwright") from exc

    profile_dir.mkdir(parents=True, exist_ok=True)
    raw: list[dict[str, Any]] = []
    with sync_playwright() as p:
        ctx = None
        options: list[dict[str, Any]] = ([{"executable_path": executable_path}] if executable_path else
                                         [{"channel": "chrome"}, {"channel": "msedge"}, {}])
        for opt in options:
            try:
                ctx = p.chromium.launch_persistent_context(
                    str(profile_dir), headless=headless, locale="en-US", viewport={"width": 1280, "height": 900},
                    **opt)
                break
            except PlaywrightError:
                continue
        if ctx is None:
            raise TikTokError("Aucun navigateur utilisable : installez Google Chrome ou Microsoft Edge.")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        responses: list[Any] = []
        # On garde seulement les réponses ; leur contenu est lu après le défilement (jamais dans le gestionnaire).
        page.on("response", lambda r: responses.append(r) if "item_list" in r.url or "/api/challenge" in r.url
                else None)
        try:
            for tag in tags:
                responses.clear()
                say(f"#{tag} : ouverture de la page…")
                page.goto(f"{base_url}/tag/{tag}", wait_until="domcontentloaded", timeout=60_000)
                try:
                    page.wait_for_selector('a[href*="/video/"]', timeout=wait_seconds * 1000)
                except PlaywrightError:
                    say(f"#{tag} : aucune vidéo affichée (vérification TikTok non résolue ?) — hashtag ignoré.")
                    continue
                for _ in range(scrolls):
                    page.mouse.wheel(0, 3500)
                    page.wait_for_timeout(scroll_pause_ms)
                api: list[dict[str, Any]] = []
                for r in responses:
                    try:
                        extract_items(r.json(), api)
                    except Exception as exc:  # noqa: BLE001 — réponse non JSON ou déjà libérée
                        log.debug("réponse TikTok ignorée (%s) : %s", r.url[:80], exc)
                api += _page_state_items(page)
                found = page.evaluate(COLLECT_JS)
                with_views = sum(1 for it in api if str(it.get("views")) not in ("", "None"))
                say(f"#{tag} : {len(found)} vignette(s), {len(api)} fiche(s) détaillée(s) dont {with_views} avec vues")
                raw.extend(api)
                raw.extend(found)
        finally:
            ctx.close()
    return raw


STATE_JS = """
() => [...document.querySelectorAll('script#__UNIVERSAL_DATA_FOR_REHYDRATION__, script#SIGI_STATE')]
        .map(s => s.textContent)
"""


def _page_state_items(page: Any) -> list[dict[str, Any]]:
    """Vidéos présentes dans les données embarquées de la page (premier lot, page vidéo…)."""
    import json

    out: list[dict[str, Any]] = []
    try:
        for text in page.evaluate(STATE_JS) or []:
            try:
                extract_items(json.loads(text), out)
            except ValueError:
                continue
    except Exception as exc:  # noqa: BLE001
        log.debug("données embarquées illisibles : %s", exc)
    return out


def fetch_details(urls: Sequence[str], *, profile_dir: Path, headless: bool = False, say: Callable[[str], None] = print,
                  executable_path: str | None = None, pause_ms: int = 1200) -> list[dict[str, Any]]:
    """Ouvre chaque page vidéo pour lire vues et langue dans ses données embarquées (quand la liste ne les donne pas)."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise TikTokError("Playwright n'est pas installé : pip install playwright") from exc
    raw: list[dict[str, Any]] = []
    with sync_playwright() as p:
        ctx = None
        options: list[dict[str, Any]] = ([{"executable_path": executable_path}] if executable_path else
                                         [{"channel": "chrome"}, {"channel": "msedge"}, {}])
        for opt in options:
            try:
                ctx = p.chromium.launch_persistent_context(str(profile_dir), headless=headless, locale="en-US",
                                                           viewport={"width": 1280, "height": 900}, **opt)
                break
            except PlaywrightError:
                continue
        if ctx is None:
            raise TikTokError("Aucun navigateur utilisable : installez Google Chrome ou Microsoft Edge.")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for i, url in enumerate(urls, 1):
                if i == 1 or i % 10 == 0:
                    say(f"lecture des vues : {i}/{len(urls)}")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                    page.wait_for_timeout(pause_ms)
                except PlaywrightError:
                    continue
                raw += _page_state_items(page)
        finally:
            ctx.close()
    return raw


def run_toptiktok(cfg: Mapping[str, Any], *, hours: float = 48, count: int = 10, tags: Sequence[str] | None = None,
                  scrolls: int | None = None, headless: bool = False, gameplay_only: bool = True,
                  output: str | Path = "top10_tiktok_gta6.txt", now: datetime | None = None,
                  languages: Sequence[str] | None = None,
                  collect_fn: Callable[..., list[dict[str, Any]]] = collect,
                  details_fn: Callable[..., list[dict[str, Any]]] = fetch_details,
                  say: Callable[[str], None] = print) -> tuple[Path, list[tuple[TikTokVideo, str]]]:
    tcfg = get(cfg, "tiktok_top", {}) or {}
    tags = list(tags or tcfg.get("tags") or ["gta6", "gtavi", "gta6gameplay"])
    languages = list(languages if languages is not None else (tcfg.get("languages") or ["en"]))
    profile = resolve_path(cfg, get(cfg, "paths.cache", "cache")) / "tiktok_profile"
    now = now or datetime.now(timezone.utc)
    raw = collect_fn(tags, profile_dir=profile, scrolls=int(scrolls or tcfg.get("scrolls", 8)), headless=headless,
                     say=say)
    videos = parse_items(raw)
    if not videos:
        raise TikTokError("Aucune vidéo lue sur TikTok. Si une vérification s'est affichée, relancez et résolvez-la "
                          "dans la fenêtre du navigateur (elle ne sera plus demandée ensuite).")
    # Vues / langue manquantes : on ouvre la page des vidéos récentes concernées (au plus detail_limit).
    since = now - timedelta(hours=hours)
    missing = [v for v in videos if v.published_at >= since and (not v.views or not v.lang)]
    limit = int(tcfg.get("detail_limit", 60))
    if missing and limit > 0:
        say(f"{len(missing)} vidéo(s) récente(s) sans vues ou sans langue : lecture de leur page "
            f"({min(len(missing), limit)} max, ~2 s chacune)")
        missing.sort(key=lambda v: -v.published_at.timestamp())
        details = details_fn([v.url for v in missing[:limit]], profile_dir=profile, headless=headless, say=say)
        videos = parse_items([*raw, *details])
    if sum(1 for v in videos if v.published_at >= since and v.views) < 3:
        say("⚠ Vues illisibles sur la plupart des vidéos : le classement peut être approximatif.")
    kept, dropped = rank(videos, now=now, hours=hours, count=count, gameplay_only=gameplay_only, cfg=cfg,
                         languages=languages)
    out = write_report(output, kept, dropped, hours=hours, seen=len(videos), gameplay_only=gameplay_only)
    return out, kept
