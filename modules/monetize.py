"""Module 6 — monetize (laptop) : campagnes de clipping rémunérées, classement et validation des clips.

1. **Campagnes** (``clipgod monetize``) : scraping léger de Whop Content Rewards (1 page liste + 1 page par campagne
   suivie, cache ``scrape_cache_seconds``) -> CPM, budget restant, plateformes, règles (durées, hashtags et mentions
   obligatoires, transformations). Extraction en cascade, pour survivre aux refontes du site :
       a. JSON Next.js embarqué (``__NEXT_DATA__``, puis flux ``self.__next_f``) : objets « ressemblant » à une campagne ;
       b. texte visible : « $1.50 per 1K views », « $12,000 budget », « max 30 seconds », « #gta6 », « @creator »…
   + ``data/whop_campaigns.yaml`` (saisie manuelle) qui **prime** toujours sur le scraping.
2. **Classement** par $/heure estimé : vues attendues par clip (historique ``data/historical_scores.csv`` de la
   campagne, sinon de toutes les campagnes, sinon ``prior_views_per_clip``) x CPM / 1000 x clips par heure de VOD
   (mesuré dans ``data/state.json``, sinon ``clips_per_vod_hour``), plafonné par le budget restant.
3. **Validation** (appelée par ``publish``) : règles par défaut fusionnées avec celles de la campagne du clip ;
   refus consignés dans ``data/rejected.json``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import get, path, resolve_path
from core.logger import get_logger
from core.storage import JsonStore, StateStore, read_json, utc_now_iso, write_json

log = get_logger(__name__)

# Transformations qui comptent comme un vrai travail éditorial (au moins 2 en plus du recadrage vertical).
EDITORIAL = ("word_subtitles", "hook_overlay", "facecam_split", "voiceover")
PLATFORM_WORDS = {"tiktok": "tiktok", "instagram": "instagram_reels", "reels": "instagram_reels",
                  "youtube": "youtube_shorts", "shorts": "youtube_shorts"}


# ===========================================================================
# Modèle
# ===========================================================================
@dataclass
class Rules:
    min_duration_seconds: float = 7.0
    max_duration_seconds: float = 30.0
    hook_within_seconds: float = 1.0
    require_transformation: bool = True
    min_editorial_transformations: int = 2
    require_subtitles: bool = False
    vertical_only: bool = True
    required_hashtags: list[str] = field(default_factory=list)
    required_mentions: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)          # vide = toutes
    campaign: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Rules:
        base = dict(((cfg.get("monetize") or {}).get("rules_default")) or {})
        base.update({k: v for k, v in (overrides or {}).items() if v not in (None, [], "")})
        known = {k: v for k, v in base.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Campaign:
    slug: str
    name: str = ""
    url: str | None = None
    cpm_usd: float | None = None              # $ pour 1000 vues
    budget_total_usd: float | None = None
    budget_remaining_usd: float | None = None
    platforms: list[str] = field(default_factory=list)
    streamers: list[str] = field(default_factory=list)       # créateurs dont les clips sont acceptés
    rules: dict[str, Any] = field(default_factory=dict)      # champs de Rules
    active: bool = True
    source: str = "scrape"                    # scrape | manual
    fetched_at: str | None = None

    def merge(self, other: Campaign) -> Campaign:
        """``other`` (prioritaire) complète/écrase ``self`` champ par champ."""
        data = asdict(self)
        for k, v in asdict(other).items():
            if k == "rules":
                data["rules"] = {**self.rules, **{rk: rv for rk, rv in v.items() if rv not in (None, [], "")}}
            elif v not in (None, [], "", {}):
                data[k] = v
        return Campaign(**data)


@dataclass
class Validation:
    clip_id: str
    ok: bool
    reasons: list[str] = field(default_factory=list)


# ===========================================================================
# Extraction (texte, JSON)
# ===========================================================================
def parse_money(value: Any, *, cents: bool = False) -> float | None:
    """``"$1,234.50"`` -> 1234.5 ; ``"2.5k"`` -> 2500 ; nombres en cents si ``cents``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) / (100 if cents else 1)
    m = re.search(r"(\d[\d,\s]*(?:\.\d+)?)\s*([kKmM])?", str(value))
    if not m:
        return None
    number = float(m.group(1).replace(",", "").replace(" ", ""))
    number *= {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1)
    return number / (100 if cents else 1)


_CPM_RE = re.compile(r"\$\s?(\d+(?:[.,]\d+)?)\s*(?:/|per|for every|every)\s*(?:1[,.]?000|1\s?k|thousand)\s*views?", re.IGNORECASE)
_BUDGET_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:total\s+)?(?:budget|pool)", re.IGNORECASE)
_REMAINING_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:remaining|left)", re.IGNORECASE)
_PAID_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:of\s+\$\s?(\d[\d,]*(?:\.\d+)?\s*[kKmM]?))?\s*paid", re.IGNORECASE)
_SECONDS = r"(\d+)\s*(?:s\b|sec\b|secs\b|seconds?\b)"


def extract_rules_from_text(text: str) -> dict[str, Any]:
    """Règles d'une campagne à partir de son texte (exigences, description). Heuristique, conservatrice."""
    low = text.lower()
    rules: dict[str, Any] = {}
    m = re.search(rf"between\s+(\d+)\s*(?:s|sec|seconds?)?\s*(?:and|-|to)\s*{_SECONDS}", low)
    if m:
        rules["min_duration_seconds"], rules["max_duration_seconds"] = float(m.group(1)), float(m.group(2))
    m = re.search(rf"(?:max(?:imum)?(?:\s+length|\s+duration)?|under|less than|no longer than|shorter than|up to)\s*:?\s*{_SECONDS}", low)
    if m:
        rules["max_duration_seconds"] = float(m.group(1))
    m = re.search(rf"(?:min(?:imum)?(?:\s+length|\s+duration)?|at least|longer than)\s*:?\s*{_SECONDS}", low)
    if m:
        rules["min_duration_seconds"] = float(m.group(1))
    if re.search(r"captions?|subtitles?", low) and re.search(r"must|required|need|include", low):
        rules["require_subtitles"] = True
    if re.search(r"no (?:raw )?re-?uploads?|must be edited|transformative|original edit", low):
        rules["require_transformation"] = True
    hashtags = list(dict.fromkeys(h.lower() for h in re.findall(r"(?<![\w&])#(\w{2,40})", text)))
    if hashtags:
        rules["required_hashtags"] = [f"#{h}" for h in hashtags[:8]]
    mentions = [m for m in re.findall(r"(?<![\w.])@([A-Za-z0-9_.]{2,30})", text) if not m.endswith(".")]
    if mentions:
        rules["required_mentions"] = [f"@{m}" for m in dict.fromkeys(mentions)][:5]
    return rules


def _platforms_from(values: Iterable[Any]) -> list[str]:
    found = []
    for v in values:
        for word, platform in PLATFORM_WORDS.items():
            if word in str(v).lower() and platform not in found:
                found.append(platform)
    return found


def campaign_from_text(slug: str, text: str, *, url: str | None = None, name: str = "") -> Campaign:
    m = _CPM_RE.search(text)
    budget = _BUDGET_RE.search(text)
    remaining = _REMAINING_RE.search(text)
    paid = _PAID_RE.search(text)
    total = parse_money(budget.group(1)) if budget else None
    rem = parse_money(remaining.group(1)) if remaining else None
    if rem is None and paid and paid.group(2):
        total = total or parse_money(paid.group(2))
        rem = (total or 0) - (parse_money(paid.group(1)) or 0)
    platforms = _platforms_from(re.findall(r"tiktok|instagram|reels|youtube|shorts", text, re.IGNORECASE))
    return Campaign(slug=slug, name=name or slug, url=url,
                    cpm_usd=parse_money(m.group(1).replace(",", ".")) if m else None,
                    budget_total_usd=total, budget_remaining_usd=rem, platforms=platforms,
                    rules=extract_rules_from_text(text), fetched_at=utc_now_iso())


_KEY_CPM = ("cpm", "ratepermille", "rateperthousandviews", "rateper1000views", "rewardrate", "payoutper1kviews",
            "ratepermilleviews", "rateper1kviews", "payoutrate")
_KEY_TOTAL = ("budget", "totalbudget", "budgettotal", "maxbudget", "totalpool")
_KEY_REMAINING = ("remainingbudget", "budgetremaining", "remaining", "amountremaining")
_KEY_PAID = ("paidout", "amountpaid", "totalpaid", "spent")


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _money_field(obj: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for k, v in obj.items():
        nk = _norm(k)
        base = nk.removesuffix("cents").removesuffix("usd").removesuffix("amount")
        if base in keys or nk in keys:
            if isinstance(v, Mapping):            # {"amount": 150, "currency": "usd"}
                v = v.get("amount") or v.get("value")
            return parse_money(v, cents=nk.endswith("cents"))
    return None


def campaigns_from_json(data: Any) -> list[Campaign]:
    """Objets JSON qui ressemblent à une campagne : un nom + un CPM ou un budget."""
    out: dict[str, Campaign] = {}
    for obj in _walk(data):
        cpm = _money_field(obj, _KEY_CPM)
        total = _money_field(obj, _KEY_TOTAL)
        if cpm is None and total is None:
            continue
        name = obj.get("title") or obj.get("name") or obj.get("campaignName")
        slug = obj.get("slug") or obj.get("route") or obj.get("id")
        if not (name or slug):
            continue
        remaining = _money_field(obj, _KEY_REMAINING)
        paid = _money_field(obj, _KEY_PAID)
        if remaining is None and total is not None and paid is not None:
            remaining = total - paid
        text = " ".join(str(obj.get(k, "")) for k in ("description", "requirements", "rules", "instructions"))
        platforms = obj.get("platforms") or obj.get("allowedPlatforms") or []
        c = Campaign(slug=str(slug or re.sub(r"\W+", "-", str(name).lower())), name=str(name or slug),
                     url=obj.get("url"), cpm_usd=cpm, budget_total_usd=total, budget_remaining_usd=remaining,
                     platforms=_platforms_from(platforms if isinstance(platforms, list) else [platforms]),
                     rules=extract_rules_from_text(text) if text.strip() else {},
                     active=str(obj.get("status", "active")).lower() not in ("ended", "paused", "closed", "inactive"),
                     fetched_at=utc_now_iso())
        out[c.slug] = out[c.slug].merge(c) if c.slug in out else c
    return list(out.values())


def embedded_json(html: str) -> list[Any]:
    """JSON embarqué par Next.js : ``__NEXT_DATA__`` et fragments du flux ``self.__next_f.push``."""
    blobs: list[Any] = []
    m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if m:
        try:
            blobs.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    flight = "".join(json.loads(f'"{chunk}"') for chunk in
                     re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html))
    for candidate in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", flight):
        if re.search(r"cpm|budget|rate", candidate, re.IGNORECASE):
            try:
                blobs.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
    return blobs


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, ch in (("&amp;", "&"), ("&nbsp;", " "), ("&#x27;", "'"), ("&quot;", '"'), ("&#36;", "$")):
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text)


def parse_page(html: str, *, slug: str | None = None, url: str | None = None) -> list[Campaign]:
    """Page liste (plusieurs campagnes) ou page campagne (``slug`` donné)."""
    found = []
    for blob in embedded_json(html):
        found += campaigns_from_json(blob)
    if slug:
        text_campaign = campaign_from_text(slug, html_to_text(html), url=url)
        match = next((c for c in found if c.slug == slug or slug in (c.url or "")), None)
        return [match.merge(text_campaign) if match else text_campaign]
    return found


# ===========================================================================
# Récupération (scraping léger + saisie manuelle + cache)
# ===========================================================================
def _whop_targets(cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``whop:<slug|url>`` de campaigns.txt -> [(slug, url)]."""
    template = str(get(cfg, "monetize.campaign_url_template", "https://whop.com/{slug}/"))
    file = path(cfg, "campaigns_txt")
    out = []
    if file.is_file():
        for raw in file.read_text(encoding="utf-8").splitlines():
            line = re.split(r"\s#", raw.strip(), maxsplit=1)[0].strip()   # commentaire de fin de ligne
            if not line.lower().startswith("whop:"):
                continue
            target = line[5:].split()[0]
            if target.startswith("http"):
                out.append((target.rstrip("/").split("/")[-1], target))
            else:
                out.append((target, template.format(slug=target)))
    return out


def load_manual(cfg: Mapping[str, Any]) -> list[Campaign]:
    file = resolve_path(cfg, get(cfg, "monetize.manual_file", "data/whop_campaigns.yaml"))
    if not file.is_file():
        return []
    import yaml

    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
    out = []
    for item in raw if isinstance(raw, list) else raw.get("campaigns", []):
        if not isinstance(item, dict) or not item.get("slug"):
            continue
        rules = {k: item[k] for k in Rules.__dataclass_fields__ if k in item}
        rules.update(item.get("rules") or {})
        out.append(Campaign(slug=str(item["slug"]), name=str(item.get("name", item["slug"])), url=item.get("url"),
                            cpm_usd=parse_money(item.get("cpm_usd")), budget_total_usd=parse_money(item.get("budget_total_usd")),
                            budget_remaining_usd=parse_money(item.get("budget_remaining_usd")),
                            platforms=list(item.get("platforms") or []),
                            streamers=[str(s).lstrip("@").lower() for s in item.get("streamers") or []],
                            rules=rules, active=bool(item.get("active", True)), source="manual",
                            fetched_at=utc_now_iso()))
    return out


def scrape_whop(cfg: Mapping[str, Any], *, client: Any = None) -> list[Campaign]:
    mcfg = cfg.get("monetize") or {}
    own = client is None
    if own:
        import httpx

        client = httpx.Client(timeout=20, follow_redirects=True,
                              headers={"User-Agent": str(mcfg.get("user_agent", "Mozilla/5.0 (compatible; ClipGod/0.1)")),
                                       "Accept": "text/html,application/xhtml+xml"})
    found: dict[str, Campaign] = {}
    try:
        listing = str(mcfg.get("whop_rewards_url", "https://whop.com/discover/content-rewards/"))
        try:
            resp = client.get(listing)
            resp.raise_for_status()
            for c in parse_page(resp.text):
                found[c.slug] = c
        except Exception as exc:  # noqa: BLE001
            log.warning("Liste Whop illisible (%s) : seules les campagnes suivies/manuelles seront utilisées.", exc)
        for slug, url in _whop_targets(cfg):
            time.sleep(float(mcfg.get("delay_seconds", 1.0)))     # scraping léger : une page à la fois
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                log.warning("Campagne Whop %s illisible : %s", slug, exc)
                continue
            for c in parse_page(resp.text, slug=slug, url=url):
                found[c.slug] = found[c.slug].merge(c) if c.slug in found else c
    finally:
        if own:
            client.close()
    return list(found.values())


def fetch_campaigns(cfg: Mapping[str, Any], *, refresh: bool = False, client: Any = None,
                    allow_scrape: bool = True) -> list[Campaign]:
    """Campagnes connues : cache (``data/campaigns.json``) ou scraping, puis saisie manuelle par-dessus.

    ``allow_scrape=False`` (utilisé par publish) : cache même expiré + manuel, aucun appel réseau.
    """
    cache = resolve_path(cfg, get(cfg, "monetize.cache_file", "data/campaigns.json"))
    ttl = float(get(cfg, "monetize.scrape_cache_seconds", 1800))
    cached = read_json(cache, default=None)
    fresh = isinstance(cached, dict) and time.time() - float(cached.get("ts", 0)) < ttl
    if not allow_scrape or (not refresh and fresh):
        scraped = [Campaign(**c) for c in (cached or {}).get("campaigns", [])] if isinstance(cached, dict) else []
    else:
        scraped = scrape_whop(cfg, client=client)
        write_json(cache, {"ts": time.time(), "fetched_at": utc_now_iso(), "campaigns": [asdict(c) for c in scraped]})
    merged = {c.slug: c for c in scraped}
    for manual in load_manual(cfg):
        merged[manual.slug] = merged[manual.slug].merge(manual) if manual.slug in merged else manual
    return list(merged.values())


# ===========================================================================
# Classement $/heure
# ===========================================================================
def expected_views(cfg: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    """(médiane des vues par campagne, médiane globale) depuis historical_scores.csv (vues à 72h, sinon 24h)."""
    prior = float(get(cfg, "monetize.prior_views_per_clip", 2000))
    file = path(cfg, "historical_scores")
    if not file.is_file():
        return {}, prior
    import pandas as pd

    try:
        df = pd.read_csv(file)
    except Exception:  # noqa: BLE001 — CSV vide ou corrompu
        return {}, prior
    col = next((c for c in ("views_72h", "views_24h") if c in df and df[c].notna().any()), None)
    if col is None or df.empty:
        return {}, prior
    df = df[df[col].notna()]
    by_campaign = df.groupby("campaign")[col].median().to_dict() if "campaign" in df else {}
    return {str(k): float(v) for k, v in by_campaign.items() if isinstance(k, str)}, float(df[col].median())


def clips_per_vod_hour(cfg: Mapping[str, Any]) -> float:
    """Clips validés obtenus par heure de VOD traitée (historique), sinon valeur de config."""
    default = float(get(cfg, "monetize.clips_per_vod_hour", 3))
    state = StateStore(path(cfg, "state")).load()
    jobs = [j for j in state.get("jobs", {}).values() if j.get("status") == "synced" and j.get("source_duration")]
    hours = sum(float(j["source_duration"]) for j in jobs) / 3600
    clips = sum(int(j.get("clips", 0)) for j in jobs)
    return clips / hours if hours >= 2 and clips else default


def rank_campaigns(cfg: Mapping[str, Any], campaigns: Sequence[Campaign]) -> list[dict[str, Any]]:
    mcfg = cfg.get("monetize") or {}
    per_campaign, global_views = expected_views(cfg)
    cph = clips_per_vod_hour(cfg)
    rows = []
    for c in campaigns:
        reasons = []
        if not c.active:
            reasons.append("inactive")
        if c.cpm_usd is None:
            reasons.append("CPM inconnu")
        elif c.cpm_usd < float(mcfg.get("min_cpm", 0.5)):
            reasons.append(f"CPM < {mcfg.get('min_cpm')}")
        if c.budget_remaining_usd is not None and c.budget_remaining_usd < float(mcfg.get("min_budget_remaining_usd", 100)):
            reasons.append("budget presque épuisé")
        views = per_campaign.get(c.slug, global_views)
        per_clip = (views * (c.cpm_usd or 0) / 1000) if c.cpm_usd else 0.0
        per_hour = per_clip * cph
        if c.budget_remaining_usd is not None:
            per_hour = min(per_hour, c.budget_remaining_usd)
        rows.append({"slug": c.slug, "name": c.name, "url": c.url, "cpm_usd": c.cpm_usd,
                     "budget_remaining_usd": c.budget_remaining_usd, "platforms": c.platforms,
                     "expected_views": round(views), "views_source": "campagne" if c.slug in per_campaign else "global",
                     "usd_per_clip": round(per_clip, 2), "usd_per_hour": round(per_hour, 2),
                     "eligible": not reasons, "excluded_because": reasons, "rules": rules_summary(c), "source": c.source})
    return sorted(rows, key=lambda r: (not r["eligible"], -r["usd_per_hour"]))


def rules_summary(c: Campaign) -> str:
    r = c.rules
    parts = []
    if "min_duration_seconds" in r or "max_duration_seconds" in r:
        parts.append(f"{r.get('min_duration_seconds', '?')}-{r.get('max_duration_seconds', '?')}s")
    if r.get("required_hashtags"):
        parts.append(" ".join(r["required_hashtags"]))
    if r.get("required_mentions"):
        parts.append(" ".join(r["required_mentions"]))
    if r.get("require_subtitles"):
        parts.append("sous-titres")
    return " · ".join(parts)


# ===========================================================================
# Validation
# ===========================================================================
def rules_for(cfg: Mapping[str, Any], campaign: Campaign | None) -> Rules:
    if campaign is None:
        return Rules.from_config(cfg)
    rules = Rules.from_config(cfg, {**campaign.rules, "platforms": campaign.platforms or campaign.rules.get("platforms")})
    rules.campaign = campaign.slug
    return rules


def match_campaign(campaigns: Sequence[Campaign], *, slug: str | None = None,
                   streamer: str | None = None) -> Campaign | None:
    """Campagne d'un clip : celle choisie au prepare (``--campaign``), sinon celle qui cible son streamer."""
    by_slug = {c.slug: c for c in campaigns}
    if slug and slug in by_slug:
        return by_slug[slug]
    who = (streamer or "").lstrip("@").lower()
    if who:
        for c in campaigns:
            if c.active and who in c.streamers:
                return c
    return None


def validate_clip(meta: Mapping[str, Any], rules: Rules, clip_path: str | Path | None = None) -> Validation:
    """Vérifie un clip (métadonnées du sidecar render) contre les règles. Ne lit pas la vidéo elle-même."""
    clip_id = str(meta.get("clip_id") or (Path(clip_path).stem if clip_path else "?"))
    reasons: list[str] = []
    if clip_path is not None:
        p = Path(clip_path)
        if not p.is_file() or p.stat().st_size == 0:
            reasons.append("fichier vidéo absent ou vide")
    duration = float(meta.get("duration") or 0)
    if duration < rules.min_duration_seconds:
        reasons.append(f"trop court ({duration:.1f}s < {rules.min_duration_seconds:g}s)")
    if duration >= rules.max_duration_seconds:
        reasons.append(f"trop long ({duration:.1f}s, max strictement < {rules.max_duration_seconds:g}s)")
    hook_at = meta.get("hook_at_seconds")
    if not meta.get("hook"):
        reasons.append("aucun hook affiché")
    elif hook_at is None or float(hook_at) > rules.hook_within_seconds:
        reasons.append(f"hook après {rules.hook_within_seconds:g}s")
    transformations = set(meta.get("transformations") or [])
    if rules.require_transformation:
        editorial = transformations & set(EDITORIAL)
        if "vertical_reframe" not in transformations or len(editorial) < rules.min_editorial_transformations:
            reasons.append(
                f"transformation insuffisante (simple re-upload ?) : {sorted(transformations) or 'aucune'} — "
                f"il faut le recadrage vertical + {rules.min_editorial_transformations} parmi {', '.join(EDITORIAL)}"
            )
    if rules.require_subtitles and "word_subtitles" not in transformations:
        reasons.append("sous-titres exigés par la campagne")
    if rules.vertical_only:
        w, h = int(meta.get("width") or 0), int(meta.get("height") or 0)
        if not (w and h and h > w):
            reasons.append(f"format non vertical ({w}x{h})")
    return Validation(clip_id, not reasons, reasons)


def record_rejection(cfg: Mapping[str, Any], validation: Validation, *, campaign: str | None = None) -> None:
    """Ajoute (ou met à jour) le clip dans ``data/rejected.json``."""
    store = JsonStore(path(cfg, "rejected"), lambda: {"rejected": {}})
    with store.transaction() as data:
        data["rejected"][validation.clip_id] = {
            "reasons": validation.reasons, "campaign": campaign, "rejected_at": utc_now_iso(),
        }
    log.warning("Clip %s refusé : %s", validation.clip_id, " ; ".join(validation.reasons))


def estimate_payout(views: float, cpm_usd: float | None) -> float:
    return 0.0 if not cpm_usd or not math.isfinite(views) else views * cpm_usd / 1000
