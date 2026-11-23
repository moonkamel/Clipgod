"""Module 1 — discover (laptop) : trouve les meilleures sources GTA 6 à clipper -> ``data/sources.json``.

Plateformes (APIs gratuites uniquement) :
    - YouTube Data API v3 (clé ``YOUTUBE_API_KEY``) — économe en quota : 2 ``search.list`` par run (live + VODs,
      100 u chacun), puis ``videos.list`` / ``channels.list`` / ``playlistItems.list`` à 1 u. Le quota est partagé
      avec ``publish`` via :mod:`core.quota` (budget discover plafonné par ``discover.youtube.daily_budget``).
    - Twitch Helix (``TWITCH_CLIENT_ID`` / ``TWITCH_CLIENT_SECRET``, token d'application client-credentials) :
      lives de la catégorie GTA VI + top VODs du jour de la catégorie + chaînes de ``data/campaigns.txt``.
    - Kick : lecture légère (httpx, sans navigateur) des endpoints JSON publics du site pour les chaînes listées.
      Non officiels : ils peuvent changer ou être bloqués par Cloudflare -> la plateforme est alors ignorée.

Filtres : streams **live** depuis >= ``min_duration_minutes``, ou **VODs** de moins de ``max_vod_age_hours`` et de
plus de ``min_duration_minutes`` ; titre/catégorie GTA 6 ; ``min_viewers``.
Classement : chaînes de ``campaigns.txt`` d'abord, puis ratio **vues/heure** (VOD : vues / heures depuis la
publication ; live : spectateurs simultanés).
Cache local de ``cache_ttl_seconds`` (5 min) par plateforme.

Chaque source : ``{url, title, viewers, duration, streamer, platform, status, published_at, views_per_hour,
campaign, id}`` (``duration`` en secondes ; ``status`` = ``live`` | ``vod``).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import ConfigError, get, get_env, path, resolve_path
from core.logger import get_logger
from core.quota import QuotaExceeded, YouTubeQuota
from core.storage import read_json, utc_now, utc_now_iso, write_json

log = get_logger(__name__)

SOURCES_FORMAT = "clipgod-sources/1"
TWITCH_API = "https://api.twitch.tv/helix"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
KICK_API = "https://kick.com/api/v2"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


class DiscoverError(RuntimeError):
    """Aucune plateforme exploitable (clés manquantes, réseau…)."""


@dataclass
class SourceItem:
    url: str
    title: str
    viewers: int
    duration: float                  # secondes (live : durée écoulée)
    streamer: str
    platform: str                    # youtube | twitch | kick
    status: str                      # live | vod
    published_at: str | None = None
    views_per_hour: float = 0.0
    campaign: str | None = None
    id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def parse_iso8601_duration(value: str | None) -> float:
    """``PT3H2M1S`` -> 10921.0 (durées YouTube)."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value or "")
    if not m:
        return 0.0
    d, h, mi, s = (float(x) if x else 0.0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def parse_twitch_duration(value: str | None) -> float:
    """``3h2m10s`` -> 10930.0 (durées Twitch)."""
    total = 0.0
    for num, unit in re.findall(r"(\d+)([hms])", value or ""):
        total += int(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def title_matches(title: str, keywords: Sequence[str]) -> bool:
    """Titre pertinent : contient un mot-clé entier (« GTA VI » ne correspond pas à « GTA V »)."""
    low = title.lower()
    return any(re.search(rf"(?<!\w){re.escape(k.lower())}(?!\w)", low) for k in keywords)


@dataclass
class CampaignTargets:
    youtube: dict[str, str] = field(default_factory=dict)   # identifiant -> nom de campagne
    twitch: dict[str, str] = field(default_factory=dict)
    kick: dict[str, str] = field(default_factory=dict)

    def campaign_for(self, platform: str, *names: str | None) -> str | None:
        table = getattr(self, platform, {})
        for n in names:
            if n and n.lower().lstrip("@") in table:
                return table[n.lower().lstrip("@")]
        return None


def load_campaigns(file: Path) -> CampaignTargets:
    """``plateforme:identifiant [nom_campagne]`` par ligne ; ``#`` = commentaire ; ``whop:`` ignoré ici."""
    targets = CampaignTargets()
    if not file.is_file():
        return targets
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        platform, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        ident = parts[0].strip().lower().lstrip("@")
        name = parts[1] if len(parts) > 1 else ident
        if platform.strip().lower() in ("youtube", "twitch", "kick"):
            getattr(targets, platform.strip().lower())[ident] = name
    return targets


def _http_get(client: Any, url: str, *, params: Any = None, headers: Mapping[str, str] | None = None,
              retries: int = 2) -> Any:
    """GET JSON avec 2 nouvelles tentatives sur 429/5xx (backoff exponentiel)."""
    for attempt in range(retries + 1):
        resp = client.get(url, params=params, headers=headers)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(min(8.0, float(resp.headers.get("Retry-After", 2 ** attempt))))
            continue
        resp.raise_for_status()
        return resp.json()
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------
class YouTubeDiscover:
    def __init__(self, service: Any, quota: YouTubeQuota, cfg: Mapping[str, Any], *, budget: int):
        self.service, self.quota, self.cfg = service, quota, cfg
        self.budget = budget            # unités max pour ce run
        self.spent = 0

    def _spend(self, units: int, kind: str) -> None:
        if self.spent + units > self.budget:
            raise QuotaExceeded(f"budget discover YouTube atteint ({self.budget} u)")
        self.quota.spend(units, kind)
        self.spent += units

    def search(self, query: str, *, live: bool, published_after: datetime, limit: int) -> list[str]:
        self._spend(100, "search.list")
        params: dict[str, Any] = {"part": "id", "q": query, "type": "video", "order": "viewCount",
                                  "maxResults": min(50, limit)}
        languages = (self.cfg.get("discover") or {}).get("languages") or []
        if languages:
            params["relevanceLanguage"] = str(languages[0])   # oriente la recherche vers l'audience visée
        if live:
            params["eventType"] = "live"
        else:
            params.update({"videoDuration": "long",
                           "publishedAfter": published_after.isoformat(timespec="seconds").replace("+00:00", "Z")})
        resp = self.service.search().list(**params).execute()
        return [it["id"]["videoId"] for it in resp.get("items", []) if it.get("id", {}).get("videoId")]

    def search_recent(self, query: str, *, video_duration: str, published_after: datetime,
                      limit: int) -> list[str]:
        """Vidéos publiées depuis ``published_after``, les plus vues d'abord (``short`` < 4 min, ``long`` > 20 min)."""
        self._spend(100, "search.list")
        params: dict[str, Any] = {"part": "id", "q": query, "type": "video", "order": "viewCount",
                                  "maxResults": min(50, limit), "videoDuration": video_duration,
                                  "publishedAfter": published_after.isoformat(timespec="seconds").replace("+00:00",
                                                                                                         "Z")}
        languages = (self.cfg.get("discover") or {}).get("languages") or []
        if languages:
            params["relevanceLanguage"] = str(languages[0])
        resp = self.service.search().list(**params).execute()
        return [it["id"]["videoId"] for it in resp.get("items", []) if it.get("id", {}).get("videoId")]

    def channel_uploads(self, ident: str, limit: int = 10) -> list[str]:
        self._spend(1, "channels.list")
        kwargs = {"id": ident} if ident.upper().startswith("UC") and len(ident) == 24 else {"forHandle": f"@{ident}"}
        resp = self.service.channels().list(part="contentDetails", **kwargs).execute()
        items = resp.get("items", [])
        if not items:
            log.warning("Chaîne YouTube introuvable : %s", ident)
            return []
        uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        self._spend(1, "playlistItems.list")
        resp = self.service.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=limit).execute()
        return [it["contentDetails"]["videoId"] for it in resp.get("items", [])]

    def videos(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        unique = list(dict.fromkeys(ids))
        for i in range(0, len(unique), 50):
            self._spend(1, "videos.list")
            resp = self.service.videos().list(part="snippet,contentDetails,statistics,liveStreamingDetails",
                                              id=",".join(unique[i:i + 50]), maxResults=50).execute()
            out.extend(resp.get("items", []))
        return out


def _yt_language(snippet: Mapping[str, Any]) -> str | None:
    """Langue déclarée de la vidéo (audio de préférence) : ``"en-US"`` -> ``"en"``."""
    lang = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
    return str(lang).split("-")[0].lower() if lang else None


def youtube_items(videos: Iterable[Mapping[str, Any]], *, now: datetime, campaigns: CampaignTargets,
                  campaign_ids: set[str] = frozenset()) -> list[SourceItem]:
    items = []
    for v in videos:
        sn, st = v.get("snippet", {}), v.get("statistics", {})
        live = v.get("liveStreamingDetails", {})
        state = sn.get("liveBroadcastContent", "none")
        if state == "upcoming":
            continue
        channel = sn.get("channelTitle", "")
        campaign = campaigns.campaign_for("youtube", sn.get("channelId"), channel) or (
            "campaign" if v.get("id") in campaign_ids else None)
        if state == "live":
            started = parse_ts(live.get("actualStartTime")) or now
            items.append(SourceItem(
                url=f"https://www.youtube.com/watch?v={v['id']}", title=sn.get("title", ""),
                viewers=int(live.get("concurrentViewers") or 0), duration=(now - started).total_seconds(),
                streamer=channel, platform="youtube", status="live", published_at=started.isoformat(),
                views_per_hour=float(live.get("concurrentViewers") or 0), campaign=campaign, id=v["id"],
                extra={"language": _yt_language(sn)}))
        else:
            published = parse_ts(live.get("actualEndTime")) or parse_ts(sn.get("publishedAt")) or now
            views = int(st.get("viewCount") or 0)
            hours = max(1.0, (now - published).total_seconds() / 3600)
            items.append(SourceItem(
                url=f"https://www.youtube.com/watch?v={v['id']}", title=sn.get("title", ""), viewers=views,
                duration=parse_iso8601_duration(v.get("contentDetails", {}).get("duration")), streamer=channel,
                platform="youtube", status="vod", published_at=published.isoformat(), views_per_hour=views / hours,
                campaign=campaign, id=v["id"], extra={"language": _yt_language(sn)}))
    return items


def _youtube_service(cfg: Mapping[str, Any]) -> Any:
    key = get_env(get(cfg, "discover.env.youtube_api_key", "YOUTUBE_API_KEY"))
    if not key:
        raise ConfigError("YOUTUBE_API_KEY absente : YouTube ignoré.")
    from googleapiclient.discovery import build

    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def discover_youtube(cfg: Mapping[str, Any], campaigns: CampaignTargets, *, now: datetime,
                     service: Any = None, limit: int = 25) -> list[SourceItem]:
    dcfg = cfg.get("discover") or {}
    ycfg = dcfg.get("youtube") or {}
    if service is None:
        service = _youtube_service(cfg)
    quota = YouTubeQuota.from_config(cfg)
    budget = min(int(ycfg.get("max_units_per_run", 300)),
                 max(0, int(ycfg.get("daily_budget", 1500)) - _discover_units_today(quota)))
    yt = YouTubeDiscover(service, quota, cfg, budget=budget)
    query = "|".join(f'"{k}"' for k in dcfg.get("keywords", ["GTA 6"]))
    since = now - timedelta(hours=float(dcfg.get("max_vod_age_hours", 24)))
    ids: list[str] = []
    campaign_ids: set[str] = set()
    try:
        for ident in campaigns.youtube:
            got = yt.channel_uploads(ident)
            ids += got
            campaign_ids |= set(got)
        if ycfg.get("search_live", True):
            ids += yt.search(query, live=True, published_after=since, limit=limit)
        if ycfg.get("search_vods", True):
            ids += yt.search(query, live=False, published_after=since, limit=limit)
    except QuotaExceeded as exc:
        log.warning("YouTube : %s — résultats partiels.", exc)
    try:
        videos = yt.videos(ids) if ids else []
    except QuotaExceeded as exc:
        log.warning("YouTube : %s — détails des vidéos indisponibles.", exc)
        videos = []
    log.info("YouTube : %d vidéo(s) examinée(s), %d unité(s) de quota utilisées", len(videos), yt.spent)
    return youtube_items(videos, now=now, campaigns=campaigns, campaign_ids=campaign_ids)


def _discover_units_today(quota: YouTubeQuota) -> int:
    entry = quota.state.load().get("youtube_quota") or {}
    if entry.get("day") != quota.today():
        return 0
    by_kind = entry.get("by_kind", {})
    return sum(int(v) for k, v in by_kind.items() if k != "videos.insert")


# ---------------------------------------------------------------------------
# Twitch
# ---------------------------------------------------------------------------
class TwitchDiscover:
    def __init__(self, client: Any, client_id: str, client_secret: str, token_cache: Path):
        self.client, self.client_id, self.client_secret = client, client_id, client_secret
        self.token_cache = token_cache
        self._token: str | None = None

    def token(self) -> str:
        if self._token:
            return self._token
        cached = read_json(self.token_cache, default={}) or {}
        if cached.get("client_id") == self.client_id and float(cached.get("expires_at", 0)) > time.time() + 300:
            self._token = cached["access_token"]
            return self._token
        resp = self.client.post(TWITCH_TOKEN_URL, data={"client_id": self.client_id,
                                                        "client_secret": self.client_secret,
                                                        "grant_type": "client_credentials"})
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        write_json(self.token_cache, {"client_id": self.client_id, "access_token": self._token,
                                      "expires_at": time.time() + float(data.get("expires_in", 3600))})
        return self._token

    def get(self, endpoint: str, params: Any) -> list[dict[str, Any]]:
        headers = {"Client-Id": self.client_id, "Authorization": f"Bearer {self.token()}"}
        data = _http_get(self.client, f"{TWITCH_API}/{endpoint}", params=params, headers=headers)
        return list((data or {}).get("data", []))

    def game_id(self, name: str) -> str | None:
        games = self.get("games", {"name": name})
        return games[0]["id"] if games else None


def twitch_items(streams: Iterable[Mapping[str, Any]], vods: Iterable[Mapping[str, Any]], *, now: datetime,
                 campaigns: CampaignTargets) -> list[SourceItem]:
    items = []
    for s in streams:
        started = parse_ts(s.get("started_at")) or now
        login = s.get("user_login", "")
        items.append(SourceItem(
            url=f"https://www.twitch.tv/{login}", title=s.get("title", ""), viewers=int(s.get("viewer_count", 0)),
            duration=(now - started).total_seconds(), streamer=login, platform="twitch", status="live",
            published_at=started.isoformat(), views_per_hour=float(s.get("viewer_count", 0)),
            campaign=campaigns.campaign_for("twitch", login), id=str(s.get("id")),
            extra={"language": s.get("language"), "categories": [s.get("game_name") or ""]}))
    for v in vods:
        created = parse_ts(v.get("created_at")) or now
        duration = parse_twitch_duration(v.get("duration"))
        end = created + timedelta(seconds=duration)
        views = int(v.get("view_count", 0))
        login = v.get("user_login", "")
        items.append(SourceItem(
            url=v.get("url") or f"https://www.twitch.tv/videos/{v.get('id')}", title=v.get("title", ""),
            viewers=views, duration=duration, streamer=login, platform="twitch", status="vod",
            published_at=end.isoformat(), views_per_hour=views / max(1.0, (now - end).total_seconds() / 3600),
            campaign=campaigns.campaign_for("twitch", login), id=str(v.get("id")),
            extra={"language": v.get("language"), "categories": [v.get("_category") or ""]}))
    return items


def discover_twitch(cfg: Mapping[str, Any], campaigns: CampaignTargets, *, now: datetime, client: Any = None,
                    limit: int = 25) -> list[SourceItem]:
    dcfg = cfg.get("discover") or {}
    cid = get_env(get(cfg, "discover.env.twitch_client_id", "TWITCH_CLIENT_ID"))
    secret = get_env(get(cfg, "discover.env.twitch_client_secret", "TWITCH_CLIENT_SECRET"))
    if not (cid and secret):
        raise ConfigError("TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET absents : Twitch ignoré.")
    own = client is None
    if own:
        import httpx

        client = httpx.Client(timeout=15)
    try:
        tw = TwitchDiscover(client, cid, secret, resolve_path(cfg, get(cfg, "paths.cache", "cache")) / "twitch_token.json")
        game = tw.game_id(str(dcfg.get("twitch_game_name", "Grand Theft Auto VI")))
        streams: list[dict[str, Any]] = []
        vods: list[dict[str, Any]] = []
        game_name = str(dcfg.get("twitch_game_name", "Grand Theft Auto VI"))
        if game:
            streams += tw.get("streams", {"game_id": game, "first": min(100, limit * 2), "type": "live"})
            for v in tw.get("videos", {"game_id": game, "period": "day", "sort": "views", "type": "archive",
                                       "first": min(100, limit * 2)}):
                vods.append({**v, "_category": game_name})   # l'objet vidéo Helix n'a pas de catégorie
        else:
            log.warning("Catégorie Twitch introuvable : %s", dcfg.get("twitch_game_name"))
        logins = list(campaigns.twitch)
        for i in range(0, len(logins), 100):
            chunk = logins[i:i + 100]
            streams += tw.get("streams", [("user_login", lg) for lg in chunk])
            for user in tw.get("users", [("login", lg) for lg in chunk]):
                vods += tw.get("videos", {"user_id": user["id"], "type": "archive", "first": 5})
        streams = list({s.get("id"): s for s in reversed(streams)}.values())     # dédoublonnage par id
        vods = list({v.get("id"): v for v in reversed(vods)}.values())
        log.info("Twitch : %d live(s), %d VOD(s) examinés", len(streams), len(vods))
        return twitch_items(streams, vods, now=now, campaigns=campaigns)
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------------------
# Kick (lecture légère, endpoints non officiels)
# ---------------------------------------------------------------------------
def kick_items(slug: str, channel: Mapping[str, Any], videos: Iterable[Mapping[str, Any]], *, now: datetime,
               campaign: str | None) -> list[SourceItem]:
    items = []
    streamer = (channel.get("user") or {}).get("username") or slug
    live = channel.get("livestream") or {}
    if live and live.get("is_live", True):
        started = parse_ts(live.get("created_at") or live.get("start_time")) or now
        cats = [c.get("name", "") for c in live.get("categories") or []]
        items.append(SourceItem(
            url=f"https://kick.com/{slug}", title=live.get("session_title", ""),
            viewers=int(live.get("viewer_count") or live.get("viewers") or 0),
            duration=(now - started).total_seconds(), streamer=streamer, platform="kick", status="live",
            published_at=started.isoformat(), views_per_hour=float(live.get("viewer_count") or 0), campaign=campaign,
            id=str(live.get("id")), extra={"categories": cats}))
    for v in videos:
        created = parse_ts(v.get("created_at") or v.get("start_time")) or now
        duration = float(v.get("duration") or 0) / 1000.0           # millisecondes
        end = created + timedelta(seconds=duration)
        views = int(v.get("views") or v.get("view_count") or 0)
        uuid = (v.get("video") or {}).get("uuid") or v.get("uuid") or v.get("id")
        cats = [c.get("name", "") for c in v.get("categories") or []]
        items.append(SourceItem(
            url=f"https://kick.com/{slug}/videos/{uuid}", title=v.get("session_title") or v.get("title") or "",
            viewers=views, duration=duration, streamer=streamer, platform="kick", status="vod",
            published_at=end.isoformat(), views_per_hour=views / max(1.0, (now - end).total_seconds() / 3600),
            campaign=campaign, id=str(uuid), extra={"categories": cats}))
    return items


def discover_kick(cfg: Mapping[str, Any], campaigns: CampaignTargets, *, now: datetime, client: Any = None,
                  limit: int = 25) -> list[SourceItem]:
    kcfg = (cfg.get("discover") or {}).get("kick") or {}
    slugs = list(dict.fromkeys([*campaigns.kick, *[str(s).lower() for s in kcfg.get("channels", [])]]))
    if not slugs:
        log.info("Kick : aucune chaîne dans campaigns.txt / discover.kick.channels.")
        return []
    own = client is None
    if own:
        import httpx

        client = httpx.Client(timeout=15, headers=BROWSER_HEADERS, follow_redirects=True)
    items: list[SourceItem] = []
    try:
        for slug in slugs:
            try:
                channel = _http_get(client, f"{KICK_API}/channels/{slug}") or {}
                videos = _http_get(client, f"{KICK_API}/channels/{slug}/videos") or []
            except Exception as exc:  # noqa: BLE001 — Cloudflare / endpoint modifié : on passe
                log.warning("Kick %s indisponible (%s).", slug, exc)
                continue
            if isinstance(videos, dict):
                videos = videos.get("data", [])
            items += kick_items(slug, channel, videos[: int(kcfg.get("videos_per_channel", 5))], now=now,
                                campaign=campaigns.campaign_for("kick", slug))
            time.sleep(float(kcfg.get("delay_seconds", 0.5)))        # lecture légère, pas de rafale
    finally:
        if own:
            client.close()
    return items


# ---------------------------------------------------------------------------
# Filtre, classement, cache
# ---------------------------------------------------------------------------
def language_ok(item: SourceItem, dcfg: Mapping[str, Any]) -> bool:
    """Langue compatible avec ``discover.languages`` (vide = toutes).

    Langue déclarée par la plateforme si elle existe ; sinon, un titre écrit dans une autre écriture que
    l'alphabet latin (hindi, arabe, russe…) est écarté quand toutes les langues visées s'écrivent en latin.
    """
    from core.scoring import is_latin_text

    wanted = [str(x).lower() for x in dcfg.get("languages") or []]
    if not wanted:
        return True
    lang = str(item.extra.get("language") or "").split("-")[0].lower()
    if lang:
        return lang in wanted
    latin_targets = {"en", "fr", "es", "pt", "de", "it", "nl", "pl", "tr", "id", "vi"}
    if set(wanted) <= latin_targets and not is_latin_text(item.title, threshold=0.6):
        return False
    return bool(dcfg.get("keep_unknown_language", True))


def keep(item: SourceItem, dcfg: Mapping[str, Any], *, now: datetime) -> bool:
    """Live depuis >= N min, ou VOD < 24 h de > N min ; pertinence GTA 6 ; viewers minimum (hors campagne)."""
    min_dur = float(dcfg.get("min_duration_minutes", 45)) * 60
    if item.duration < min_dur:
        return False
    if item.status == "vod":
        published = parse_ts(item.published_at)
        if published is None or now - published > timedelta(hours=float(dcfg.get("max_vod_age_hours", 24))):
            return False
        max_dur = dcfg.get("max_duration_minutes")
        if max_dur and item.duration > float(max_dur) * 60:
            return False                      # VOD trop longue : téléchargement et traitement disproportionnés
    if not language_ok(item, dcfg):
        return False
    keywords = dcfg.get("keywords") or ["GTA 6"]
    categories = [str(c).lower() for c in item.extra.get("categories") or []]
    on_topic = title_matches(item.title, keywords) or any(
        c in ("grand theft auto vi", "gta vi", "gta 6") for c in categories)
    if not on_topic:
        return False
    return item.campaign is not None or item.viewers >= int(dcfg.get("min_viewers", 0))


def rank(items: Iterable[SourceItem]) -> list[SourceItem]:
    unique: dict[str, SourceItem] = {}
    for it in items:
        if it.url not in unique or it.views_per_hour > unique[it.url].views_per_hour:
            unique[it.url] = it
    return sorted(unique.values(), key=lambda it: (it.campaign is None, -it.views_per_hour))


def _cache_file(cfg: Mapping[str, Any], platform: str) -> Path:
    return resolve_path(cfg, get(cfg, "paths.cache", "cache")) / f"discover_{platform}.json"


def _cached(cfg: Mapping[str, Any], platform: str, ttl: float) -> list[SourceItem] | None:
    data = read_json(_cache_file(cfg, platform), default=None)
    if not isinstance(data, dict) or time.time() - float(data.get("ts", 0)) > ttl:
        return None
    return [SourceItem(**it) for it in data.get("items", [])]


Fetcher = Callable[..., list[SourceItem]]
FETCHERS: dict[str, Fetcher] = {"youtube": discover_youtube, "twitch": discover_twitch, "kick": discover_kick}


def discover(cfg: Mapping[str, Any], *, platforms: Sequence[str] | None = None, limit: int | None = None,
             use_cache: bool = True, output: str | Path | None = None, now: datetime | None = None,
             fetchers: Mapping[str, Fetcher] | None = None) -> list[dict[str, Any]]:
    """Interroge les plateformes, filtre, classe et écrit ``data/sources.json``. Retourne les sources."""
    dcfg: Mapping[str, Any] = cfg.get("discover") or {}
    now = now or utc_now()
    platforms = list(platforms or dcfg.get("platforms") or ["youtube", "twitch", "kick"])
    limit = int(limit or dcfg.get("max_results_per_platform", 25))
    ttl = float(dcfg.get("cache_ttl_seconds", 300))
    campaigns = load_campaigns(path(cfg, "campaigns_txt"))
    fetchers = {**FETCHERS, **(fetchers or {})}

    collected: list[SourceItem] = []
    errors: dict[str, str] = {}
    for platform in platforms:
        if platform not in fetchers:
            errors[platform] = "plateforme inconnue"
            continue
        cached = _cached(cfg, platform, ttl) if use_cache else None
        if cached is not None:
            log.info("%s : cache local (< %ds)", platform, int(ttl))
            collected += cached
            continue
        try:
            items = fetchers[platform](cfg, campaigns, now=now, limit=limit)
        except ConfigError as exc:
            errors[platform] = str(exc)
            log.warning("%s", exc)
            continue
        except Exception as exc:  # noqa: BLE001 — une plateforme en panne n'arrête pas les autres
            errors[platform] = f"{type(exc).__name__}: {exc}"
            log.warning("%s indisponible : %s", platform, exc)
            continue
        write_json(_cache_file(cfg, platform), {"ts": time.time(), "items": [asdict(it) for it in items]})
        collected += items

    if errors and len(errors) == len(platforms):
        raise DiscoverError("Aucune plateforme exploitable :\n" + "\n".join(f"  - {p}: {e}" for p, e in errors.items()))

    kept = [it for it in collected if keep(it, dcfg, now=now)]
    ranked = rank(kept)
    per_platform: dict[str, int] = {}
    selected = []
    for it in ranked:
        if per_platform.get(it.platform, 0) < limit:
            per_platform[it.platform] = per_platform.get(it.platform, 0) + 1
            selected.append(it)
    sources = [{k: v for k, v in asdict(it).items() if k != "extra"} | {"duration": round(it.duration),
               "views_per_hour": round(it.views_per_hour, 1), "language": it.extra.get("language")}
               for it in selected]
    out = Path(output) if output else path(cfg, "sources_json")
    write_json(out, {"format": SOURCES_FORMAT, "generated_at": utc_now_iso(), "platforms": platforms,
                     "errors": errors, "count": len(sources), "sources": sources})
    log.info("%d source(s) retenue(s) sur %d examinée(s) -> %s", len(sources), len(collected), out)
    return sources


# ---------------------------------------------------------------------------
# Tendances : ce qui buzze sur GTA 6 dans les dernières 24 h
# ---------------------------------------------------------------------------
TRENDING_FORMAT = "clipgod-trending/1"


def discover_trending(cfg: Mapping[str, Any], *, now: datetime | None = None, service: Any = None,
                      output: str | Path | None = None) -> dict[str, Any]:
    """Repère les moments GTA 6 viraux du jour et en déduit les **VODs d'origine** à clipper.

    1. ``search.list`` Shorts (< 4 min) des dernières ``hours`` h, les plus vus d'abord : les clips viraux ;
    2. ``search.list`` vidéos longues (> 20 min) de la même période : les VODs qui cartonnent ;
    3. pour les ``hot_channels`` chaînes dont les Shorts explosent, leurs dernières vidéos longues
       (streamers qui postent eux-mêmes leurs clips) — ces VODs reçoivent le « boost » vues/heure du Short.

    On ne republie jamais les Shorts des autres : ils servent uniquement de signal. Les clips sont recoupés
    dans les VODs d'origine (crédit au streamer) ; la courbe « most replayed » y localise ensuite le moment viral.
    Écrit ``data/trending.json`` (clips viraux) et ``data/sources.json`` (VODs à préparer, classées).
    Coût : ~200 + 2 × hot_channels + quelques unités de quota (sur 10 000/jour).
    """
    dcfg: Mapping[str, Any] = cfg.get("discover") or {}
    tcfg: Mapping[str, Any] = dcfg.get("trending") or {}
    now = now or utc_now()
    service = service or _youtube_service(cfg)
    quota = YouTubeQuota.from_config(cfg)
    budget = min(int(tcfg.get("max_units", 400)),
                 max(0, int((dcfg.get("youtube") or {}).get("daily_budget", 1500)) - _discover_units_today(quota)))
    yt = YouTubeDiscover(service, quota, cfg, budget=budget)
    query = "|".join(f'"{k}"' for k in dcfg.get("keywords", ["GTA 6"]))
    since = now - timedelta(hours=float(tcfg.get("hours", 24)))
    limit = int(tcfg.get("results", 25))
    campaigns = load_campaigns(path(cfg, "campaigns_txt"))

    short_ids: list[str] = []
    long_ids: list[str] = []
    try:
        short_ids = yt.search_recent(query, video_duration="short", published_after=since, limit=limit)
        long_ids = yt.search_recent(query, video_duration="long", published_after=since, limit=limit)
    except QuotaExceeded as exc:
        log.warning("YouTube : %s — tendances partielles.", exc)
    details = {v["id"]: v for v in (yt.videos(short_ids + long_ids) if short_ids or long_ids else [])}

    shorts = youtube_items([details[i] for i in short_ids if i in details], now=now, campaigns=campaigns)
    viral = sorted((it for it in shorts if title_matches(it.title, dcfg.get("keywords") or ["GTA 6"])
                    and language_ok(it, dcfg)), key=lambda it: -it.views_per_hour)

    # Chaînes dont les Shorts explosent : leurs dernières VODs, avec le boost du Short.
    boost: dict[str, tuple[float, str]] = {}
    hot_ids: list[str] = []
    seen_channels: set[str] = set()
    for it in viral:
        channel_id = (details.get(it.id or "", {}).get("snippet") or {}).get("channelId")
        if not channel_id or channel_id in seen_channels:
            continue
        seen_channels.add(channel_id)
        if len(seen_channels) > int(tcfg.get("hot_channels", 5)):
            break
        try:
            uploads = yt.channel_uploads(channel_id, limit=int(tcfg.get("uploads_per_channel", 8)))
        except QuotaExceeded as exc:
            log.warning("YouTube : %s", exc)
            break
        for vid in uploads:
            boost.setdefault(vid, (it.views_per_hour, it.title))
        hot_ids += [v for v in uploads if v not in details]
    if hot_ids:
        try:
            details.update({v["id"]: v for v in yt.videos(hot_ids)})
        except QuotaExceeded as exc:
            log.warning("YouTube : %s", exc)

    vod_ids = list(dict.fromkeys(long_ids + [v for v in boost if v in details]))
    vods = youtube_items([details[i] for i in vod_ids if i in details], now=now, campaigns=campaigns)
    vod_cfg = {**dcfg, "max_vod_age_hours": float(tcfg.get("max_vod_age_hours", 48)), "min_viewers": 0}
    candidates = []
    for it in vods:
        if it.status != "vod" or not keep(it, vod_cfg, now=now):
            continue
        b, why = boost.get(it.id or "", (0.0, ""))
        it.extra["boost"] = b
        it.extra["reason"] = f"Short viral : {why[:60]}" if why else "VOD très vue"
        candidates.append(it)
    candidates.sort(key=lambda it: (it.campaign is None, -(it.views_per_hour + it.extra["boost"])))

    sources = [{k: v for k, v in asdict(it).items() if k != "extra"} | {
        "duration": round(it.duration), "views_per_hour": round(it.views_per_hour, 1),
        "language": it.extra.get("language"), "trending": True, "boost": round(it.extra["boost"], 1),
        "reason": it.extra["reason"]} for it in candidates]
    viral_out = [{"url": it.url, "title": it.title, "channel": it.streamer, "views": it.viewers,
                  "views_per_hour": round(it.views_per_hour, 1), "published_at": it.published_at} for it in viral]
    doc = {"format": TRENDING_FORMAT, "generated_at": utc_now_iso(), "hours": float(tcfg.get("hours", 24)),
           "viral": viral_out, "sources": sources, "units": yt.spent}
    write_json(resolve_path(cfg, get(cfg, "paths.trending", "data/trending.json")), doc)
    out = Path(output) if output else path(cfg, "sources_json")
    write_json(out, {"format": SOURCES_FORMAT, "generated_at": utc_now_iso(), "platforms": ["youtube"],
                     "errors": {}, "count": len(sources), "sources": sources, "trending": True})
    log.info("Tendances : %d Short(s) viral(aux), %d VOD(s) d'origine, %d unité(s) de quota",
             len(viral_out), len(sources), yt.spent)
    return doc


def best_source(config: Mapping[str, Any]) -> str:
    """URL de la meilleure **VOD** (``clipgod run`` sans --source). Les lives relèvent de ``clipgod watch``."""
    sources = discover(config)
    vods = [s for s in sources if s["status"] == "vod"]
    if not vods:
        raise DiscoverError("Aucune VOD GTA 6 récente trouvée. Essayez clipgod discover --no-cache, ou --source <url>.")
    return str(vods[0]["url"])
