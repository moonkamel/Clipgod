"""Module 5 — publish (laptop) : valide puis publie / programme les clips de ``output/clips/``.

Déroulé de ``clipgod publish`` :
    1. candidats  clips ``synced`` / ``validated`` de ``data/state.json`` (ou ``--clip``), avec leur sidecar JSON
    2. validation ``monetize.validate_clip`` (durée < 30 s, hook dans la 1re seconde, vraie transformation…) ;
                  refus -> ``data/rejected.json`` + statut ``rejected``
    3. plan       A/B test : les variantes de hook d'un même moment partent sur des **créneaux différents**
                  (et tournent entre les plateformes) ; meilleurs moments d'abord ; max ``max_posts_per_day``
                  posts par jour sur les créneaux de forte audience (matin EU, soirée US)
    4. envoi      par plateforme selon ``publish.routes`` :
                    youtube_shorts -> API YouTube officielle (OAuth, programmation native ``publishAt``) ;
                    tiktok / instagram_reels -> Upload-Post (optionnel, clé ``UPLOAD_POST_API_KEY``) ;
                    dry_run -> rien n'est envoyé, le plan est enregistré.
                  Quota YouTube vérifié avant chaque upload (1600 u sur 10 000/jour, partagé avec discover).
    5. état       ``state["posts"]`` (plateforme, créneau, id/URL distants) ; clip ``scheduled`` / ``published``.

Mode test (défaut, ``--test``) : YouTube en **privé** sans programmation ; les autres plateformes sont sautées.
NB : tant que le projet Google Cloud n'a pas passé l'audit YouTube, l'API force de toute façon les uploads en privé.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, ClassVar, Protocol
from zoneinfo import ZoneInfo

from core.config import ConfigError, get, get_env, path, resolve_path
from core.logger import get_logger
from core.quota import QuotaExceeded, YouTubeQuota
from core.storage import StateStore, atomic_write_text, read_json, utc_now, utc_now_iso
from modules.monetize import Rules, record_rejection, validate_clip

log = get_logger(__name__)

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
PUBLISHED_STATES = ("scheduled", "published")


class PublishError(RuntimeError):
    """Échec d'un envoi (réseau, API, authentification)."""


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
@dataclass
class ClipInfo:
    clip_id: str
    path: Path
    meta: dict[str, Any]

    @property
    def group(self) -> str:
        """Moment d'origine : les variantes A/B d'un même moment partagent le même groupe."""
        return f"{self.meta.get('video_id')}_{self.meta.get('moment_idx', 0)}"

    @property
    def variant(self) -> int:
        return int(self.meta.get("variant", 0) or 0)

    @property
    def score(self) -> float:
        return float(self.meta.get("score") or 0)


@dataclass
class Post:
    clip: ClipInfo
    platform: str
    slot: datetime | None          # None = immédiat
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def post_id(self) -> str:
        return f"{self.clip.clip_id}@{self.platform}"


@dataclass
class PublishResult:
    remote_id: str | None
    url: str | None
    status: str                    # scheduled | published | private | planned
    privacy: str | None = None


class Backend(Protocol):
    name: str

    def publish(self, post: Post, *, test: bool) -> PublishResult: ...


# ---------------------------------------------------------------------------
# Métadonnées de publication
# ---------------------------------------------------------------------------
def build_metadata(clip: ClipInfo, pcfg: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    """Titre (hook + hashtags, <= 100 car.), description avec crédit du streamer, tags."""
    hashtags = list(pcfg.get("hashtags") or ["#gta6", "#gtavi", "#shorts"])
    hook = re.sub(r"\s+", " ", str(clip.meta.get("hook") or "GTA 6 moment")).strip()
    suffix = " " + " ".join(hashtags[:2])
    title = (hook[: 100 - len(suffix)].rstrip() + suffix).strip()
    streamer = str(clip.meta.get("streamer") or "").lstrip("@")
    credit = str(pcfg.get("credit_template", "🎮 Credit: @{streamer}")).format(streamer=streamer) if streamer else ""
    lines = [hook, "", credit, f"Source: {clip.meta['source_url']}" if clip.meta.get("source_url") else "",
             "", " ".join(hashtags)]
    description = "\n".join(line for line in lines if line is not None).strip()
    tags = [h.lstrip("#") for h in hashtags] + ([streamer] if streamer else []) + ["GTA 6", "GTA VI"]
    return title, description[:4900], list(dict.fromkeys(tags))


# ---------------------------------------------------------------------------
# Créneaux
# ---------------------------------------------------------------------------
def slot_times(pcfg: Mapping[str, Any]) -> list[dtime]:
    out = []
    for values in (pcfg.get("slots") or {}).values():
        for v in values:
            h, m = (int(x) for x in str(v).split(":"))
            out.append(dtime(h, m))
    return sorted(set(out)) or [dtime(18, 0)]


def next_slots(n: int, *, now: datetime, tz: ZoneInfo, times: Sequence[dtime], max_per_day: int,
               taken: Sequence[datetime], min_delay: timedelta = timedelta(minutes=20),
               horizon_days: int = 14) -> list[datetime]:
    """``n`` prochains créneaux libres (heure locale ``tz``), en respectant ``max_per_day`` posts par jour local."""
    taken_set = {t.astimezone(tz).replace(second=0, microsecond=0) for t in taken}
    per_day: dict[str, int] = {}
    for t in taken_set:
        per_day[t.date().isoformat()] = per_day.get(t.date().isoformat(), 0) + 1
    out: list[datetime] = []
    local_now = now.astimezone(tz)
    for day in range(horizon_days):
        date = (local_now + timedelta(days=day)).date()
        for t in times:
            if len(out) >= n:
                return out
            slot = datetime.combine(date, t, tzinfo=tz)
            key = date.isoformat()
            if slot < local_now + min_delay or slot in taken_set or per_day.get(key, 0) >= max_per_day:
                continue
            out.append(slot)
            per_day[key] = per_day.get(key, 0) + 1
    return out


def plan_posts(clips: Sequence[ClipInfo], platforms: Sequence[str], *, already: set[str]) -> list[tuple[ClipInfo, str]]:
    """Ordre de publication et plateforme de chaque variante.

    Les meilleurs moments d'abord ; toutes les 1res variantes, puis toutes les 2es… (deux variantes d'un même moment
    ne se suivent jamais) ; la variante k part sur la plateforme ``platforms[k % n]``.
    Une paire (clip, plateforme) déjà publiée/programmée n'est jamais reprise.
    """
    groups: dict[str, list[ClipInfo]] = {}
    for c in clips:
        groups.setdefault(c.group, []).append(c)
    ordered_groups = sorted(groups.values(), key=lambda g: -max(c.score for c in g))
    for g in ordered_groups:
        g.sort(key=lambda c: c.variant)
    plan: list[tuple[ClipInfo, str]] = []
    depth = max((len(g) for g in ordered_groups), default=0)
    for k in range(depth):
        for g in ordered_groups:
            if k >= len(g):
                continue
            clip = g[k]
            for offset in range(len(platforms)):
                platform = platforms[(k + offset) % len(platforms)]
                if f"{clip.clip_id}@{platform}" not in already:
                    plan.append((clip, platform))
                    break
    return plan


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class DryRunBackend:
    name = "dry_run"

    def publish(self, post: Post, *, test: bool) -> PublishResult:
        return PublishResult(None, None, "planned")


class YouTubeBackend:
    """Upload YouTube Shorts via l'API officielle (videos.insert, upload résumable)."""

    name = "youtube_api"

    def __init__(self, cfg: Mapping[str, Any], *, service: Any = None, quota: YouTubeQuota | None = None):
        self.cfg = cfg
        self._service = service
        self.quota = quota or YouTubeQuota.from_config(cfg)
        self.cost = int(get(cfg, "publish.youtube_quota.upload_cost", 1600))

    def service(self) -> Any:
        if self._service is None:
            from googleapiclient.discovery import build

            self._service = build("youtube", "v3", credentials=youtube_credentials(self.cfg), cache_discovery=False)
        return self._service

    def publish(self, post: Post, *, test: bool) -> PublishResult:
        if not self.quota.can_spend(self.cost):
            raise QuotaExceeded(f"Quota YouTube insuffisant (reste {self.quota.remaining()} u, upload = {self.cost} u).")
        status: dict[str, Any] = {"selfDeclaredMadeForKids": False}
        if test:
            status["privacyStatus"] = "private"
        elif post.slot is not None:
            status.update(privacyStatus="private",              # obligatoire pour publishAt
                          publishAt=post.slot.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"))
        else:
            status["privacyStatus"] = str(get(self.cfg, "publish.youtube_privacy", "public"))
        body = {"snippet": {"title": post.title, "description": post.description, "tags": post.tags[:30],
                            "categoryId": "20"},                  # 20 = Gaming
                "status": status}
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(post.clip.path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
        request = self.service().videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        try:
            while response is None:
                _, response = request.next_chunk(num_retries=5)
        except Exception as exc:
            raise PublishError(f"YouTube : {exc}") from exc
        self.quota.spend(self.cost, "videos.insert")
        vid = response.get("id")
        real_privacy = (response.get("status") or {}).get("privacyStatus", status["privacyStatus"])
        state = "private" if test else ("scheduled" if "publishAt" in status else "published")
        return PublishResult(vid, f"https://youtube.com/shorts/{vid}", state, real_privacy)


class UploadPostBackend:
    """Publication TikTok / Instagram via l'API Upload-Post (optionnel, clé ``UPLOAD_POST_API_KEY``).

    Endpoint et champs configurables (``publish.upload_post``) : vérifiez-les dans la documentation Upload-Post.
    """

    name = "upload_post"
    PLATFORM_NAMES: ClassVar[dict[str, str]] = {"tiktok": "tiktok", "instagram_reels": "instagram",
                                               "youtube_shorts": "youtube"}

    def __init__(self, cfg: Mapping[str, Any], *, client: Any = None):
        self.cfg = cfg
        ucfg = get(cfg, "publish.upload_post", {}) or {}
        self.endpoint = str(ucfg.get("endpoint", "https://api.upload-post.com/api/upload"))
        key = get_env(get(cfg, "publish.env.upload_post_api_key", "UPLOAD_POST_API_KEY"))
        user = get_env(str(ucfg.get("user_env", "UPLOAD_POST_USER")))
        if not key or not user:
            raise ConfigError("UPLOAD_POST_API_KEY / UPLOAD_POST_USER absents : TikTok/Instagram désactivés.")
        self.key, self.user = key, user
        self._client = client

    def publish(self, post: Post, *, test: bool) -> PublishResult:
        if test:
            return PublishResult(None, None, "skipped")
        import httpx

        data: dict[str, Any] = {"user": self.user, "title": post.title,
                                "platform[]": [self.PLATFORM_NAMES.get(post.platform, post.platform)]}
        if post.slot is not None:
            data["scheduled_date"] = post.slot.isoformat()
        client = self._client or httpx.Client(timeout=300)
        try:
            with open(post.clip.path, "rb") as fh:
                resp = client.post(self.endpoint, headers={"Authorization": f"Apikey {self.key}"}, data=data,
                                   files={"video": (post.clip.path.name, fh, "video/mp4")})
        finally:
            if self._client is None:
                client.close()
        if resp.status_code >= 400:
            raise PublishError(f"Upload-Post HTTP {resp.status_code} : {resp.text[:300]}")
        payload = resp.json() if resp.content else {}
        return PublishResult(str(payload.get("request_id") or payload.get("id") or ""), payload.get("url"),
                             "scheduled" if post.slot else "published")


def youtube_credentials(cfg: Mapping[str, Any]) -> Any:
    """OAuth « application de bureau » pour l'upload YouTube (token mis en cache dans secrets/)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = resolve_path(cfg, get_env("CLIPGOD_YOUTUBE_TOKEN") or "secrets/youtube_token.json")
    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        secrets = get_env(get(cfg, "publish.env.youtube_client_secrets", "YOUTUBE_CLIENT_SECRETS"), required=True,
                          hint="ID client OAuth « Application de bureau » avec l'API YouTube Data v3 activée.")
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = InstalledAppFlow.from_client_secrets_file(str(resolve_path(cfg, secrets)), YOUTUBE_SCOPES
                                                          ).run_local_server(port=0, open_browser=True)
    atomic_write_text(token_path, creds.to_json())
    with contextlib.suppress(OSError):
        os.chmod(token_path, 0o600)
    return creds


def build_backends(cfg: Mapping[str, Any], platforms: Sequence[str], *, dry_run: bool) -> dict[str, Backend]:
    """Plateforme -> backend selon ``publish.routes`` ; une plateforme sans identifiants est retirée (warning)."""
    routes = get(cfg, "publish.routes", {}) or {}
    out: dict[str, Backend] = {}
    for platform in platforms:
        kind = "dry_run" if dry_run else routes.get(platform, "dry_run")
        try:
            out[platform] = {"dry_run": DryRunBackend, "youtube_api": YouTubeBackend,
                             "upload_post": UploadPostBackend}[kind](*(() if kind == "dry_run" else (cfg,)))
        except KeyError:
            log.warning("Backend inconnu pour %s : %s", platform, kind)
        except ConfigError as exc:
            log.warning("%s", exc)
    return out


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------
@dataclass
class PublishReport:
    posted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def load_candidates(cfg: Mapping[str, Any], state: StateStore, clip: str | Path | None) -> list[ClipInfo]:
    if clip is not None:
        p = Path(clip)
        if not p.is_file():
            raise PublishError(f"Clip introuvable : {p}")
        return [ClipInfo(p.stem, p, read_json(p.with_suffix(".json"), default={}) or {})]
    out = []
    for clip_id, entry in state.clips().items():
        if entry.get("status") not in ("synced", "validated"):
            continue
        p = Path(entry.get("path") or "")
        if not p.is_file():
            log.warning("Clip %s : fichier absent (%s), relancez clipgod sync.", clip_id, p)
            continue
        out.append(ClipInfo(clip_id, p, read_json(p.with_suffix(".json"), default={}) or {}))
    return out


def publish(cfg: Mapping[str, Any], *, clip: str | Path | None = None, platforms: Sequence[str] | None = None,
            test: bool = True, schedule: bool = True, skip_validation: bool = False, dry_run: bool = False,
            backends: Mapping[str, Backend] | None = None, now: datetime | None = None) -> PublishReport:
    pcfg: Mapping[str, Any] = cfg.get("publish") or {}
    state = StateStore(path(cfg, "state"))
    report = PublishReport()
    tz = ZoneInfo(str(get(cfg, "project.timezone", "Europe/Paris")))
    now = now or utc_now()

    candidates = load_candidates(cfg, state, clip)
    if not skip_validation:
        rules = Rules.from_config(cfg)
        valid = []
        for c in candidates:
            result = validate_clip({"clip_id": c.clip_id, **c.meta}, rules, c.path)
            if result.ok:
                valid.append(c)
                if not dry_run:
                    state.upsert_clip(c.clip_id, status="validated")
            else:
                report.rejected.append((c.clip_id, result.reasons))
                if not dry_run:
                    record_rejection(cfg, result)
                    state.upsert_clip(c.clip_id, status="rejected", rejected_reasons=result.reasons)
        candidates = valid

    wanted = list(platforms or pcfg.get("platforms") or ["youtube_shorts"])
    if test:
        wanted = [p for p in wanted if p == "youtube_shorts"] or wanted[:1]
    backends = dict(backends) if backends is not None else build_backends(cfg, wanted, dry_run=dry_run)
    active = [p for p in wanted if p in backends]
    if not active:
        report.errors.append(("*", "aucune plateforme utilisable (identifiants manquants ?)"))
        return report

    posts_state = state.load().get("posts", {})
    already = {pid for pid, p in posts_state.items() if p.get("status") in (*PUBLISHED_STATES, "private")}
    plan = plan_posts(candidates, active, already=already)
    max_per_day = int(pcfg.get("max_posts_per_day", 6))
    if schedule and not test:
        taken = [t for p in posts_state.values() if p.get("slot") and (t := _parse(p["slot"])) and t > now]
        slots = next_slots(len(plan), now=now, tz=tz, times=slot_times(pcfg), max_per_day=max_per_day, taken=taken)
    else:
        today = now.astimezone(tz).date().isoformat()
        done_today = sum(1 for p in posts_state.values()
                         if (_parse(p.get("posted_at")) or now).astimezone(tz).date().isoformat() == today)
        slots = [None] * max(0, max_per_day - done_today)
    if len(slots) < len(plan):
        for c, platform in plan[len(slots):]:
            report.skipped.append((f"{c.clip_id}@{platform}", f"limite de {max_per_day} posts/jour atteinte"))
        plan = plan[: len(slots)]

    blocked: set[str] = set()
    for (c, platform), slot in zip(plan, slots, strict=False):
        title, description, tags = build_metadata(c, pcfg)
        post = Post(c, platform, slot, title, description, tags)
        if platform in blocked:
            report.skipped.append((post.post_id, "plateforme bloquée pour ce run (quota/erreur)"))
            continue
        try:
            result = backends[platform].publish(post, test=test)
        except QuotaExceeded as exc:
            blocked.add(platform)
            report.skipped.append((post.post_id, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — un post en échec n'arrête pas les autres
            report.errors.append((post.post_id, str(exc)))
            log.error("Publication %s impossible : %s", post.post_id, exc)
            continue
        if result.status == "skipped":
            report.skipped.append((post.post_id, "mode test"))
            continue
        entry = {"clip_id": c.clip_id, "platform": platform, "backend": getattr(backends[platform], "name", "?"),
                 "slot": slot.isoformat() if slot else None, "status": result.status, "remote_id": result.remote_id,
                 "url": result.url, "privacy": result.privacy, "test": test, "title": title,
                 "hook": c.meta.get("hook"), "variant": c.variant, "group": c.group, "posted_at": utc_now_iso()}
        report.posted.append({"post_id": post.post_id, **entry})
        if dry_run or result.status == "planned":
            continue
        with state.transaction() as data:
            data.setdefault("posts", {})[post.post_id] = entry
        state.upsert_clip(c.clip_id, status="published" if result.status in ("published", "private") else "scheduled",
                          last_post=post.post_id)
        log.info("✔ %s -> %s (%s)", c.clip_id, platform, result.url or result.status)
    return report


def _parse(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None

