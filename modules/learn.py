"""Module 7 — learn (laptop, CPU) : apprend des performances réelles pour améliorer le scoring de viralité.

``clipgod learn`` :
    1. collecte   vues actuelles de chaque post publié -> ``state["observations"][post_id]`` (série temporelle)
                    - YouTube : ``videos.list(part=statistics)`` (1 unité / 50 vidéos ; OAuth -> vidéos privées incluses)
                    - Upload-Post (TikTok/Instagram) : endpoint d'analytics configurable, si disponible
                    - saisie manuelle : ``data/manual_views.csv`` (post_id ou url, views[, measured_at])
    2. historique vues à 1 h / 6 h / 24 h / 72 h par interpolation linéaire entre mesures (jamais d'extrapolation)
                  -> ``data/historical_scores.csv`` (réécrit à chaque run : idempotent)
    3. modèle     tous les ``retrain_every_n_clips`` nouveaux clips ou ``retrain_every_days`` jours (ou --retrain) :
                  GradientBoostingRegressor(signaux du moment -> log1p(vues) - médiane de la plateforme)
                  -> ``data/model.pkl`` (format :data:`core.scoring.MODEL_FORMAT`), envoyé sur Drive pour le scan suivant
    4. rapport    ``report.html`` : top/flop, hooks gagnants (A/B), créneaux, campagnes rentables, recommandations
"""

from __future__ import annotations

import csv
import html
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.config import get, get_env, path, resolve_path
from core.logger import get_logger
from core.scoring import FEATURES, MODEL_FORMAT, vectorize
from core.storage import StateStore, atomic_write_text, read_json, utc_now, utc_now_iso

log = get_logger(__name__)

CSV_COLUMNS = ["clip_id", "signals", "hook", "platform", "views_1h", "views_6h", "views_24h", "views_72h",
               "post_id", "campaign", "variant", "group", "score", "published_at", "url"]


# ===========================================================================
# Collecte
# ===========================================================================
def _parse(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def published_at(post: Mapping[str, Any]) -> datetime | None:
    """Heure de mise en ligne : créneau programmé s'il y en a un, sinon heure d'envoi."""
    return _parse(post.get("slot")) or _parse(post.get("posted_at"))


def youtube_stats(cfg: Mapping[str, Any], video_ids: Sequence[str], *, service: Any = None) -> dict[str, int]:
    """Vues YouTube par id (1 unité de quota par paquet de 50)."""
    if not video_ids:
        return {}
    from core.quota import YouTubeQuota

    if service is None:
        from googleapiclient.discovery import build

        token = resolve_path(cfg, get_env("CLIPGOD_YOUTUBE_TOKEN") or "secrets/youtube_token.json")
        if token.is_file():                              # OAuth : voit aussi les vidéos privées (mode test)
            from modules.publish import youtube_credentials

            service = build("youtube", "v3", credentials=youtube_credentials(cfg), cache_discovery=False)
        else:
            key = get_env(get(cfg, "discover.env.youtube_api_key", "YOUTUBE_API_KEY"))
            if not key:
                log.warning("Ni token YouTube (publish) ni YOUTUBE_API_KEY : vues YouTube non collectées.")
                return {}
            service = build("youtube", "v3", developerKey=key, cache_discovery=False)
    quota = YouTubeQuota.from_config(cfg)
    out: dict[str, int] = {}
    for i in range(0, len(video_ids), 50):
        quota.spend(1, "videos.list")
        resp = service.videos().list(part="statistics", id=",".join(video_ids[i:i + 50]), maxResults=50).execute()
        for item in resp.get("items", []):
            out[item["id"]] = int((item.get("statistics") or {}).get("viewCount") or 0)
    return out


def upload_post_stats(cfg: Mapping[str, Any], posts: Mapping[str, Mapping[str, Any]], *, client: Any = None) -> dict[str, int]:
    """Vues via l'endpoint d'analytics Upload-Post, s'il est configuré (``publish.upload_post.analytics_endpoint``)."""
    endpoint = get(cfg, "publish.upload_post.analytics_endpoint", None)
    key = get_env(get(cfg, "publish.env.upload_post_api_key", "UPLOAD_POST_API_KEY"))
    if not endpoint or not key or not posts:
        return {}
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=30)
    out: dict[str, int] = {}
    try:
        for post_id, post in posts.items():
            try:
                resp = client.get(str(endpoint).format(request_id=post.get("remote_id")),
                                  headers={"Authorization": f"Apikey {key}"})
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                log.warning("Analytics Upload-Post indisponibles pour %s : %s", post_id, exc)
                continue
            views = data.get("views") or data.get("view_count") or (data.get("stats") or {}).get("views")
            if views is not None:
                out[post_id] = int(views)
    finally:
        if own:
            client.close()
    return out


def manual_views(cfg: Mapping[str, Any], posts: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, datetime, int]]:
    """``data/manual_views.csv`` : colonnes ``post_id`` ou ``url``, ``views``, ``measured_at`` (optionnel)."""
    file = resolve_path(cfg, get(cfg, "learn.manual_views_file", "data/manual_views.csv"))
    if not file.is_file():
        return []
    by_url = {p.get("url"): pid for pid, p in posts.items() if p.get("url")}
    out = []
    with open(file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("post_id") or by_url.get(row.get("url"))
            if not pid or pid not in posts or not str(row.get("views", "")).strip():
                continue
            when = _parse(row.get("measured_at")) or utc_now()
            out.append((pid, when, int(float(str(row["views"]).replace(",", "").replace(" ", "")))))
    return out


@dataclass
class CollectReport:
    observed: int = 0
    by_platform: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def collect(cfg: Mapping[str, Any], *, now: datetime | None = None, youtube_service: Any = None,
            http_client: Any = None) -> CollectReport:
    """Ajoute une mesure de vues pour chaque post en ligne depuis moins de 72 h (+ marge)."""
    now = now or utc_now()
    state = StateStore(path(cfg, "state"))
    data = state.load()
    posts: dict[str, dict[str, Any]] = data.get("posts", {})
    horizon = timedelta(hours=max(get(cfg, "learn.checkpoints_hours", [1, 6, 24, 72])) * 1.5)
    live = {pid: p for pid, p in posts.items()
            if p.get("remote_id") and (pub := published_at(p)) and pub <= now <= pub + horizon}
    report = CollectReport()
    measures: list[tuple[str, datetime, int]] = []

    yt = {pid: p for pid, p in live.items() if p.get("platform") == "youtube_shorts"}
    try:
        views = youtube_stats(cfg, [p["remote_id"] for p in yt.values()], service=youtube_service)
    except Exception as exc:  # noqa: BLE001 — quota, réseau… : on continue avec les autres sources
        log.warning("Statistiques YouTube indisponibles : %s", exc)
        views = {}
    measures += [(pid, now, views[p["remote_id"]]) for pid, p in yt.items() if p["remote_id"] in views]
    others = {pid: p for pid, p in live.items() if pid not in yt}
    measures += [(pid, now, v) for pid, v in upload_post_stats(cfg, others, client=http_client).items()]
    measures += manual_views(cfg, posts)

    with state.transaction() as doc:
        obs = doc.setdefault("observations", {})
        for pid, when, v in measures:
            series = obs.setdefault(pid, [])
            stamp = when.isoformat()
            if not any(o["t"] == stamp for o in series):
                series.append({"t": stamp, "views": v})
                series.sort(key=lambda o: o["t"])
                report.observed += 1
                platform = posts[pid].get("platform", "?")
                report.by_platform[platform] = report.by_platform.get(platform, 0) + 1
    seen = {m[0] for m in measures}
    report.missing = [pid for pid in live if pid not in seen]
    return report


# ===========================================================================
# Historique
# ===========================================================================
def views_at(series: Sequence[Mapping[str, Any]], published: datetime, hours: float) -> float | None:
    """Vues à ``hours`` h après publication : interpolation linéaire (0 vue à la publication), sans extrapolation."""
    points = [(0.0, 0.0)] + sorted(
        ((t - published).total_seconds() / 3600, float(o["views"]))
        for o in series if (t := _parse(o["t"])) is not None and t >= published
    )
    if points[-1][0] < hours:
        return None
    for (h0, v0), (h1, v1) in itertools.pairwise(points):
        if h0 <= hours <= h1:
            return v0 if h1 == h0 else v0 + (v1 - v0) * (hours - h0) / (h1 - h0)
    return None  # pragma: no cover


def build_history(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Une ligne par post (clip x plateforme), avec signaux du moment et vues aux checkpoints."""
    data = StateStore(path(cfg, "state")).load()
    clips = data.get("clips", {})
    observations = data.get("observations", {})
    checkpoints = [int(h) for h in get(cfg, "learn.checkpoints_hours", [1, 6, 24, 72])]
    rows = []
    for pid, post in data.get("posts", {}).items():
        pub = published_at(post)
        if pub is None or post.get("status") == "planned":
            continue
        clip = clips.get(post.get("clip_id"), {})
        sidecar = read_json(Path(clip["path"]).with_suffix(".json"), default={}) if clip.get("path") else {}
        row: dict[str, Any] = {
            "clip_id": post.get("clip_id"), "signals": json.dumps((sidecar or {}).get("signals") or {}, sort_keys=True),
            "hook": post.get("hook"), "platform": post.get("platform"), "post_id": pid,
            "campaign": post.get("campaign") or clip.get("campaign"), "variant": post.get("variant"),
            "group": post.get("group"), "score": (sidecar or {}).get("score", clip.get("score")),
            "published_at": pub.isoformat(), "url": post.get("url"),
        }
        series = observations.get(pid, [])
        for h in checkpoints:
            v = views_at(series, pub, h) if series else None
            row[f"views_{h}h"] = round(v) if v is not None else None
        rows.append(row)
    return rows


def write_history(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Path:
    file = path(cfg, "historical_scores")
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    atomic_write_text(file, buf.getvalue())
    return file


# ===========================================================================
# Modèle
# ===========================================================================
def training_set(rows: Sequence[Mapping[str, Any]], target: str) -> tuple[list[list[float]], list[float], dict[str, float]]:
    """X (signaux), y = log1p(vues) - médiane log1p de la plateforme, médianes par plateforme."""
    usable = [r for r in rows if r.get(target) not in (None, "") and r.get("signals")]
    logs: dict[str, list[float]] = defaultdict(list)
    for r in usable:
        logs[str(r.get("platform"))].append(math.log1p(float(r[target])))
    medians = {p: statistics.median(v) for p, v in logs.items()}
    X, y = [], []
    for r in usable:
        signals = json.loads(r["signals"]) if isinstance(r["signals"], str) else r["signals"]
        X.append(vectorize(signals))
        y.append(math.log1p(float(r[target])) - medians[str(r.get("platform"))])
    return X, y, medians


def should_retrain(cfg: Mapping[str, Any], n_samples: int, *, now: datetime) -> tuple[bool, str]:
    lcfg = cfg.get("learn") or {}
    meta = StateStore(path(cfg, "state")).load().get("model_meta") or {}
    if n_samples < int(lcfg.get("min_samples_to_train", 10)):
        return False, f"{n_samples} clip(s) avec vues à {lcfg.get('target', 'views_24h')} (minimum {lcfg.get('min_samples_to_train', 10)})"
    if not meta:
        return True, "premier entraînement"
    new = n_samples - int(meta.get("n_samples", 0))
    if new >= int(lcfg.get("retrain_every_n_clips", 20)):
        return True, f"{new} nouveaux clips"
    last = _parse(meta.get("trained_at"))
    if last is None or now - last >= timedelta(days=float(lcfg.get("retrain_every_days", 7))):
        return True, "réentraînement hebdomadaire"
    return False, f"modèle à jour ({new} nouveau(x) clip(s) depuis le dernier entraînement)"


def train(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    """Entraîne le GradientBoostingRegressor et écrit ``data/model.pkl``. Retourne les métadonnées."""
    import joblib
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score

    lcfg = cfg.get("learn") or {}
    mcfg = lcfg.get("model") or {}
    target = str(lcfg.get("target", "views_24h"))
    X, y, medians = training_set(rows, target)
    if len(X) < 2:
        raise ValueError("Pas assez de données pour entraîner.")
    model = GradientBoostingRegressor(
        n_estimators=int(mcfg.get("n_estimators", 300)), learning_rate=float(mcfg.get("learning_rate", 0.05)),
        max_depth=int(mcfg.get("max_depth", 3)), subsample=float(mcfg.get("subsample", 0.9)),
        random_state=int(mcfg.get("random_state", 42)))
    Xa, ya = np.asarray(X), np.asarray(y)
    cv = None
    if len(X) >= 15:
        folds = KFold(n_splits=min(5, len(X) // 3), shuffle=True, random_state=0)
        cv = float(np.mean(cross_val_score(model, Xa, ya, cv=folds, scoring="r2")))
    model.fit(Xa, ya)
    importances = dict(zip(FEATURES, (float(v) for v in model.feature_importances_), strict=True))
    weights = importances if sum(importances.values()) > 0 else None
    bundle = {
        "format": MODEL_FORMAT, "features": list(FEATURES), "model": model, "weights": weights,
        "n_samples": len(X), "target": target, "target_transform": "log1p_minus_platform_median",
        "platform_medians": medians, "reference_log_target": sorted(float(v) for v in ya),
        "trained_at": (now or utc_now()).isoformat(), "cv_r2": cv,
    }
    out = path(cfg, "model")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pkl.tmp")
    joblib.dump(bundle, tmp)
    tmp.replace(out)
    meta = {"trained_at": bundle["trained_at"], "n_samples": len(X), "cv_r2": cv, "importances": importances}
    with StateStore(path(cfg, "state")).transaction() as doc:
        doc["model_meta"] = meta
    log.info("model.pkl entraîné sur %d clips (R² CV %s)", len(X), f"{cv:.2f}" if cv is not None else "n/a")
    return meta


def push_model_to_drive(cfg: Mapping[str, Any]) -> bool:
    """Envoie le nouveau modèle dans ClipGod/model.pkl (utilisé automatiquement par le prochain scan)."""
    from core.drive import get_drive
    from modules.prepare import push_model

    return push_model(cfg, get_drive(cfg), StateStore(path(cfg, "state")))


# ===========================================================================
# Rapport
# ===========================================================================
def hook_style(hook: str | None) -> str:
    h = (hook or "").strip()
    if not h:
        return "aucun"
    if h[:1] in "\"«" or re.match(r"^[\w@]+\s*:\s*[\"«]", h):
        return "citation"
    if "?" in h:
        return "question"
    if re.search(r"wait for it|attends|watch till|regarde jusqu", h, re.IGNORECASE):
        return "suspense"
    if re.search(r"nobody|personne|never|jamais|craziest|plus fou|insane", h, re.IGNORECASE):
        return "superlatif"
    return "affirmation"


def _median(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def analyze(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Agrégats du rapport (aussi utilisés par la CLI)."""
    target = str(get(cfg, "learn.target", "views_24h"))
    measured = [r for r in rows if r.get(target) not in (None, "")]
    ranked = sorted(measured, key=lambda r: -float(r[target]))
    tz = ZoneInfo(str(get(cfg, "project.timezone", "Europe/Paris")))

    ab = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in measured:
        groups[(str(r.get("group")), str(r.get("platform")))].append(r)
    for (group, platform), members in groups.items():
        if len({m.get("hook") for m in members}) >= 2:
            members = sorted(members, key=lambda m: -float(m[target]))
            best, worst = members[0], members[-1]
            lift = (float(best[target]) / max(1.0, float(worst[target])) - 1) * 100
            ab.append({"group": group, "platform": platform, "winner": best.get("hook"), "loser": worst.get("hook"),
                       "winner_views": int(float(best[target])), "loser_views": int(float(worst[target])),
                       "lift_pct": round(lift)})
    # A/B inter-plateformes : on compare les variantes au sein de chaque plateforme uniquement.

    def by(key_fn) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in measured:
            buckets[key_fn(r)].append(float(r[target]))
        return sorted(({"key": k, "n": len(v), "median": round(statistics.median(v))} for k, v in buckets.items()),
                      key=lambda d: -d["median"])

    campaigns = {c["slug"]: c for c in (read_json(path(cfg, "campaigns_json"), default={}) or {}).get("campaigns", [])}
    try:
        from modules.monetize import load_manual

        campaigns.update({c.slug: c.__dict__ for c in load_manual(cfg)})
    except Exception as exc:  # noqa: BLE001 — le rapport se fait quand même, sans les CPM manuels
        log.debug("campagnes manuelles illisibles : %s", exc)
    per_campaign: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in measured:
        per_campaign[r.get("campaign") or "—"].append(r)
    money = []
    for camp, items in per_campaign.items():
        cpm = (campaigns.get(camp) or {}).get("cpm_usd")
        views = sum(float(r.get("views_72h") or r[target]) for r in items)
        money.append({"campaign": camp, "clips": len(items), "views": int(views), "cpm": cpm,
                      "earned": round(views * cpm / 1000, 2) if cpm else None,
                      "per_clip": round(views * cpm / 1000 / len(items), 2) if cpm else None})
    money.sort(key=lambda m: -(m["earned"] or 0))
    return {
        "target": target, "n_posts": len(rows), "n_measured": len(measured),
        "total_views": int(sum(float(r[target]) for r in measured)),
        "top": ranked[:5], "flop": ranked[-5:][::-1] if len(ranked) > 5 else [],
        "ab": sorted(ab, key=lambda a: -a["lift_pct"]),
        "hook_styles": by(lambda r: hook_style(r.get("hook"))),
        "platforms": by(lambda r: str(r.get("platform"))),
        "slots": by(lambda r: (_parse(r.get("published_at")) or utc_now()).astimezone(tz).strftime("%Hh")),
        "campaigns": money,
    }


def recommendations(stats: Mapping[str, Any], model_meta: Mapping[str, Any] | None) -> list[str]:
    recs = []
    if stats["n_measured"] < 10:
        recs.append(f"Encore peu de données ({stats['n_measured']} clips mesurés) : publiez régulièrement et lancez "
                    "`clipgod learn` 1 h, 6 h, 24 h et 72 h après les publications.")
    if stats["hook_styles"] and stats["hook_styles"][0]["n"] >= 3:
        best = stats["hook_styles"][0]
        recs.append(f"Les hooks de type « {best['key']} » performent le mieux (médiane {best['median']} vues) : "
                    "privilégiez-les dans core/scoring.py (HOOK_TEMPLATES).")
    if stats["slots"] and len(stats["slots"]) > 1 and stats["slots"][0]["n"] >= 3:
        recs.append(f"Meilleur créneau : {stats['slots'][0]['key']} (médiane {stats['slots'][0]['median']} vues) ; "
                    f"le moins bon : {stats['slots'][-1]['key']}. Ajustez publish.slots.")
    if stats["platforms"] and len(stats["platforms"]) > 1:
        recs.append(f"Plateforme la plus porteuse : {stats['platforms'][0]['key']}.")
    paid = [c for c in stats["campaigns"] if c["per_clip"]]
    if paid:
        recs.append(f"Campagne la plus rentable : {paid[0]['campaign']} ({paid[0]['per_clip']} $/clip). "
                    "Ciblez-la avec clipgod prepare --campaign.")
    wins = [a for a in stats["ab"] if a["lift_pct"] >= 30]
    if wins:
        recs.append(f"{len(wins)} test(s) A/B avec un écart > 30 % : le hook compte autant que le moment.")
    if model_meta and model_meta.get("importances"):
        top = sorted(model_meta["importances"].items(), key=lambda kv: -kv[1])[:3]
        recs.append("Signaux les plus prédictifs : " + ", ".join(f"{k} ({v:.0%})" for k, v in top) + ".")
    if model_meta and model_meta.get("cv_r2") is not None and model_meta["cv_r2"] < 0:
        recs.append("Le modèle ne prédit pas encore mieux que la moyenne (R² < 0) : son poids reste faible dans le "
                    "scan tant que l'historique est court.")
    return recs or ["Rien à signaler : continuez à publier et à mesurer."]


def _table(rows: Sequence[Mapping[str, Any]], cols: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>Pas encore de données.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
    body = "".join("<tr>" + "".join(
        f"<td>{_cell(r.get(k))}</td>" for k, _ in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cell(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, str) and v.startswith("http"):
        return f"<a href='{html.escape(v)}'>lien</a>"
    if isinstance(v, float):
        return f"{v:,.2f}".replace(",", " ")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return html.escape(str(v))


def _bars(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "<p class='muted'>Pas encore de données.</p>"
    top = max(i["median"] for i in items) or 1
    return "".join(f"<div class='bar'><span class='lbl'>{html.escape(str(i['key']))} <small>({i['n']})</small></span>"
                   f"<span class='fill' style='width:{max(2, 100 * i['median'] / top):.0f}%'></span>"
                   f"<span class='val'>{i['median']:,}</span></div>".replace(",", " ") for i in items)


def render_report(cfg: Mapping[str, Any], stats: Mapping[str, Any], model_meta: Mapping[str, Any] | None,
                  out: Path) -> Path:
    t = stats["target"]
    clip_cols = [("hook", "Hook"), ("platform", "Plateforme"), (t, "Vues"), ("views_72h", "Vues 72 h"),
                 ("score", "Score scan"), ("campaign", "Campagne"), ("url", "")]
    model_html = "<p class='muted'>Pas encore de modèle (pondérations par défaut).</p>"
    if model_meta:
        imp = sorted((model_meta.get("importances") or {}).items(), key=lambda kv: -kv[1])
        model_html = (f"<p>Entraîné le {html.escape(str(model_meta.get('trained_at', ''))[:16])} sur "
                      f"<b>{model_meta.get('n_samples')}</b> clips — R² (validation croisée) : "
                      f"<b>{model_meta['cv_r2']:.2f}</b></p>" if model_meta.get("cv_r2") is not None else
                      f"<p>Entraîné sur <b>{model_meta.get('n_samples')}</b> clips.</p>")
        model_html += _bars([{"key": k, "n": "", "median": round(v * 100)} for k, v in imp]).replace("<small>()</small>", "")
    doc = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ClipGod — rapport</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--accent:#ff3b30;--card:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#f2f2f7;--muted:#98989d;--line:#2c2c2e;--card:#1c1c1e}}}}
body{{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1040px;margin:0 auto;padding:24px 16px 64px}} h1{{margin:0 0 4px}} h2{{margin:36px 0 10px;font-size:19px}}
.muted{{color:var(--muted)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border-radius:12px;padding:14px}} .kpi b{{display:block;font-size:26px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:600}} .wrap{{overflow-x:auto}} a{{color:var(--accent)}}
.bar{{display:grid;grid-template-columns:200px 1fr 80px;align-items:center;gap:8px;margin:4px 0}}
.fill{{height:12px;border-radius:6px;background:var(--accent)}} .val{{text-align:right;font-variant-numeric:tabular-nums}}
ul.recs li{{margin:6px 0}}
</style></head><body><main>
<h1>ClipGod — rapport de performance</h1>
<p class="muted">Généré le {utc_now_iso()} · métrique principale : vues à {html.escape(t.removeprefix('views_'))}</p>
<div class="kpis">
<div class="kpi"><span class="muted">Posts suivis</span><b>{stats['n_posts']}</b></div>
<div class="kpi"><span class="muted">Mesurés à {html.escape(t.removeprefix('views_'))}</span><b>{stats['n_measured']}</b></div>
<div class="kpi"><span class="muted">Vues cumulées</span><b>{_cell(stats['total_views'])}</b></div>
<div class="kpi"><span class="muted">Gains estimés</span><b>{sum(c['earned'] or 0 for c in stats['campaigns']):,.2f} $</b></div>
</div>
<h2>Recommandations</h2><ul class="recs">{''.join(f'<li>{html.escape(r)}</li>' for r in recommendations(stats, model_meta))}</ul>
<h2>Top clips</h2><div class="wrap">{_table(stats['top'], clip_cols)}</div>
<h2>Flop clips</h2><div class="wrap">{_table(stats['flop'], clip_cols)}</div>
<h2>Hooks gagnants (tests A/B, 10 plus gros écarts)</h2><div class="wrap">{_table(stats['ab'][:10], [('winner', 'Hook gagnant'), ('winner_views', 'Vues'), ('loser', 'Hook perdant'), ('loser_views', 'Vues'), ('lift_pct', 'Écart %'), ('platform', 'Plateforme')])}</div>
<h2>Styles de hook (médiane des vues)</h2>{_bars(stats['hook_styles'])}
<h2>Créneaux (heure locale)</h2>{_bars(stats['slots'])}
<h2>Plateformes</h2>{_bars(stats['platforms'])}
<h2>Campagnes les plus rentables</h2><div class="wrap">{_table(stats['campaigns'], [('campaign', 'Campagne'), ('clips', 'Clips'), ('views', 'Vues (72 h)'), ('cpm', 'CPM $'), ('earned', 'Gains estimés $'), ('per_clip', '$/clip')])}</div>
<h2>Modèle de viralité</h2>{model_html}
</main></body></html>"""
    atomic_write_text(out, doc)
    return out


# ===========================================================================
# Point d'entrée
# ===========================================================================
@dataclass
class LearnReport:
    collected: CollectReport | None = None
    rows: int = 0
    measured: int = 0
    trained: bool = False
    train_reason: str = ""
    model_meta: dict[str, Any] | None = None
    pushed: bool | None = None
    report_path: Path | None = None


def learn(cfg: Mapping[str, Any], *, collect_views: bool = True, retrain: bool = False, report: bool = True,
          push: bool = True, now: datetime | None = None, youtube_service: Any = None) -> LearnReport:
    now = now or utc_now()
    out = LearnReport()
    if collect_views:
        out.collected = collect(cfg, now=now, youtube_service=youtube_service)
    rows = build_history(cfg)
    write_history(cfg, rows)
    target = str(get(cfg, "learn.target", "views_24h"))
    out.rows, out.measured = len(rows), sum(1 for r in rows if r.get(target) is not None)
    do_train, out.train_reason = should_retrain(cfg, out.measured, now=now)
    if retrain and out.measured >= 2:
        do_train, out.train_reason = True, "--retrain"
    if do_train:
        out.model_meta = train(cfg, rows, now=now)
        out.trained = True
        if push:
            try:
                out.pushed = push_model_to_drive(cfg)
            except Exception as exc:  # noqa: BLE001 — Drive indisponible : le modèle local reste à jour
                log.warning("model.pkl non envoyé sur Drive (%s) : il le sera au prochain clipgod prepare.", exc)
                out.pushed = False
    if report:
        meta = out.model_meta or StateStore(path(cfg, "state")).load().get("model_meta")
        out.report_path = render_report(cfg, analyze(cfg, rows), meta, resolve_path(cfg, get(cfg, "paths.report", "report.html")))
    return out
