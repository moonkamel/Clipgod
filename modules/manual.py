"""Publication manuelle assistée (laptop) : TikTok / Instagram sans API payante.

``clipgod tiktok`` prépare chaque post et l'utilisateur le dépose lui-même dans TikTok Studio
(https://www.tiktok.com/tiktokstudio/upload, programmation jusqu'à 10 jours à l'avance) :

    1. plan      mêmes candidats que ``publish`` (validation monetize, règles de campagne), meilleurs moments
                 d'abord, créneaux de forte audience, ``publish.manual.max_per_day`` posts par jour ;
    2. post      la légende (hook, crédit du streamer, hashtags) est copiée dans le presse-papiers et le clip est
                 sélectionné dans l'explorateur de fichiers ;
    3. état      une fois confirmé, le post est enregistré dans ``state["posts"]`` (backend ``manual``) : il n'est
                 jamais reproposé, et son lien éventuel sert à ``learn`` (via ``data/manual_views.csv``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.config import get, path, resolve_path
from core.logger import get_logger
from core.scoring import is_latin_text
from core.storage import StateStore, read_json, utc_now, utc_now_iso
from modules.publish import (
    PUBLISHED_STATES,
    ClipInfo,
    _parse,
    next_slots,
    plan_posts,
    slot_times,
    validated_candidates,
)

log = get_logger(__name__)

STUDIO_URLS = {"tiktok": "https://www.tiktok.com/tiktokstudio/upload",
               "instagram_reels": "https://www.instagram.com/"}
DEFAULT_HASHTAGS = {"tiktok": ["#gta6", "#gtavi", "#gta", "#gaming", "#fyp"],
                    "instagram_reels": ["#gta6", "#gtavi", "#gta", "#gaming", "#reels"]}


@dataclass
class ManualPost:
    clip: ClipInfo
    platform: str
    slot: datetime | None
    caption: str

    @property
    def post_id(self) -> str:
        return f"{self.clip.clip_id}@{self.platform}"


def build_caption(clip: ClipInfo, cfg: Mapping[str, Any], platform: str = "tiktok") -> str:
    """Légende prête à coller : hook, crédit du streamer, mentions et hashtags (campagne d'abord)."""
    mcfg = get(cfg, "publish.manual", {}) or {}
    required = list(clip.rules.required_hashtags) if clip.rules else []
    mentions = list(clip.rules.required_mentions) if clip.rules else []
    base = (mcfg.get("hashtags") or {}).get(platform) if isinstance(mcfg.get("hashtags"), Mapping) else None
    hashtags = list(dict.fromkeys([*required, *(base or DEFAULT_HASHTAGS.get(platform, DEFAULT_HASHTAGS["tiktok"]))]))
    hook = re.sub(r"\s+", " ", str(clip.meta.get("hook") or "GTA 6 moment")).strip()
    streamer = str(clip.meta.get("streamer") or "").lstrip("@")
    template = str(get(cfg, "publish.credit_template", "🎮 Credit: @{streamer}"))
    credit = template.format(streamer=streamer) if streamer else ""
    lines = [hook, "", credit, " ".join(mentions), "", " ".join(hashtags)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:2200]                                             # limite TikTok


def _manual_posts(state: StateStore, platform: str) -> dict[str, dict[str, Any]]:
    return {pid: p for pid, p in state.load().get("posts", {}).items() if p.get("platform") == platform}


def plan_manual(cfg: Mapping[str, Any], *, platform: str = "tiktok", limit: int | None = None,
                schedule: bool = True, now: datetime | None = None,
                rejected: list[tuple[str, list[str]]] | None = None, one_per_moment: bool = True,
                only: Sequence[str] | None = None) -> list[ManualPost]:
    """Prochains posts à déposer à la main sur ``platform`` (un clip déjà posté n'est jamais reproposé).

    ``one_per_moment`` : une seule variante de hook par moment (la 1re du plan) ;
    ``only`` : clips choisis par l'utilisateur (ids), publiés dans cet ordre.
    """
    mcfg = get(cfg, "publish.manual", {}) or {}
    state = StateStore(path(cfg, "state"))
    tz = ZoneInfo(str(get(cfg, "project.timezone", "Europe/Paris")))
    now = now or utc_now()
    # Un clip déjà posté ailleurs (YouTube…) reste candidat pour cette plateforme.
    candidates = validated_candidates(cfg, state, rejected=rejected,
                                      statuses=("synced", "validated", "scheduled", "published"))
    posts = _manual_posts(state, platform)
    already = {pid for pid, p in state.load().get("posts", {}).items() if p.get("status") in (*PUBLISHED_STATES,
                                                                                               "private")}
    # Sur un compte manuel, une seule variante A/B par moment (la meilleure) : pas de doublons visibles.
    seen_groups = {p.get("group") for p in posts.values()}
    plan = [(c, p) for c, p in plan_posts(candidates, [platform], already=already) if c.group not in seen_groups]
    if only is not None:
        by_id = {c.clip_id: (c, p) for c, p in plan}
        plan = [by_id[cid] for cid in dict.fromkeys(only) if cid in by_id]
    elif one_per_moment:
        best: dict[str, tuple[ClipInfo, str]] = {}
        for c, p in plan:
            best.setdefault(c.group, (c, p))
        plan = list(best.values())
    else:                                                          # liste complète : meilleurs moments d'abord
        plan.sort(key=lambda cp: (-cp[0].score, cp[0].group, cp[0].variant))
    if limit is not None:
        plan = plan[: max(0, limit)]
    if schedule:
        taken = [t for p in posts.values() if p.get("slot") and (t := _parse(p["slot"])) and t > now]
        slots: Sequence[datetime | None] = next_slots(
            len(plan), now=now, tz=tz, times=slot_times(get(cfg, "publish", {}) or {}),
            max_per_day=int(mcfg.get("max_per_day", 3)), taken=taken,
            horizon_days=int(mcfg.get("horizon_days", 10)))
        plan = plan[: len(slots)]
    else:
        slots = [None] * len(plan)
    return [ManualPost(c, p, s, build_caption(c, cfg, p)) for (c, p), s in zip(plan, slots, strict=False)]


def record_manual(cfg: Mapping[str, Any], post: ManualPost, *, url: str | None = None) -> dict[str, Any]:
    """Enregistre un post déposé à la main (statut ``scheduled`` si créneau futur, sinon ``published``)."""
    state = StateStore(path(cfg, "state"))
    status = "scheduled" if post.slot else "published"
    entry = {"clip_id": post.clip.clip_id, "platform": post.platform, "backend": "manual",
             "slot": post.slot.isoformat() if post.slot else None, "status": status, "remote_id": None,
             "url": url or None, "privacy": "public", "test": False, "title": post.caption.split("\n", 1)[0],
             "hook": post.clip.meta.get("hook"), "variant": post.clip.variant, "group": post.clip.group,
             "campaign": post.clip.meta.get("campaign"), "posted_at": utc_now_iso()}
    with state.transaction() as data:
        data.setdefault("posts", {})[post.post_id] = entry
    state.upsert_clip(post.clip.clip_id, status=status, last_post=post.post_id)
    log.info("✔ %s -> %s (manuel, %s)", post.clip.clip_id, post.platform, url or status)
    return entry


# ---------------------------------------------------------------------------
# Export : un dossier prêt à publier (meilleure variante par moment + légendes)
# ---------------------------------------------------------------------------
GENERIC_HOOKS = ("the loudest moment of the stream", "watch till the end", "gta 6 is something else", "wait for it",
                 "the craziest gta 6 moment yet")


def hook_quality(hook: str | None) -> float:
    """Note d'un hook pour choisir la variante à publier : précis > citation > générique ; non latin = 0."""
    text = re.sub(r"\s+", " ", str(hook or "")).strip()
    if not text or not is_latin_text(text):
        return 0.0
    plain = re.sub(r"[^\w\s']", "", text.lower()).strip()
    if any(plain.startswith(g) for g in GENERIC_HOOKS):
        return 1.0
    if '"' in text or "“" in text:                              # citation du transcript : souvent un fragment
        return 1.5
    return 2.0


def safe_title(hook: str | None, max_len: int = 60) -> str:
    """Hook -> nom de fichier Windows valide (sans emojis, guillemets ni caractères interdits)."""
    text = "".join(c for c in str(hook or "") if c.isalnum() or c in " '-,!?.&")
    text = re.sub(r"[?]", "", text)                                # « ? » interdit dans un nom Windows
    text = re.sub(r"\s+", " ", text).strip(" .-,'")
    return (text[:max_len].rstrip(" .-,'") or "clip")


def _source_languages(cfg: Mapping[str, Any]) -> dict[str, str | None]:
    """video_id -> langue détectée par Whisper (``output/moments/<video_id>.json``)."""
    moments_dir = resolve_path(cfg, get(cfg, "sync.moments_dir", "output/moments"))
    out: dict[str, str | None] = {}
    if moments_dir.is_dir():
        for f in moments_dir.glob("*.json"):
            doc = read_json(f, default={}) or {}
            if isinstance(doc, dict) and doc.get("video_id"):
                out[str(doc["video_id"])] = doc.get("language")
    return out


@dataclass
class ExportReport:
    folder: Path
    files: list[tuple[Path, ManualPost]]
    skipped_language: list[str]


def export_ready(cfg: Mapping[str, Any], *, platform: str = "tiktok", limit: int | None = None,
                 folder: str | Path | None = None, schedule: bool = True, now: datetime | None = None,
                 rejected: list[tuple[str, list[str]]] | None = None) -> ExportReport:
    """Copie la meilleure variante de chaque moment dans un dossier « prêt à publier » + ``legendes.txt``.

    - une seule vidéo par moment (le meilleur hook, cf. ``hook_quality``) : aucun doublon ;
    - sources dans une langue hors ``publish.manual.languages`` (défaut : ``discover.languages``) écartées ;
    - fichiers nommés ``01 - <hook>.mp4`` (meilleur score d'abord) ; le dossier est vidé à chaque export ;
    - les moments exportés sont notés (statut ``exported``) et ne seront plus jamais réexportés.
    """
    mcfg = get(cfg, "publish.manual", {}) or {}
    tz = ZoneInfo(str(get(cfg, "project.timezone", "Europe/Paris")))
    langs = {str(x).lower() for x in (mcfg.get("languages") or get(cfg, "discover.languages", []) or [])}
    src_lang = _source_languages(cfg)
    listing = plan_manual(cfg, platform=platform, schedule=False, now=now, rejected=rejected, one_per_moment=False)
    best: dict[str, ClipInfo] = {}
    skipped_language: list[str] = []
    for post in listing:
        c = post.clip
        lang = src_lang.get(str(c.meta.get("video_id")))
        if langs and lang and lang.lower() not in langs:
            if c.group not in skipped_language:
                skipped_language.append(c.group)
            continue
        cur = best.get(c.group)
        key = (hook_quality(c.meta.get("hook")), -c.variant)
        if cur is None or key > (hook_quality(cur.meta.get("hook")), -cur.variant):
            best[c.group] = c
    chosen = sorted(best.values(), key=lambda c: (-c.score, c.group))
    chosen = [c for c in chosen if hook_quality(c.meta.get("hook")) > 0]      # hook illisible : on écarte
    if limit is not None:
        chosen = chosen[: max(0, limit)]
    posts = plan_manual(cfg, platform=platform, schedule=schedule, now=now, only=[c.clip_id for c in chosen])

    out = resolve_path(cfg, folder or mcfg.get("export_dir") or "output/a_publier")
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("*.mp4"), *out.glob("*.txt")]:
        old.unlink()
    files: list[tuple[Path, ManualPost]] = []
    lines: list[str] = []
    state = StateStore(path(cfg, "state"))
    for i, post in enumerate(posts, 1):
        name = f"{i:02d} - {safe_title(post.clip.meta.get('hook'))}.mp4"
        target = out / name
        shutil.copy2(post.clip.path, target)
        files.append((target, post))
        when = post.slot.astimezone(tz).strftime("%d/%m à %H:%M") if post.slot else "dès que possible"
        lines += [f"===== {name} =====", f"Programmer : {when}", "", post.caption, "", ""]
        with state.transaction() as data:
            data.setdefault("posts", {})[post.post_id] = {
                "clip_id": post.clip.clip_id, "platform": platform, "backend": "manual", "status": "exported",
                "slot": post.slot.isoformat() if post.slot else None, "file": str(target),
                "hook": post.clip.meta.get("hook"), "variant": post.clip.variant, "group": post.clip.group,
                "campaign": post.clip.meta.get("campaign"), "exported_at": utc_now_iso()}
    if files:
        # UTF-8 avec BOM : le Bloc-notes affiche correctement les emojis.
        (out / "legendes.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")
    log.info("%d clip(s) exporté(s) dans %s", len(files), out)
    return ExportReport(out, files, skipped_language)


def open_folder(folder: Path) -> bool:
    try:
        if sys.platform == "win32":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(folder)])
        else:
            return False
        return True
    except OSError as exc:
        log.warning("Impossible d'ouvrir le dossier : %s", exc)
        return False


# ---------------------------------------------------------------------------
# Bureau : presse-papiers et explorateur de fichiers (best effort, jamais bloquant)
# ---------------------------------------------------------------------------
def copy_to_clipboard(text: str) -> bool:
    """Copie ``text`` (emojis compris) dans le presse-papiers ; False si aucun outil n'est disponible."""
    try:
        if sys.platform == "win32":
            # clip.exe lit l'UTF-16 avec BOM : c'est la seule façon fiable de garder les emojis.
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True)
            return True
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd, input=text.encode("utf-8"), check=True)
                return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Presse-papiers indisponible : %s", exc)
    return False


def play_file(file: Path) -> bool:
    """Ouvre le clip dans le lecteur vidéo par défaut."""
    try:
        if sys.platform == "win32":
            os.startfile(str(file))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(file)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(file)])
        else:
            return False
        return True
    except OSError as exc:
        log.warning("Impossible d'ouvrir le clip : %s", exc)
        return False


def parse_selection(text: str, n: int) -> list[int]:
    """« 1,3 5-7 » -> [1, 3, 5, 6, 7] (numéros 1..n, doublons et hors bornes ignorés)."""
    out: list[int] = []
    for part in re.split(r"[,\s;]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"sélection invalide : {part!r}")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.extend(i for i in range(min(a, b), max(a, b) + 1) if 1 <= i <= n)
    return list(dict.fromkeys(out))


def reveal_file(file: Path) -> bool:
    """Ouvre l'explorateur sur le dossier du clip, fichier sélectionné (glisser-déposer direct dans TikTok)."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{file.resolve()}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(file)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(file.parent)])
        else:
            return False
        return True
    except OSError as exc:
        log.warning("Impossible d'ouvrir l'explorateur : %s", exc)
        return False
