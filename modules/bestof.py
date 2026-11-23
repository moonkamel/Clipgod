"""« Top 5 GTA 6 du jour » (laptop) : compilation verticale des meilleurs moments récents -> ``output/bestof/``.

Déroulé de ``clipgod bestof`` :
    1. candidats  clips synchronisés des dernières ``bestof.hours`` heures (``output/clips`` + sidecars) ;
                  langue de la source dans ``discover.languages`` ; cinématiques écartées (``cutscene`` du scan) ;
                  moments déjà utilisés dans un précédent best-of écartés ;
    2. choix      une variante par moment (meilleur hook), au plus ``max_per_streamer`` moments par streamer,
                  les ``count`` meilleurs scores ;
    3. montage    compte à rebours #5 -> #1 : carte titre, puis pour chaque moment une carte « #N @streamer »
                  (fond = le clip flouté) suivie du clip (coupé à ``max_clip_seconds``) avec un badge « #N » ;
                  un seul ffmpeg (filter_complex + ASS), NVENC si dispo, sinon libx264 ;
    4. sortie     ``TOP5_<date>.mp4`` + ``TOP5_<date>.txt`` (légende : classement + crédits + hashtags).

Chaque moment garde ses sous-titres, son hook et le crédit @streamer déjà incrustés au rendu.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.config import get, path, resolve_path
from core.logger import get_logger
from core.storage import StateStore, read_json, utc_now, utc_now_iso

log = get_logger(__name__)

TITLES = {"en": ("TOP {n}", "GTA 6 MOMENTS", "OF THE DAY"), "fr": ("TOP {n}", "GTA 6", "DU JOUR")}
MONTHS = {"en": ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
          "fr": ["JANV.", "FÉVR.", "MARS", "AVR.", "MAI", "JUIN", "JUIL.", "AOÛT", "SEPT.", "OCT.", "NOV.", "DÉC."]}


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


class BestOfError(RuntimeError):
    """Pas assez de moments, ffmpeg en échec…"""


@dataclass
class Pick:
    clip_id: str
    path: Path
    meta: dict[str, Any]
    duration: float
    rank: int = 0                    # 1 = meilleur
    use_seconds: float = 0.0

    @property
    def group(self) -> str:
        return f"{self.meta.get('video_id')}_{self.meta.get('moment_idx', 0)}"

    @property
    def score(self) -> float:
        return float(self.meta.get("score") or 0)

    @property
    def streamer(self) -> str:
        return str(self.meta.get("streamer") or "").lstrip("@")


@dataclass
class BestOf:
    video: Path
    caption_file: Path
    caption: str
    picks: list[Pick] = field(default_factory=list)
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Sélection
# ---------------------------------------------------------------------------
def _moments_info(cfg: Mapping[str, Any]) -> tuple[dict[str, str | None], dict[str, float]]:
    """(video_id -> langue, « video_id_idx » -> part de cinématique) depuis ``output/moments/*.json``."""
    moments_dir = resolve_path(cfg, get(cfg, "sync.moments_dir", "output/moments"))
    langs: dict[str, str | None] = {}
    cutscene: dict[str, float] = {}
    if moments_dir.is_dir():
        for f in moments_dir.glob("*.json"):
            doc = read_json(f, default={}) or {}
            if not isinstance(doc, dict) or not doc.get("video_id"):
                continue
            vid = str(doc["video_id"])
            langs[vid] = doc.get("language")
            for m in [*(doc.get("moments") or []), *(doc.get("candidates") or [])]:
                if m.get("cutscene") is not None:
                    cutscene[f"{vid}_{m.get('idx', 0)}"] = float(m["cutscene"])
    return langs, cutscene


def _recent(meta: Mapping[str, Any], entry: Mapping[str, Any], since: datetime) -> bool:
    """Rendu (ou, à défaut, synchronisé) après ``since`` ; sans date connue, le clip est gardé."""
    value = meta.get("rendered_at") or entry.get("synced_at")
    if not value:
        return True
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) >= since
    except ValueError:
        return True


def select_picks(cfg: Mapping[str, Any], *, now: datetime | None = None, count: int | None = None,
                 hours: float | None = None, allow_repeat: bool | None = None, viral: bool = False) -> list[Pick]:
    """Les meilleurs moments récents, une variante chacun, classés (rang 1 = meilleur score).

    ``viral`` : seulement les moments issus de ``clipgod viral``, classés par vues du Short d'origine.
    """
    from modules.manual import hook_quality

    bcfg = get(cfg, "bestof", {}) or {}
    now = now or utc_now()
    count = int(count or bcfg.get("count", 5))
    since = now - timedelta(hours=float(hours or bcfg.get("hours", 48)))
    allow_repeat = bool(bcfg.get("allow_repeat", False)) if allow_repeat is None else allow_repeat
    max_cut = float(bcfg.get("max_cutscene", 0.5))
    wanted_langs = {str(x).lower() for x in (bcfg.get("languages") or get(cfg, "discover.languages", []) or [])}
    state = StateStore(path(cfg, "state"))
    used = set() if allow_repeat else {g for b in (state.load().get("bestof") or {}).values() for g in b.get("groups", [])}
    langs, cutscene = _moments_info(cfg)

    best: dict[str, Pick] = {}
    for clip_id, entry in state.clips().items():
        if entry.get("status") == "rejected":
            continue
        p = Path(entry.get("path") or "")
        if not p.is_file():
            continue
        meta = read_json(p.with_suffix(".json"), default={}) or {}
        pick = Pick(clip_id, p, meta, float(meta.get("duration") or entry.get("duration") or 0))
        if viral and not meta.get("viral"):
            continue
        if pick.duration <= 3 or pick.group in used or not _recent(meta, entry, since):
            continue
        lang = langs.get(str(meta.get("video_id")))
        if wanted_langs and lang and lang.lower() not in wanted_langs:
            continue
        if cutscene.get(pick.group, 0.0) > max_cut:
            continue
        if hook_quality(meta.get("hook")) <= 0 and meta.get("hook"):
            continue                                        # hook illisible (écriture non latine)
        cur = best.get(pick.group)
        key = (hook_quality(meta.get("hook")), -int(meta.get("variant", 0) or 0))
        if cur is None or key > (hook_quality(cur.meta.get("hook")), -int(cur.meta.get("variant", 0) or 0)):
            best[pick.group] = pick

    per_streamer: dict[str, int] = {}
    chosen: list[Pick] = []
    def order(p: Pick) -> tuple[float, str]:
        return (-float((p.meta.get("viral") or {}).get("views") or 0) if viral else -p.score, p.group)

    for pick in sorted(best.values(), key=order):
        key = pick.streamer.lower()
        if per_streamer.get(key, 0) >= int(bcfg.get("max_per_streamer", 2)):
            continue
        per_streamer[key] = per_streamer.get(key, 0) + 1
        chosen.append(pick)
        if len(chosen) >= count:
            break
    for i, pick in enumerate(chosen, 1):
        pick.rank = i
    return chosen


def plan_durations(picks: Sequence[Pick], *, max_clip: float, target_total: float, intro: float,
                   card: float) -> None:
    """Durée gardée par moment : <= ``max_clip`` et total <= ``target_total`` (on coupe la fin des plus longs)."""
    for p in picks:
        p.use_seconds = min(p.duration, max_clip)
    budget = target_total - intro - card * len(picks)
    while sum(p.use_seconds for p in picks) > budget and max(p.use_seconds for p in picks) > 8:
        longest = max(picks, key=lambda p: p.use_seconds)
        longest.use_seconds = max(8.0, longest.use_seconds - 0.5)
    for p in picks:
        p.use_seconds = round(p.use_seconds, 2)


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------
def build_overlay(picks: Sequence[Pick], *, W: int, H: int, intro: float, card: float, lang: str, date_label: str,
                  font: str) -> str:
    """Script ASS de la compilation : carte titre, cartes « #N », badges pendant chaque moment."""
    from modules.render import _ass_escape, _ass_time

    top, middle, bottom = TITLES.get(lang, TITLES["en"])
    n = len(picks)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Big,{font},{int(W * 0.26)},&H0000E6FF,&H0000E6FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,10,4,5,40,40,0,1
Style: Title,{font},{int(W * 0.105)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,40,40,0,1
Style: Small,{font},{int(W * 0.055)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,40,40,0,1
Style: Badge,{font},{int(W * 0.075)},&H00000000,&H00000000,&H0000E6FF,&H00000000,-1,0,0,0,100,100,0,0,3,18,0,7,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    pop = r"\fscx40\fscy40\t(0,180,\fscx110\fscy110)\t(180,280,\fscx100\fscy100)"
    ev = [
        (f"Dialogue: 2,{_ass_time(0)},{_ass_time(intro)},Big,,0,0,0,,{{\\an5\\pos({W // 2},{int(H * 0.36)}){pop}}}"
        f"{_ass_escape(top.format(n=n))}"),
        (f"Dialogue: 2,{_ass_time(0.15)},{_ass_time(intro)},Title,,0,0,0,,{{\\an5\\pos({W // 2},{int(H * 0.5)})"
        f"\\fad(120,0)}}{_ass_escape(middle)}"),
        (f"Dialogue: 2,{_ass_time(0.3)},{_ass_time(intro)},Title,,0,0,0,,{{\\an5\\pos({W // 2},{int(H * 0.58)})"
        f"\\fad(120,0)}}{_ass_escape(bottom)}"),
        (f"Dialogue: 2,{_ass_time(0.45)},{_ass_time(intro)},Small,,0,0,0,,{{\\an5\\pos({W // 2},{int(H * 0.68)})"
        f"\\fad(120,0)}}{_ass_escape(date_label)}"),
    ]
    t = intro
    for p in sorted(picks, key=lambda p: -p.rank):          # compte à rebours : #5 … #1
        label = f"#{p.rank}"
        ev.append(f"Dialogue: 2,{_ass_time(t)},{_ass_time(t + card)},Big,,0,0,0,,"
                  f"{{\\an5\\pos({W // 2},{int(H * 0.44)}){pop}}}{label}")
        if p.streamer:
            ev.append(f"Dialogue: 2,{_ass_time(t + 0.1)},{_ass_time(t + card)},Small,,0,0,0,,"
                      f"{{\\an5\\pos({W // 2},{int(H * 0.56)})\\fad(100,0)}}@{_ass_escape(p.streamer)}")
        t += card
        ev.append(f"Dialogue: 1,{_ass_time(t)},{_ass_time(t + p.use_seconds)},Badge,,0,0,0,,"
                  f"{{\\an7\\pos({int(W * 0.05)},{int(H * 0.035)})}}{label}")
        t += p.use_seconds
    return header + "\n".join(ev) + "\n"


def build_filter(picks: Sequence[Pick], *, W: int, H: int, fps: int, intro: float, card: float,
                 ass_file: Path, fonts_dir: Path | None) -> str:
    """Graphe ffmpeg : entrée k = clip du rang k (ordre de lecture) ; la dernière entrée sert à la carte titre."""
    from modules.render import _ass_path

    norm_v = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={fps},format=yuv420p"
    norm_a = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    blur = "boxblur=24:2,eq=brightness=-0.28:saturation=1.2"
    order = sorted(picks, key=lambda p: -p.rank)
    parts: list[str] = []
    labels: list[str] = []
    title_in = len(order)
    parts.append(f"[{title_in}:v]trim=0:{intro},setpts=PTS-STARTPTS,{norm_v},{blur}[tv]")
    parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{intro},{norm_a}[ta]")
    labels.append("[tv][ta]")
    for k, p in enumerate(order):
        parts.append(f"[{k}:v]{norm_v},split[cv{k}][mv{k}]")
        parts.append(f"[cv{k}]trim=0:{card},setpts=PTS-STARTPTS,{blur}[c{k}]")
        parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{card},{norm_a}[ca{k}]")
        parts.append(f"[mv{k}]trim=0:{p.use_seconds},setpts=PTS-STARTPTS[m{k}]")
        audio_in = f"[{k}:a]" if p.meta.get("has_audio", True) else "anullsrc=r=48000:cl=stereo,"
        if p.meta.get("has_audio", True) and p.meta.get("channels"):
            # Réétiquetage en stéréo canal par canal, sans le convertisseur de ffmpeg : il refuse certaines
            # étiquettes exotiques (« 2 channels (LFE+LFE2) » vu sur des vidéos TikTok). Mono -> dupliqué.
            mapping = "0|1" if int(p.meta["channels"]) >= 2 else "0|0"
            audio_in += f"channelmap=map={mapping}:channel_layout=stereo,"
        parts.append(f"{audio_in}atrim=0:{p.use_seconds},asetpts=PTS-STARTPTS,{norm_a},"
                     f"afade=t=out:st={max(0.0, p.use_seconds - 0.25)}:d=0.25[ma{k}]")
        labels.append(f"[c{k}][ca{k}][m{k}][ma{k}]")
    fonts = f":fontsdir={_ass_path(fonts_dir)}" if fonts_dir else ""
    parts.append(f"{''.join(labels)}concat=n={1 + 2 * len(order)}:v=1:a=1[cat][aout]")
    parts.append(f"[cat]ass=filename={_ass_path(ass_file)}{fonts}[vout]")
    return ";".join(parts)


def build_caption(picks: Sequence[Pick], cfg: Mapping[str, Any], *, lang: str, date_label: str) -> str:
    top, middle, bottom = TITLES.get(lang, TITLES["en"])
    title = f"{top.format(n=len(picks))} {middle} {bottom}".replace("  ", " ").title().replace("Gta", "GTA")
    lines = [f"{title} 🔥 ({date_label})", ""]
    for p in sorted(picks, key=lambda p: -p.rank):
        hook = str(p.meta.get("hook") or "").strip()
        who = f" @{p.streamer}" if p.streamer else ""
        lines.append(f"#{p.rank}{who}" + (f" — {hook}" if hook else ""))
    credits = " ".join(dict.fromkeys(f"@{p.streamer}" for p in picks if p.streamer))
    if not credits:
        credits = "@... (ajoute ici les créateurs des vidéos)"
    hashtags = get(cfg, "bestof.hashtags", None) or ["#gta6", "#gtavi", "#gta", "#gaming", "#top5", "#fyp"]
    lines += ["", f"🎮 Credits: {credits}", "", " ".join(hashtags)]
    return "\n".join(line for line in lines if line is not None).replace("\n\n\n", "\n\n").strip()


def picks_from_folder(folder: str | Path, *, limit: int = 10) -> list[Pick]:
    """Vidéos d'un dossier, dans l'ordre des noms de fichiers (1er fichier = n°1 du classement).

    Un ``@pseudo`` dans le nom du fichier sert de crédit (« 1 - @KaiCenat crash.mp4 »).
    """
    import json
    import re
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise BestOfError("ffprobe introuvable (il est installé avec ffmpeg).")
    root = Path(folder)
    if not root.is_dir():
        raise BestOfError(f"Dossier introuvable : {root}")

    def natural(p: Path) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    files = sorted((f for f in root.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS), key=natural)
    picks: list[Pick] = []
    for f in files[:limit]:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,channels",
                              "-of", "json", str(f)], capture_output=True, text=True, check=False)
        try:
            info = json.loads(out.stdout or "{}")
            duration = float(info["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            log.warning("Vidéo illisible ignorée : %s", f.name)
            continue
        kinds = {st.get("codec_type") for st in info.get("streams") or []}
        channels = next((int(st.get("channels") or 0) for st in info.get("streams") or []
                         if st.get("codec_type") == "audio"), 0)
        if "video" not in kinds or duration < 1:
            continue
        handle = re.search(r"@([A-Za-z0-9_.]{2,30})", f.stem)
        meta = {"video_id": f"folder_{re.sub(r'[^A-Za-z0-9_]', '_', f.stem)[:60]}", "moment_idx": 0,
                "streamer": handle.group(1).rstrip(".") if handle else "", "hook": None,
                "has_audio": "audio" in kinds, "channels": channels, "source_file": f.name}
        picks.append(Pick(f.stem, f, meta, duration, rank=len(picks) + 1))
    return picks


def make_bestof(cfg: Mapping[str, Any], *, now: datetime | None = None, count: int | None = None,
                hours: float | None = None, allow_repeat: bool | None = None, output_dir: str | Path | None = None,
                record: bool = True, viral: bool = False, folder: str | Path | None = None,
                archive: bool = True) -> BestOf:
    """Sélectionne, monte et enregistre le best-of du jour. Lève :class:`BestOfError` s'il manque des moments.

    ``folder`` : compile les vidéos de ce dossier (ordre des noms = classement) au lieu des clips ClipGod ;
    ``archive`` : les déplace ensuite dans ``<folder>/publies/<date>/`` pour que le dossier soit prêt le lendemain.
    """
    from modules.render import _ffmpeg, _run, audio_args, encoder_args, ensure_font

    bcfg = get(cfg, "bestof", {}) or {}
    rcfg: Mapping[str, Any] = cfg.get("render") or {}
    now = now or utc_now()
    if folder is not None:
        picks = picks_from_folder(resolve_path(cfg, folder), limit=int(count or bcfg.get("folder_max_videos", 10)))
        minimum = 2
        if len(picks) < minimum:
            raise BestOfError(f"{len(picks)} vidéo(s) dans {folder} : il en faut au moins 2 (.mp4, .mov, .webm…).")
    else:
        picks = select_picks(cfg, now=now, count=count, hours=hours, allow_repeat=allow_repeat, viral=viral)
        minimum = int(bcfg.get("min_count", 3))
    if len(picks) < minimum:
        hint = ("Lancez clipgod viral, puis Colab (Tout exécuter), puis clipgod sync." if viral else
                "Traitez d'autres VODs (clipgod discover --trending --prepare 2) puis clipgod sync.")
        raise BestOfError(f"Seulement {len(picks)} moment(s) récent(s) utilisable(s) (minimum {minimum}). {hint}")
    intro, card = float(bcfg.get("intro_seconds", 1.6)), float(bcfg.get("card_seconds", 0.9))
    if folder is not None:
        plan_durations(picks, max_clip=float(bcfg.get("folder_max_clip_seconds", 30)),
                       target_total=float(bcfg.get("folder_target_seconds", 170)), intro=intro, card=card)
    else:
        plan_durations(picks, max_clip=float(bcfg.get("max_clip_seconds", 18)),
                       target_total=float(bcfg.get("target_seconds", 95)), intro=intro, card=card)

    tz = ZoneInfo(str(get(cfg, "project.timezone", "Europe/Paris")))
    local = now.astimezone(tz)
    lang = str(bcfg.get("language") or get(cfg, "scan.hook_language", "en") or "en")
    month = MONTHS.get(lang, MONTHS["en"])[local.month - 1]
    date_label = f"{local.day} {month} {local.year}"
    W, H, fps = int(rcfg.get("width", 1080)), int(rcfg.get("height", 1920)), int(rcfg.get("fps", 30))
    font = str((rcfg.get("subtitles") or {}).get("font_name") or "Montserrat ExtraBold")

    out_dir = resolve_path(cfg, output_dir or bcfg.get("output_dir") or "output/bestof")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"TOP{len(picks)}_{local.date().isoformat()}"
    video = out_dir / f"{stem}.mp4"
    order = sorted(picks, key=lambda p: -p.rank)
    with tempfile.TemporaryDirectory(prefix="clipgod_bestof_") as tmpd:
        ass = Path(tmpd) / "overlay.ass"
        ass.write_text(build_overlay(picks, W=W, H=H, intro=intro, card=card, lang=lang, date_label=date_label,
                                     font=font), encoding="utf-8")
        graph = build_filter(picks, W=W, H=H, fps=fps, intro=intro, card=card, ass_file=ass,
                             fonts_dir=ensure_font(rcfg))
        inputs: list[str] = []
        for p in [*order, picks[0]]:                      # dernière entrée : le n°1, flouté sous la carte titre
            inputs += ["-i", str(p.path)]
        cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *inputs,
               "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]", *encoder_args(rcfg),
               *audio_args(rcfg), "-movflags", "+faststart", str(video)]
        log.info("Montage du best-of : %d moments, ~%.0f s", len(picks),
                 intro + sum(card + p.use_seconds for p in picks))
        try:
            _run(cmd, "best-of")
        except Exception as exc:
            raise BestOfError(str(exc)) from exc

    caption = build_caption(picks, cfg, lang=lang, date_label=date_label)
    caption_file = out_dir / f"{stem}.txt"
    caption_file.write_text(caption + "\n", encoding="utf-8-sig")
    total = round(intro + sum(card + p.use_seconds for p in picks), 2)
    if record:
        state = StateStore(path(cfg, "state"))
        with state.transaction() as data:
            data.setdefault("bestof", {})[stem] = {
                "file": str(video), "groups": [p.group for p in picks], "clips": [p.clip_id for p in picks],
                "duration": total, "created_at": utc_now_iso()}
    if folder is not None and archive:
        done = resolve_path(cfg, folder) / "publies" / local.date().isoformat()
        done.mkdir(parents=True, exist_ok=True)
        for p in picks:
            target = done / p.path.name
            shutil.move(str(p.path), str(target))
            p.path = target
    log.info("✔ best-of %s (%.0f s)", video, total)
    return BestOf(video, caption_file, caption, picks, total)
