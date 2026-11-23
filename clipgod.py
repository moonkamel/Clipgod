"""ClipGod — CLI d'orchestration (laptop).

Ce fichier ne contient AUCUNE logique métier lourde : il parse les arguments,
charge la configuration et délègue aux modules (`modules/*`).

Les modules lourds `scan` et `render` sont exécutés sur GPU cloud (Colab/Kaggle)
via les notebooks ; les commandes `scan` / `render` ci-dessous existent pour le
debug local sur de petits extraits ou une machine équipée d'un GPU.

Chaque commande retourne le code 0 en cas de succès, 1 en cas d'erreur (message lisible, détails dans logs/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from core.config import ConfigError, load_config
from core.drive import DriveError
from core.logger import setup_logging
from core.storage import ensure_project_dirs

app = typer.Typer(
    name="clipgod",
    help="ClipGod — streams/VODs GTA 6 -> clips verticaux monétisables.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ---------------------------------------------------------------------------
# Options globales
# ---------------------------------------------------------------------------
@app.callback()
def _global_options(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Chemin vers config.yaml (défaut : $CLIPGOD_CONFIG, ./config.yaml)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Logs DEBUG en console."),
    dry_run: bool = typer.Option(False, "--dry-run", help="N'écrit rien / ne publie rien."),
) -> None:
    """Charge la config, initialise les logs et les dossiers locaux (stockés dans ctx.obj)."""
    if ctx.resilient_parsing:  # complétion shell : ne rien charger
        return
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        console.print(f"[red]Erreur de configuration :[/] {exc}")
        raise typer.Exit(1) from None
    setup_logging(cfg, level="DEBUG" if verbose else None)
    ensure_project_dirs(cfg)
    ctx.obj = {"config": cfg, "verbose": verbose, "dry_run": dry_run}


# ---------------------------------------------------------------------------
# Module 1 — discover (laptop)
# ---------------------------------------------------------------------------
@app.command()
def discover(
    ctx: typer.Context,
    platform: list[str] = typer.Option(
        None, "--platform", "-p", help="youtube | twitch | kick (répétable). Défaut : config."
    ),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Nombre max de sources par plateforme."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore le cache local de 5 min."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Défaut : data/sources.json."),
    show: int = typer.Option(15, "--show", help="Nombre de lignes affichées."),
    trending: bool = typer.Option(False, "--trending", "-t", help="Ce qui buzze : Shorts GTA 6 viraux des "
                                  "dernières 24 h -> leurs VODs d'origine (YouTube)."),
    prepare_top: int = typer.Option(0, "--prepare", help="Prépare aussitôt les N meilleures VODs (jobs Drive, "
                                    "téléchargement par le notebook). Idéal pour la tâche quotidienne."),
) -> None:
    """Trouve les streams live / VODs < 24h (> 45 min) GTA 6 -> data/sources.json."""
    from rich.table import Table

    from core.quota import YouTubeQuota
    from modules.discover import DiscoverError
    from modules.discover import discover as run_discover

    cfg = ctx.obj["config"]
    if trending:
        _discover_trending(ctx, show=show, prepare_top=prepare_top)
        return
    try:
        sources = run_discover(cfg, platforms=platform or None, limit=limit, use_cache=not no_cache, output=output)
    except DiscoverError as exc:
        console.print(f"[red]discover :[/] {exc}")
        raise typer.Exit(1) from None
    table = Table(title=f"Sources GTA 6 ({len(sources)})")
    for col in ("#", "plateforme", "statut", "streamer", "titre", "viewers", "durée", "vues/h", "campagne"):
        table.add_column(col, overflow="fold")
    for i, s in enumerate(sources[:show], 1):
        h, m = divmod(int(s["duration"]) // 60, 60)
        table.add_row(str(i), s["platform"], "🔴 live" if s["status"] == "live" else "vod", s["streamer"],
                      s["title"][:60], f"{s['viewers']:,}".replace(",", " "), f"{h}h{m:02d}",
                      f"{s['views_per_hour']:,.0f}".replace(",", " "), s.get("campaign") or "")
    console.print(table)
    console.print(f"[dim]Quota YouTube restant aujourd'hui : {YouTubeQuota.from_config(cfg).remaining()} u[/]")
    if sources:
        best = next((s for s in sources if s["status"] == "vod"), None)
        if best:
            console.print(f"→ Clipper la meilleure VOD : [bold]clipgod run --source {best['url']}[/]")


def _discover_trending(ctx: typer.Context, *, show: int, prepare_top: int) -> None:
    from rich.table import Table

    from core.config import ConfigError
    from core.quota import YouTubeQuota
    from modules.discover import discover_trending

    cfg = ctx.obj["config"]
    try:
        doc = discover_trending(cfg)
    except ConfigError as exc:
        console.print(f"[red]discover --trending :[/] {exc}")
        raise typer.Exit(1) from None
    viral = Table(title=f"Ce qui buzze sur GTA 6 ({len(doc['viral'])} Shorts, {int(doc['hours'])} h)")
    for col in ("#", "chaîne", "Short", "vues", "vues/h"):
        viral.add_column(col, overflow="fold")
    for i, v in enumerate(doc["viral"][:min(show, 10)], 1):
        viral.add_row(str(i), v["channel"], v["title"][:60], f"{v['views']:,}".replace(",", " "),
                      f"{v['views_per_hour']:,.0f}".replace(",", " "))
    console.print(viral)
    sources = doc["sources"]
    table = Table(title=f"VODs d'origine à clipper ({len(sources)})")
    for col in ("#", "streamer", "titre", "durée", "vues/h", "pourquoi"):
        table.add_column(col, overflow="fold")
    for i, s in enumerate(sources[:show], 1):
        h, m = divmod(int(s["duration"]) // 60, 60)
        table.add_row(str(i), s["streamer"], s["title"][:50], f"{h}h{m:02d}",
                      f"{s['views_per_hour'] + s.get('boost', 0):,.0f}".replace(",", " "), s.get("reason", ""))
    console.print(table)
    console.print(f"[dim]Quota YouTube restant aujourd'hui : {YouTubeQuota.from_config(cfg).remaining()} u · "
                  "les Shorts des autres servent de signal, on ne les republie pas.[/]")
    if not sources:
        console.print("[yellow]Aucune VOD d'origine exploitable aujourd'hui.[/]")
        return
    if not prepare_top:
        console.print("→ Préparer les 2 meilleures : [bold]clipgod discover --trending --prepare 2[/]")
        return
    created = 0
    for s in sources[:prepare_top]:
        console.print(f"\n[bold]Préparation[/] {s['streamer']} — {s['title'][:60]}")
        try:
            job = _prepare_job(ctx, s["url"], streamer=s["streamer"], download="cloud")
        except typer.Exit:
            continue
        created += 0 if job.get("_existing") else 1
        _print_next_steps(cfg, job)
    console.print(f"\n[bold green]{created} nouveau(x) job(s).[/] Ouvre Colab : Exécution › Tout exécuter.")


# ---------------------------------------------------------------------------
# Lien laptop -> Drive
# ---------------------------------------------------------------------------
@app.command()
def prepare(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", "-s", help="URL de VOD/stream OU chemin local .mp4."),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Forcer un id de job."),
    streamer: Optional[str] = typer.Option(None, "--streamer", help="Pseudo pour l'overlay @streamer."),
    campaign: Optional[str] = typer.Option(None, "--campaign", help="Campagne cible (monetize)."),
    voiceover: bool = typer.Option(False, "--voiceover", help="Active la voix off Piper au rendu."),
    download: Optional[str] = typer.Option(
        None, "--download", help="auto | local (laptop) | cloud (notebook). Défaut : config prepare.download."
    ),
    force: bool = typer.Option(False, "--force", help="Crée un job même si la vidéo a déjà été envoyée."),
) -> None:
    """Dépose la source dans ClipGod/sources/ (ou la confie au notebook) et crée ClipGod/jobs/<job_id>.json."""
    job = _prepare_job(ctx, source, job_id=job_id, streamer=streamer, campaign=campaign, voiceover=voiceover,
                       download=download, force=force)
    _print_next_steps(ctx.obj["config"], job)


def _prepare_job(ctx: typer.Context, source: str, **kwargs) -> dict:
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    from core.drive import get_drive
    from modules.prepare import PrepareError
    from modules.prepare import prepare as run_prepare

    cfg = ctx.obj["config"]
    if kwargs.get("download") not in (None, "auto", "local", "cloud"):
        console.print("[red]--download doit valoir auto, local ou cloud.[/]")
        raise typer.Exit(1)
    if ctx.obj.get("dry_run"):
        console.print(f"[yellow]--dry-run :[/] aucun job créé pour {source}")
        raise typer.Exit(0)
    drive = get_drive(cfg)
    with Progress("[progress.description]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                  console=console, transient=True) as bar:
        task = bar.add_task("préparation", total=None)

        def progress(done: int, total: int) -> None:
            bar.update(task, completed=done, total=total or None)

        def on_phase(label: str) -> None:
            bar.update(task, description=label, completed=0, total=None)
            console.log(f"→ {label}")

        try:
            return run_prepare(source, cfg, drive, progress=progress, on_phase=on_phase, **kwargs)
        except PrepareError as exc:
            console.print(f"[red]prepare :[/] {exc}")
            raise typer.Exit(1) from None


def _print_next_steps(cfg: dict, job: dict) -> None:
    from modules.prepare import colab_url

    if job.get("_existing"):
        console.print(f"[yellow]Déjà envoyé :[/] job {job['job_id']} ({job['_existing']}) pour {job['video_id']}. "
                      "Ajoutez --force pour recommencer.")
        return
    where = "uploadée dans ClipGod/sources/" if job.get("source_drive_path") else "téléchargée par le notebook"
    console.print(f"[green]✔ Job {job['job_id']}[/] déposé dans ClipGod/jobs/ — vidéo {job['video_id']} ({where})")
    console.print(f"→ Ouvrez le notebook Colab (GPU T4, Tout exécuter) : {colab_url(cfg)}")
    console.print("→ Une fois terminé, lancez : [bold]clipgod sync[/]")


# ---------------------------------------------------------------------------
# Module 3 — scan (CLOUD GPU — debug local uniquement)
# ---------------------------------------------------------------------------
@app.command()
def scan(
    ctx: typer.Context,
    video: Path = typer.Argument(..., help="Vidéo locale à analyser."),
    output_dir: Path = typer.Option(Path("output/moments"), "--output-dir", "-o"),
    whisper_model: Optional[str] = typer.Option(None, "--whisper-model", help="Override : large-v3 | medium | small."),
    device: Optional[str] = typer.Option(None, "--device", help="cuda | cpu (cpu = très lent)."),
    transcript: Optional[Path] = typer.Option(
        None, "--transcript", help="Transcription JSON existante (re-scan sans whisper)."
    ),
    source_url: Optional[str] = typer.Option(None, "--source-url", help="URL YouTube (graphe most replayed)."),
    streamer: Optional[str] = typer.Option(None, "--streamer", help="Pseudo utilisé dans les hooks."),
) -> None:
    """(cloud GPU) Détecte les meilleurs moments -> moments/<video_id>.json. Normalement exécuté dans le notebook."""
    import copy

    from modules.scan import ScanError
    from modules.scan import scan as run_scan

    cfg = copy.deepcopy(ctx.obj["config"])
    scan_cfg = cfg.setdefault("scan", {})
    if whisper_model:
        scan_cfg.setdefault("whisper", {})["model"] = whisper_model
    if device:
        scan_cfg.setdefault("whisper", {})["device"] = device
    if transcript:
        scan_cfg["transcript_path"] = str(transcript)
    cfg["job"] = {"source_url": source_url, "streamer": streamer}
    try:
        moments = run_scan(str(video), str(output_dir), cfg)
    except (ScanError, FileNotFoundError) as exc:
        console.print(f"[red]Scan échoué :[/] {exc}")
        raise typer.Exit(1) from None
    _print_moments(moments)


def _print_moments(moments: list[dict]) -> None:
    from rich.table import Table

    table = Table(title="Meilleurs moments", show_lines=True)
    for col in ("#", "début", "fin", "score", "raison", "hooks"):
        table.add_column(col)
    for m in moments:
        table.add_row(str(m["rank"]), f"{m['start']:.1f}s", f"{m['end']:.1f}s", f"{m['score']:.0f}",
                      m["reason"], "\n".join(m["hooks"]))
    console.print(table)


# ---------------------------------------------------------------------------
# Module 4 — render (CLOUD GPU — debug local uniquement)
# ---------------------------------------------------------------------------
@app.command()
def render(
    ctx: typer.Context,
    video: Path = typer.Argument(..., help="Vidéo source."),
    moments: Path = typer.Option(..., "--moments", "-m", help="Fichier moments/<video_id>.json."),
    output_dir: Path = typer.Option(Path("output/clips"), "--output-dir", "-o"),
    voiceover: bool = typer.Option(False, "--voiceover", help="Voix off TTS Piper locale."),
    only: Optional[int] = typer.Option(None, "--only", help="Ne rendre que le moment de rang N (1 = meilleur)."),
    layout: Optional[str] = typer.Option(None, "--layout", help="auto | facecam | center."),
    streamer: Optional[str] = typer.Option(None, "--streamer", help="Pseudo pour l'overlay @streamer."),
) -> None:
    """(cloud GPU) Rend les clips 1080x1920 -> clips/<video_id>_<idx>.mp4. Normalement exécuté dans le notebook."""
    import copy
    import json

    from modules.render import RenderError
    from modules.render import render as run_render

    cfg = copy.deepcopy(ctx.obj["config"])
    rcfg = cfg.setdefault("render", {})
    if voiceover:
        rcfg.setdefault("voiceover", {})["enabled"] = True
    if layout:
        rcfg.setdefault("layout", {})["mode"] = layout
    try:
        doc = json.loads(moments.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]Fichier de moments illisible :[/] {exc}")
        raise typer.Exit(1) from None
    selected = doc.get("moments", [])
    if only is not None:
        selected = [m for m in selected if m.get("rank") == only]
    if streamer:                                   # --streamer l'emporte sur le pseudo enregistré par le scan
        selected = [{**m, "streamer": streamer} for m in selected]
    cfg["job"] = {"video_id": doc.get("video_id"), "streamer": streamer or doc.get("streamer")}
    try:
        clips = run_render(selected, str(video), str(output_dir), cfg)
    except (RenderError, FileNotFoundError) as exc:
        console.print(f"[red]Rendu échoué :[/] {exc}")
        raise typer.Exit(1) from None
    for clip in clips:
        console.print(f"[green]✔[/] {clip}")
    if not clips:
        console.print("[yellow]Aucun clip produit.[/]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Lien Drive -> laptop
# ---------------------------------------------------------------------------
@app.command()
def sync(
    ctx: typer.Context,
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Ne synchroniser qu'un job."),
    status: bool = typer.Option(False, "--status", help="Affiche l'état des jobs sans télécharger."),
    redownload: bool = typer.Option(False, "--redownload", help="Re-vérifie aussi les jobs déjà synchronisés."),
) -> None:
    """Télécharge ClipGod/clips/ + moments/ vers output/ et met à jour l'état local."""
    from rich.table import Table

    from core.config import path
    from core.drive import get_drive
    from core.storage import StateStore
    from modules.sync import queue_overview
    from modules.sync import sync as run_sync

    cfg = ctx.obj["config"]
    drive = get_drive(cfg)
    state = StateStore(path(cfg, "state"))
    rows = queue_overview(drive, state)
    colors = {"queued": "yellow", "running": "cyan", "retry": "magenta", "done": "green", "synced": "dim",
              "failed": "red"}
    table = Table(title="File ClipGod (Drive)")
    for col in ("job", "vidéo", "streamer", "statut", "détail"):
        table.add_column(col)
    for r in rows:
        if job_id and r.job_id != job_id:
            continue
        table.add_row(r.job_id, r.video_id or "", r.streamer or "", f"[{colors.get(r.status, 'white')}]{r.status}[/]",
                      r.detail)
    console.print(table if rows else "[dim]Aucun job sur Drive. Lancez clipgod prepare --source <url>.[/]")
    if status or ctx.obj.get("dry_run"):
        return

    report = run_sync(cfg, drive, job_id=job_id, redownload=redownload)
    n_clips = sum(1 for p in report.downloaded if p.endswith(".mp4"))
    console.print(f"[green]✔ {len(report.synced_jobs)} job(s) synchronisé(s)[/], {n_clips} clip(s) téléchargé(s), "
                  f"{report.skipped} fichier(s) déjà à jour.")
    for src in report.cleaned_sources:
        console.print(f"[dim]  source retirée de Drive : {src}[/]")
    for jid, err in report.failed_jobs:
        console.print(f"[red]✖ job {jid} en échec :[/] {err}")
    for err in report.errors:
        console.print(f"[red]✖[/] {err}")
    if report.errors:
        raise typer.Exit(1)
    if report.synced_jobs:
        console.print("→ Étape suivante : [bold]clipgod publish[/] (validation monetize incluse)")


# ---------------------------------------------------------------------------
# Module 5 — publish (laptop)
# ---------------------------------------------------------------------------
@app.command()
def publish(
    ctx: typer.Context,
    clip: Optional[Path] = typer.Option(None, "--clip", help="Un seul clip ; défaut : tous les clips synchronisés."),
    platform: list[str] = typer.Option(None, "--platform", "-p", help="youtube_shorts | tiktok | instagram_reels."),
    test: Optional[bool] = typer.Option(None, "--test/--live", help="Mode test : YouTube en privé (défaut config)."),
    schedule: bool = typer.Option(True, "--schedule/--now", help="Étale sur les créneaux de forte audience."),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="DANGER : saute la validation monetize."),
    plan: bool = typer.Option(False, "--plan", help="Affiche le plan sans rien envoyer."),
) -> None:
    """Valide (monetize) puis publie/programme les clips (max 6/jour, A/B test des hooks)."""
    from rich.table import Table

    from core.quota import YouTubeQuota
    from modules.publish import PublishError
    from modules.publish import publish as run_publish

    cfg = ctx.obj["config"]
    test_mode = bool(cfg.get("publish", {}).get("test_mode", True)) if test is None else test
    dry = plan or bool(ctx.obj.get("dry_run"))
    if not test_mode and not dry:
        console.print("[bold yellow]Mode LIVE : les clips seront publiés publiquement aux créneaux prévus.[/]")
    try:
        report = run_publish(cfg, clip=clip, platforms=platform or None, test=test_mode, schedule=schedule,
                             skip_validation=skip_validation, dry_run=dry)
    except PublishError as exc:
        console.print(f"[red]publish :[/] {exc}")
        raise typer.Exit(1) from None

    for clip_id, reasons in report.rejected:
        console.print(f"[red]✖ refusé[/] {clip_id} : {' ; '.join(reasons)}")
    if report.posted:
        title = "Plan de publication (rien n'a été envoyé)" if dry else ("Publié (mode test : privé)" if test_mode
                                                                          else "Publié / programmé")
        table = Table(title=title)
        for col in ("clip", "plateforme", "créneau", "statut", "hook", "lien"):
            table.add_column(col, overflow="fold")
        for p in report.posted:
            table.add_row(p["clip_id"], p["platform"], p["slot"] or "immédiat", p["status"], p.get("hook") or "",
                          p.get("url") or "")
        console.print(table)
    for pid, why in report.skipped:
        console.print(f"[yellow]↷ {pid}[/] : {why}")
    for pid, err in report.errors:
        console.print(f"[red]✖ {pid}[/] : {err}")
    console.print(f"[dim]Quota YouTube restant aujourd'hui : {YouTubeQuota.from_config(cfg).remaining()} u[/]")
    if not report.posted and not report.rejected:
        console.print("[dim]Rien à publier : lancez clipgod sync pour récupérer de nouveaux clips.[/]")
    if report.errors and not report.posted:
        raise typer.Exit(1)


@app.command()
def tiktok(
    ctx: typer.Context,
    platform: str = typer.Option("tiktok", "--platform", "-p", help="tiktok | instagram_reels."),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Nombre de clips (séance : 3 par défaut ; "
                                        "--export : tous)."),
    schedule: bool = typer.Option(True, "--schedule/--now", help="Propose un créneau de forte audience par clip."),
    plan_only: bool = typer.Option(False, "--plan", help="Affiche les prochains posts sans rien marquer."),
    choose: bool = typer.Option(False, "--choose", help="Liste tous les clips : regarde-les et choisis "
                                "lesquels publier (ex. 1,3,5-7)."),
    open_studio: bool = typer.Option(True, "--open/--no-open", help="Ouvre TikTok Studio dans le navigateur."),
    export: bool = typer.Option(False, "--export", "-e", help="Copie la meilleure vidéo de chaque moment dans "
                                "output/a_publier (nommée d'après son hook) + legendes.txt. Sans doublons."),
) -> None:
    """Publication manuelle assistée : légende copiée, clip sélectionné, créneau conseillé (gratuit, sans API)."""
    import webbrowser
    from zoneinfo import ZoneInfo

    from rich.panel import Panel
    from rich.table import Table

    from modules.manual import (
        STUDIO_URLS,
        copy_to_clipboard,
        parse_selection,
        plan_manual,
        play_file,
        record_manual,
        reveal_file,
    )

    cfg = ctx.obj["config"]
    tz = ZoneInfo(str(cfg.get("project", {}).get("timezone", "Europe/Paris")))
    rejected: list[tuple[str, list[str]]] = []
    if export:
        from modules.manual import export_ready, open_folder

        rep = export_ready(cfg, platform=platform, limit=limit, schedule=schedule,
                           rejected=rejected)
        for clip_id, reasons in rejected:
            console.print(f"[red]✖ refusé[/] {clip_id} : {' ; '.join(reasons)}")
        for group in rep.skipped_language:
            console.print(f"[yellow]↷ {group}[/] : source dans une autre langue, écartée")
        if not rep.files:
            console.print("[dim]Aucun nouveau clip à exporter : lancez clipgod sync après un nouveau job.[/]")
            return
        for target, post in rep.files:
            when = post.slot.astimezone(tz).strftime("%d/%m à %H:%M") if post.slot else "dès que possible"
            console.print(f"  [green]✔[/] {target.name}  [dim]→ {when}[/]")
        console.print(f"\n[bold]{len(rep.files)} vidéo(s) + legendes.txt[/] dans {rep.folder}")
        console.print("[dim]Ces moments ne seront plus réexportés. Publie-les dans l'ordre des numéros.[/]")
        if open_studio:
            open_folder(rep.folder)
        return
    if choose:
        listing = plan_manual(cfg, platform=platform, schedule=False, rejected=rejected, one_per_moment=False)
        for clip_id, reasons in rejected:
            console.print(f"[red]✖ refusé[/] {clip_id} : {' ; '.join(reasons)}")
        if not listing:
            console.print("[dim]Rien à publier : lancez clipgod sync pour récupérer de nouveaux clips.[/]")
            return
        table = Table(title=f"Clips disponibles pour {platform}")
        for col in ("#", "clip", "score", "durée", "hook"):
            table.add_column(col, overflow="fold")
        for i, post in enumerate(listing, 1):
            table.add_row(str(i), post.clip.path.name, f"{post.clip.score:.0f}",
                          f"{float(post.clip.meta.get('duration') or 0):.0f} s", str(post.clip.meta.get("hook") or ""))
        console.print(table)
        console.print("[dim]Plusieurs variantes d'un même moment (_h1, _h2…) = même vidéo, hook différent : "
                      "n'en publie qu'une.[/]")
        selected: list[int] = []
        while not selected:
            answer = typer.prompt("Numéros à publier (ex. 1,3,5-7) · v3 = voir le clip 3 · q = quitter",
                                  default="", show_default=False).strip().lower()
            if answer in ("", "q"):
                return
            if answer.startswith("v"):
                try:
                    play_file(listing[int(answer[1:].strip()) - 1].clip.path)
                except (ValueError, IndexError):
                    console.print("[red]Numéro invalide.[/]")
                continue
            try:
                selected = parse_selection(answer, len(listing))
            except ValueError as exc:
                console.print(f"[red]{exc}[/]")
        posts = plan_manual(cfg, platform=platform, schedule=schedule, only=[listing[i - 1].clip.clip_id
                                                                             for i in selected])
        if len(posts) < len(selected):
            console.print(f"[yellow]{len(selected) - len(posts)} clip(s) sans créneau libre dans les "
                          "10 prochains jours : relance plus tard.[/]")
    else:
        posts = plan_manual(cfg, platform=platform, limit=3 if limit is None else limit, schedule=schedule,
                            rejected=rejected)
        for clip_id, reasons in rejected:
            console.print(f"[red]✖ refusé[/] {clip_id} : {' ; '.join(reasons)}")
    if not posts:
        console.print("[dim]Rien à publier : lancez clipgod sync pour récupérer de nouveaux clips.[/]")
        return

    def when(post) -> str:
        return post.slot.astimezone(tz).strftime("%a %d/%m à %H:%M") if post.slot else "maintenant"

    if plan_only or ctx.obj.get("dry_run"):
        for i, post in enumerate(posts, 1):
            console.print(Panel(post.caption, title=f"{i}. {post.clip.path.name} · {when(post)}", expand=False))
        return

    if open_studio and platform in STUDIO_URLS:
        webbrowser.open(STUDIO_URLS[platform])
    console.print(f"[bold]{len(posts)} clip(s) à déposer sur {platform}.[/] Pour chacun : glisse la vidéo, "
                  "colle la légende (Ctrl+V), règle la date, puis reviens ici.\n")
    done = 0
    for i, post in enumerate(posts, 1):
        copied = copy_to_clipboard(post.caption)
        reveal_file(post.clip.path)
        console.print(Panel(post.caption, expand=False,
                            title=f"{i}/{len(posts)} · {post.clip.path.name} · score {post.clip.score:.0f}",
                            subtitle="légende copiée ✔" if copied else "copie la légende ci-dessus"))
        console.print(f"   📁 {post.clip.path}")
        console.print(f"   🕒 Programmer : [bold cyan]{when(post)}[/] ({tz.key})")
        while True:
            answer = typer.prompt("   Entrée = c'est posté · colle le lien TikTok (optionnel) · v = voir · "
                                  "s = passer · q = quitter", default="", show_default=False).strip()
            if answer.lower() != "v":
                break
            play_file(post.clip.path)
        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        record_manual(cfg, post, url=answer if answer.startswith("http") else None)
        done += 1
        console.print("   [green]✔ enregistré[/]\n")
    console.print(f"[bold green]{done} post(s) enregistré(s).[/] Les liens collés servent à clipgod learn "
                  "(vues à saisir dans data/manual_views.csv).")


@app.command()
def topshorts(
    ctx: typer.Context,
    hours: float = typer.Option(48, "--hours", help="Shorts publiés depuis N heures (24 ou 48)."),
    count: int = typer.Option(10, "--count", "-n", help="Nombre de Shorts listés."),
    all_shorts: bool = typer.Option(False, "--all", help="Sans filtre gameplay."),
    output: Path = typer.Option(Path("top10_shorts_gta6.txt"), "--output", "-o", help="Fichier texte produit."),
) -> None:
    """Liste les Shorts GTA 6 les plus vus (gameplay uniquement par défaut) avec leurs liens -> fichier .txt."""
    from core.config import ConfigError
    from core.quota import YouTubeQuota
    from core.storage import utc_now
    from modules.discover import YouTubeDiscover, _youtube_service
    from modules.viral import top_shorts, write_top_shorts

    cfg = ctx.obj["config"]
    try:
        yt = YouTubeDiscover(_youtube_service(cfg), YouTubeQuota.from_config(cfg), cfg, budget=250)
        kept, dropped = top_shorts(cfg, yt, now=utc_now(), hours=hours, count=count, gameplay_only=not all_shorts)
    except ConfigError as exc:
        console.print(f"[red]topshorts :[/] {exc}")
        raise typer.Exit(1) from None
    out = write_top_shorts(output, kept, dropped, hours=hours, gameplay_only=not all_shorts)
    for i, (sh, why) in enumerate(kept, 1):
        console.print(f"{i:>2}. [bold]{sh.title[:70]}[/]\n    {sh.channel} · {sh.views:,} vues · {sh.url}"
                      .replace(",", " ") + (f" [dim]({why})[/]" if why else ""))
    if not all_shorts:
        console.print(f"[dim]{len(dropped)} Short(s) écarté(s) (pas du gameplay ou indéterminés) : liste en bas du "
                      "fichier.[/]")
    console.print(f"\n✔ {out.resolve()}")


@app.command()
def toptiktok(
    ctx: typer.Context,
    hours: float = typer.Option(48, "--hours", help="Vidéos publiées depuis N heures (24 ou 48)."),
    count: int = typer.Option(10, "--count", "-n", help="Nombre de vidéos listées."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Hashtags séparés par des virgules (défaut : config)."),
    scrolls: Optional[int] = typer.Option(None, "--scrolls", help="Défilements par hashtag (plus = plus de vidéos)."),
    all_videos: bool = typer.Option(False, "--all", help="Sans filtre gameplay."),
    lang: Optional[str] = typer.Option(None, "--lang", help="Langues gardées, ex. « en » ou « en,fr » ; « all » = "
                                       "toutes (défaut : config, en)."),
    headless: bool = typer.Option(False, "--headless", help="Navigateur invisible (déconseillé : TikTok bloque plus)."),
    output: Path = typer.Option(Path("top10_tiktok_gta6.txt"), "--output", "-o", help="Fichier texte produit."),
) -> None:
    """Top des vidéos TikTok GTA 6 gameplay les plus vues (Chrome piloté, sans API) -> fichier .txt."""
    from modules.tiktok_top import TikTokError, run_toptiktok

    cfg = ctx.obj["config"]
    console.print("[dim]Une fenêtre Chrome va s'ouvrir sur TikTok : ne la ferme pas. Si une vérification "
                  "s'affiche, résous-la (une seule fois).[/]")
    try:
        out, kept = run_toptiktok(cfg, hours=hours, count=count,
                                  tags=[t.strip().lstrip("#") for t in tags.split(",")] if tags else None,
                                  scrolls=scrolls, headless=headless, gameplay_only=not all_videos, output=output,
                                  languages=None if lang is None else ([] if lang.lower() == "all" else
                                                                       [x.strip() for x in lang.split(",")]),
                                  say=console.print)
    except TikTokError as exc:
        console.print(f"[red]toptiktok :[/] {exc}")
        raise typer.Exit(1) from None
    for i, (v, why) in enumerate(kept, 1):
        console.print(f"{i:>2}. [bold]{(v.text or '(sans description)')[:70]}[/]\n    @{v.author} · "
                      f"{v.views:,} vues · {v.url}".replace(",", " ") + (f" [dim]({why})[/]" if why else ""))
    if not kept:
        console.print("[yellow]Aucune vidéo gameplay récente trouvée : essaie --hours 72, --scrolls 15 ou --all.[/]")
    console.print(f"\n✔ {out.resolve()}")


@app.command()
def viral(
    ctx: typer.Context,
    count: Optional[int] = typer.Option(None, "--count", "-n", help="Moments à préparer (défaut : 5)."),
    plan_only: bool = typer.Option(False, "--plan", help="Localise les moments sans créer de jobs."),
) -> None:
    """Shorts GTA 6 les plus vus -> le même moment recoupé dans la VOD d'origine -> jobs Colab (bestof --viral)."""
    from core.config import ConfigError
    from core.drive import get_drive
    from modules.prepare import colab_url
    from modules.viral import ViralError, run_viral

    cfg = ctx.obj["config"]
    plan_only = plan_only or bool(ctx.obj.get("dry_run"))
    try:
        drive = None if plan_only else get_drive(cfg)
        found = run_viral(cfg, drive, count=count, plan_only=plan_only, on_event=console.print)
    except (ConfigError, ViralError) as exc:
        console.print(f"[red]viral :[/] {exc}")
        raise typer.Exit(1) from None
    if not found:
        console.print("[yellow]Aucun moment viral localisé aujourd'hui.[/] Réessaie plus tard, ou utilise "
                      "clipgod discover --trending --prepare 2.")
        raise typer.Exit(1)
    console.print(f"\n[bold green]{len(found)} moment(s) viral(aux) localisé(s)[/]"
                  + ("" if plan_only else " et envoyé(s) sur Drive."))
    if not plan_only:
        console.print(f"→ Colab : [bold]{colab_url(cfg)}[/] (Exécution › Tout exécuter, ~1-2 min par moment)")
        console.print("→ Puis : [bold]clipgod sync[/] et [bold]clipgod bestof --viral[/]")


@app.command()
def bestof(
    ctx: typer.Context,
    count: Optional[int] = typer.Option(None, "--count", "-n", help="Nombre de moments (défaut : 5)."),
    hours: Optional[float] = typer.Option(None, "--hours", help="Moments rendus depuis N heures (défaut : 48)."),
    allow_repeat: bool = typer.Option(False, "--allow-repeat", help="Autorise des moments déjà utilisés."),
    plan_only: bool = typer.Option(False, "--plan", help="Affiche la sélection sans monter la vidéo."),
    open_folder: bool = typer.Option(True, "--open/--no-open", help="Ouvre le dossier à la fin."),
    viral: bool = typer.Option(False, "--viral", help="Moments préparés par clipgod viral, classés par vues "
                               "du Short d'origine."),
    folder: Optional[Path] = typer.Option(None, "--folder", "-f", help="Compile les vidéos de ce dossier (ordre des "
                                          "noms = classement, 1er fichier = n°1), puis les range dans publies/<date>."),
    keep: bool = typer.Option(False, "--keep", help="Avec --folder : laisse les vidéos dans le dossier."),
) -> None:
    """Top 5 GTA 6 du jour : compilation verticale #5 -> #1 des meilleurs moments récents + légende."""
    from rich.table import Table

    from modules.bestof import BestOfError, make_bestof, select_picks

    cfg = ctx.obj["config"]
    if folder is not None and (plan_only or ctx.obj.get("dry_run")):
        from modules.bestof import picks_from_folder

        try:
            picks = picks_from_folder(folder)
        except BestOfError as exc:
            console.print(f"[red]bestof :[/] {exc}")
            raise typer.Exit(1) from None
        for p in picks:
            console.print(f"  #{p.rank}  {p.path.name}  [dim]{p.duration:.0f} s"
                          f"{'' if p.meta.get('has_audio') else ' · sans son'}[/]")
        return
    if plan_only or ctx.obj.get("dry_run"):
        picks = select_picks(cfg, count=count, hours=hours, allow_repeat=allow_repeat or None, viral=viral)
        table = Table(title=f"Sélection du best-of ({len(picks)} moments)")
        for col in ("rang", "clip", "streamer", "vues du Short" if viral else "score", "hook"):
            table.add_column(col, overflow="fold")
        for p in picks:
            value = f"{int((p.meta.get('viral') or {}).get('views') or 0):,}".replace(",", " ") if viral \
                else f"{p.score:.0f}"
            table.add_row(f"#{p.rank}", p.path.name, p.streamer, value, str(p.meta.get("hook") or ""))
        console.print(table)
        return
    try:
        result = make_bestof(cfg, count=count, hours=hours, allow_repeat=allow_repeat or None, viral=viral,
                             folder=folder, archive=not keep)
    except BestOfError as exc:
        console.print(f"[red]bestof :[/] {exc}")
        raise typer.Exit(1) from None
    for p in sorted(result.picks, key=lambda p: -p.rank):
        label = p.path.name if folder is not None else f"@{p.streamer}"
        extra = "" if folder is not None else f" · score {p.score:.0f}"
        console.print(f"  #{p.rank}  {label}  [dim]{p.use_seconds:.0f} s{extra}[/]  "
                      f"{p.meta.get('hook') or ''}")
    console.print(f"\n[bold green]✔ {result.video.name}[/] ({result.duration:.0f} s) + {result.caption_file.name} "
                  f"dans {result.video.parent}")
    console.print("[dim]Publie-le dans TikTok Studio avec la légende du .txt (crédits compris).[/]")
    if open_folder:
        from modules.manual import open_folder as _open

        _open(result.video.parent)


# ---------------------------------------------------------------------------
# Module 6 — monetize (laptop)
# ---------------------------------------------------------------------------
@app.command()
def monetize(
    ctx: typer.Context,
    refresh: bool = typer.Option(False, "--refresh", help="Force le re-scraping des campagnes Whop."),
    validate: Optional[Path] = typer.Option(None, "--validate", help="Valide un clip (.mp4) contre les règles."),
    campaign: Optional[str] = typer.Option(None, "--campaign", help="Campagne pour --validate (slug)."),
    top: int = typer.Option(10, "--top", help="Nombre de campagnes affichées."),
    all_: bool = typer.Option(False, "--all", help="Affiche aussi les campagnes exclues."),
) -> None:
    """Classe les campagnes Whop par $/heure estimé et valide les clips (-> data/rejected.json)."""
    from rich.table import Table

    from core.storage import read_json
    from modules.monetize import (
        fetch_campaigns,
        match_campaign,
        rank_campaigns,
        record_rejection,
        rules_for,
        validate_clip,
    )

    cfg = ctx.obj["config"]
    campaigns = fetch_campaigns(cfg, refresh=refresh, allow_scrape=not ctx.obj.get("dry_run"))
    if validate is not None:
        meta = read_json(validate.with_suffix(".json"), default={}) or {}
        chosen = match_campaign(campaigns, slug=campaign or meta.get("campaign"), streamer=meta.get("streamer"))
        result = validate_clip({"clip_id": validate.stem, **meta}, rules_for(cfg, chosen), validate)
        label = f" (campagne {chosen.slug})" if chosen else " (règles par défaut)"
        if result.ok:
            console.print(f"[green]✔ {validate.name} valide{label}[/]")
            return
        record_rejection(cfg, result, campaign=chosen.slug if chosen else None)
        console.print(f"[red]✖ {validate.name} refusé{label} :[/]")
        for reason in result.reasons:
            console.print(f"  - {reason}")
        raise typer.Exit(1)

    rows = rank_campaigns(cfg, campaigns)
    shown = [r for r in rows if r["eligible"] or all_][:top]
    if not rows:
        console.print("[yellow]Aucune campagne trouvée.[/] Ajoutez-en dans data/whop_campaigns.yaml ou des lignes "
                      "whop:<slug> dans data/campaigns.txt, puis clipgod monetize --refresh.")
        return
    table = Table(title="Campagnes de clipping (classées par $/heure de VOD estimé)")
    for col in ("campagne", "CPM $", "budget restant $", "vues/clip", "$/clip", "$/heure", "plateformes", "règles", "note"):
        table.add_column(col, overflow="fold")
    for r in shown:
        table.add_row(r["name"], f"{r['cpm_usd']:.2f}" if r["cpm_usd"] is not None else "?",
                      f"{r['budget_remaining_usd']:,.0f}".replace(",", " ") if r["budget_remaining_usd"] is not None else "?",
                      f"{r['expected_views']} ({r['views_source']})", f"{r['usd_per_clip']:.2f}",
                      f"[bold]{r['usd_per_hour']:.2f}[/]", ", ".join(r["platforms"]) or "toutes", r["rules"],
                      ", ".join(r["excluded_because"]) or r["source"])
    console.print(table)
    hidden = len([r for r in rows if not r["eligible"]])
    if hidden and not all_:
        console.print(f"[dim]{hidden} campagne(s) exclue(s) (CPM/budget/inactive) : --all pour les voir.[/]")
    best = next((r for r in rows if r["eligible"]), None)
    if best:
        console.print(f"→ Meilleure campagne : [bold]{best['name']}[/] — clipgod prepare --source <url> "
                      f"--campaign {best['slug']}")


# ---------------------------------------------------------------------------
# Module 7 — learn (laptop, CPU)
# ---------------------------------------------------------------------------
@app.command()
def learn(
    ctx: typer.Context,
    collect: bool = typer.Option(True, "--collect/--no-collect", help="Collecte les vues 1h/6h/24h/72h."),
    retrain: bool = typer.Option(False, "--retrain", help="Force le réentraînement du modèle."),
    report: bool = typer.Option(True, "--report/--no-report", help="Génère report.html."),
    push: bool = typer.Option(True, "--push/--no-push", help="Envoie le nouveau model.pkl sur Drive."),
) -> None:
    """Suit les performances, réentraîne data/model.pkl et génère report.html."""
    from modules.learn import learn as run_learn

    cfg = ctx.obj["config"]
    res = run_learn(cfg, collect_views=collect and not ctx.obj.get("dry_run"), retrain=retrain, report=report,
                    push=push and not ctx.obj.get("dry_run"))
    if res.collected is not None:
        detail = ", ".join(f"{k}: {v}" for k, v in res.collected.by_platform.items()) or "aucune"
        console.print(f"📈 {res.collected.observed} mesure(s) de vues ({detail})")
        if res.collected.missing:
            console.print(f"[dim]  {len(res.collected.missing)} post(s) sans source de vues (TikTok/Instagram : "
                          "data/manual_views.csv ou publish.upload_post.analytics_endpoint)[/]")
    console.print(f"🗂️  data/historical_scores.csv : {res.rows} post(s), {res.measured} mesuré(s) à la cible")
    if res.trained and res.model_meta:
        cv = res.model_meta.get("cv_r2")
        console.print(f"[green]🧠 model.pkl réentraîné[/] ({res.train_reason}) sur {res.model_meta['n_samples']} clips"
                      + (f", R² CV = {cv:.2f}" if cv is not None else ""))
        if res.pushed:
            console.print("   envoyé sur Drive : le prochain scan l'utilisera automatiquement")
    else:
        console.print(f"[dim]🧠 pas de réentraînement : {res.train_reason}[/]")
    if res.report_path:
        console.print(f"📄 Rapport : {res.report_path}")


# ---------------------------------------------------------------------------
# Module 2 — watch (laptop/VPS, désactivable)
# ---------------------------------------------------------------------------
@app.command()
def watch(
    ctx: typer.Context,
    url: str = typer.Argument(..., help="URL du live (YouTube / Twitch / Kick)."),
    force: bool = typer.Option(False, "--force", help="Ignore disable-watch de config.yaml."),
    quality: Optional[str] = typer.Option(None, "--quality", help="Qualité streamlink (ex. 720p60, 480p, best)."),
    no_chat: bool = typer.Option(False, "--no-chat", help="Buffer seul, captures manuelles uniquement (touche c)."),
    auto_prepare: Optional[bool] = typer.Option(None, "--auto-prepare/--no-auto-prepare",
                                                help="Envoie chaque capture au notebook automatiquement."),
) -> None:
    """Buffer roulant 8 min + détection de hype dans le chat -> inbox/raw_<timestamp>.mp4."""
    from modules.watch import WatchError
    from modules.watch import watch as run_watch

    cfg = ctx.obj["config"]
    if (cfg.get("watch") or {}).get("disable-watch", True) and not force:
        console.print("[yellow]watch est désactivé[/] (watch.disable-watch: true dans config.yaml) car il télécharge "
                      "le live en continu. Relancez avec [bold]--force[/] ou passez l'option à false.")
        raise typer.Exit(1)
    try:
        state = run_watch(cfg, url, quality=quality, with_chat=not no_chat, auto_prepare=auto_prepare, console=console)
    except WatchError as exc:
        console.print(f"[red]watch :[/] {exc}")
        raise typer.Exit(1) from None
    ready = [c for c in state.captures if c.status in ("ready", "sent")]
    console.print(f"[green]✔ {len(ready)} capture(s)[/] dans inbox/")
    for c in ready:
        console.print(f"  clipgod prepare --source {c.file}" + ("  [dim](déjà envoyée)[/]" if c.status == "sent" else ""))


# ---------------------------------------------------------------------------
# Pipeline côté laptop
# ---------------------------------------------------------------------------
@app.command()
def run(
    ctx: typer.Context,
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="URL/fichier. Absent : prend la meilleure source de discover."
    ),
    streamer: Optional[str] = typer.Option(None, "--streamer", help="Pseudo pour l'overlay @streamer."),
    campaign: Optional[str] = typer.Option(None, "--campaign", help="Campagne cible."),
    download: Optional[str] = typer.Option(None, "--download", help="auto | local | cloud."),
    resume: bool = typer.Option(True, "--resume/--fresh", help="--fresh : recrée un job même si la vidéo existe."),
) -> None:
    """Enchaîne discover -> prepare puis indique d'ouvrir le notebook Colab et de lancer `clipgod sync`."""
    if source is None:
        from modules.discover import DiscoverError, best_source

        try:
            source = best_source(ctx.obj["config"])
        except DiscoverError as exc:
            console.print(f"[red]discover :[/] {exc}")
            raise typer.Exit(1) from None
        console.print(f"Meilleure VOD trouvée par discover : [bold]{source}[/]")
    job = _prepare_job(ctx, source, job_id=None, streamer=streamer, campaign=campaign, voiceover=False,
                       download=download, force=not resume)
    _print_next_steps(ctx.obj["config"], job)


def main() -> None:
    """Point d'entrée console (`clipgod ...`). Convertit les erreurs en code de sortie 1."""
    try:
        app()
    except NotImplementedError as exc:  # garde-fou : sortie propre plutôt qu'une trace brute
        console.print(f"[yellow]Pas encore implémenté :[/] {exc}")
        raise SystemExit(1) from None
    except (ConfigError, DriveError) as exc:
        console.print(f"[red]Erreur :[/] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":  # point d'entrée CLI uniquement — aucune logique métier ici
    main()
