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
) -> None:
    """Trouve les streams live / VODs < 24h (> 45 min) GTA 6 -> data/sources.json."""
    from rich.table import Table

    from core.quota import YouTubeQuota
    from modules.discover import DiscoverError
    from modules.discover import discover as run_discover

    cfg = ctx.obj["config"]
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
