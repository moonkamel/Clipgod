"""ClipGod — CLI d'orchestration (laptop).

Ce fichier ne contient AUCUNE logique métier lourde : il parse les arguments,
charge la configuration et délègue aux modules (`modules/*`).

Les modules lourds `scan` et `render` sont exécutés sur GPU cloud (Colab/Kaggle)
via les notebooks ; les commandes `scan` / `render` ci-dessous existent pour le
debug local sur de petits extraits ou une machine équipée d'un GPU.

Statut : squelette — chaque commande lève NotImplementedError.
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
    limit: int = typer.Option(25, "--limit", "-n", help="Nombre max de sources par plateforme."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore le cache local de 5 min."),
    output: Path = typer.Option(Path("data/sources.json"), "--output", "-o"),
) -> None:
    """Trouve les streams live / VODs < 24h (> 45 min) GTA 6 -> data/sources.json."""
    raise NotImplementedError("discover : à implémenter (étape 8).")


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
        task = bar.add_task("transfert", total=None)

        def progress(done: int, total: int) -> None:
            bar.update(task, completed=done, total=total or None)

        try:
            return run_prepare(source, cfg, drive, progress=progress, **kwargs)
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
    clip: Optional[Path] = typer.Option(None, "--clip", help="Un seul clip ; défaut : tous les clips de output/."),
    platform: list[str] = typer.Option(None, "--platform", "-p", help="youtube_shorts | tiktok | instagram_reels."),
    test: bool = typer.Option(True, "--test/--live", help="Mode test (privé / non listé) par défaut."),
    schedule: bool = typer.Option(True, "--schedule/--now", help="Étale sur les créneaux de forte audience."),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="DANGER : saute la validation monetize."),
) -> None:
    """Valide (monetize) puis publie/programme les clips (max 6/jour, A/B test des hooks)."""
    raise NotImplementedError("publish : à implémenter (étape 9).")


# ---------------------------------------------------------------------------
# Module 6 — monetize (laptop)
# ---------------------------------------------------------------------------
@app.command()
def monetize(
    ctx: typer.Context,
    refresh: bool = typer.Option(False, "--refresh", help="Force le re-scraping des campagnes."),
    validate: Optional[Path] = typer.Option(None, "--validate", help="Valide un clip contre les règles."),
    top: int = typer.Option(10, "--top", help="Nombre de campagnes affichées."),
) -> None:
    """Classe les campagnes Whop par $/heure estimé et valide les clips (-> data/rejected.json)."""
    raise NotImplementedError("monetize : à implémenter (étape 10).")


# ---------------------------------------------------------------------------
# Module 7 — learn (laptop, CPU)
# ---------------------------------------------------------------------------
@app.command()
def learn(
    ctx: typer.Context,
    collect: bool = typer.Option(True, "--collect/--no-collect", help="Collecte les vues 1h/6h/24h/72h."),
    retrain: bool = typer.Option(False, "--retrain", help="Force le réentraînement du modèle."),
    report: bool = typer.Option(True, "--report/--no-report", help="Génère report.html."),
) -> None:
    """Suit les performances, réentraîne data/model.pkl et génère report.html."""
    raise NotImplementedError("learn : à implémenter (étape 11).")


# ---------------------------------------------------------------------------
# Module 2 — watch (laptop/VPS, désactivable)
# ---------------------------------------------------------------------------
@app.command()
def watch(
    ctx: typer.Context,
    url: str = typer.Argument(..., help="URL du live (YouTube / Twitch / Kick)."),
    force: bool = typer.Option(False, "--force", help="Ignore disable-watch de config.yaml."),
) -> None:
    """Buffer roulant 8 min + détection de hype dans le chat -> inbox/raw_<timestamp>.mp4."""
    raise NotImplementedError("watch : à implémenter (étape 12).")


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
        from modules.discover import best_source  # étape 8

        source = best_source(ctx.obj["config"])
        console.print(f"Meilleure source trouvée par discover : {source}")
    job = _prepare_job(ctx, source, job_id=None, streamer=streamer, campaign=campaign, voiceover=False,
                       download=download, force=not resume)
    _print_next_steps(ctx.obj["config"], job)


def main() -> None:
    """Point d'entrée console (`clipgod ...`). Convertit les erreurs en code de sortie 1."""
    try:
        app()
    except NotImplementedError as exc:  # squelette : sortie propre plutôt qu'une trace brute
        console.print(f"[yellow]Pas encore implémenté :[/] {exc}")
        raise SystemExit(1) from None
    except (ConfigError, DriveError) as exc:
        console.print(f"[red]Erreur :[/] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":  # point d'entrée CLI uniquement — aucune logique métier ici
    main()
