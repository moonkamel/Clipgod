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
) -> None:
    """Télécharge la source, l'uploade dans ClipGod/sources/ et crée ClipGod/jobs/<job_id>.json."""
    raise NotImplementedError("prepare : à implémenter (étape 6).")


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
) -> None:
    """Télécharge ClipGod/clips/ + moments/ vers output/ et met à jour l'état local."""
    raise NotImplementedError("sync : à implémenter (étape 6).")


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
    resume: bool = typer.Option(True, "--resume/--fresh", help="Reprend depuis le dernier checkpoint."),
) -> None:
    """Enchaîne discover -> prepare puis indique d'ouvrir le notebook Colab et de lancer `clipgod sync`."""
    raise NotImplementedError("run : à implémenter (après prepare/sync).")


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
