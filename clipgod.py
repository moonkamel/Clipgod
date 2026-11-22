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
    config: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Chemin vers config.yaml.", exists=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Logs DEBUG en console."),
    dry_run: bool = typer.Option(False, "--dry-run", help="N'écrit rien / ne publie rien."),
) -> None:
    """Options partagées par toutes les commandes (stockées dans ctx.obj)."""
    ctx.obj = {"config_path": config, "verbose": verbose, "dry_run": dry_run}


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
) -> None:
    """(cloud GPU) Détecte les meilleurs moments -> moments/<video_id>.json. Normalement exécuté dans le notebook."""
    raise NotImplementedError("scan : à implémenter (étape 3).")


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
) -> None:
    """(cloud GPU) Rend les clips 1080x1920 -> clips/<video_id>_<idx>.mp4. Normalement exécuté dans le notebook."""
    raise NotImplementedError("render : à implémenter (étape 5).")


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


if __name__ == "__main__":  # point d'entrée CLI uniquement — aucune logique métier ici
    main()
