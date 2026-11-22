"""Génère notebooks/clipgod_colab.ipynb à partir de ce script (source de vérité).

    python scripts/build_notebooks.py          # régénère
    python scripts/build_notebooks.py --check  # échoue si le notebook commité n'est pas à jour (CI / tests)

Le notebook reste volontairement mince : toute la logique est dans ``core/worker.py`` (testée).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPO_SLUG = "moonkamel/clipgod"
COLAB_BADGE = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/clipgod_colab.ipynb"


def md(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _lines(text)}


def code(cell_id: str, text: str, *, form: bool = False) -> dict:
    meta: dict = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "id": cell_id, "metadata": meta, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def colab_cells() -> list[dict]:
    return [
        md("intro", f"""
# 🎬 ClipGod — worker GPU (Colab)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BADGE})

Ce notebook traite les jobs déposés par le laptop (`clipgod prepare`) dans **Google Drive › `ClipGod/jobs/`** :

1. monte votre Google Drive,
2. clone ClipGod et installe les dépendances (faster-whisper, OpenCV, librosa…),
3. prend les jobs en attente, du plus ancien au plus récent,
4. exécute **scan** (transcription + détection des meilleurs moments) puis **render** (clips verticaux),
5. écrit les résultats dans `ClipGod/moments/`, `ClipGod/clips/` et marque le job dans `ClipGod/jobs_done/`.

**Avant de lancer :** *Exécution › Modifier le type d'exécution › Accélérateur matériel : **GPU T4***, puis
*Exécution › Tout exécuter* (`Ctrl+F9`). Utilisez le **même compte Google** que sur le laptop.

> Une déconnexion n'est pas grave : relancez *Tout exécuter*, le scan reprend à son dernier checkpoint.
> Ensuite, sur le laptop : `clipgod sync`.
"""),
        code("params", """
#@title ⚙️ Paramètres { display-mode: "form" }
REPO_URL = "https://github.com/moonkamel/clipgod.git"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
#@markdown Étapes : `scan+render` (normal), `scan` seul, ou `render` seul (re-rendu depuis moments/ existants) :
STAGES = "scan+render"  #@param ["scan+render", "scan", "render"]
#@markdown Voix off Piper (lit le hook au début du clip) pour tous les jobs de ce run :
VOICEOVER = False  #@param {type:"boolean"}
#@markdown Nombre max de jobs à traiter (0 = tous) et job précis (vide = file normale) :
MAX_JOBS = 0  #@param {type:"integer"}
ONLY_JOB = ""  #@param {type:"string"}
#@markdown Modèle whisper (`auto` = large-v3 sur T4, medium si VRAM < 10 Go) :
WHISPER_MODEL = "auto"  #@param ["auto", "large-v3", "medium", "small"]
#@markdown Repo privé : ajoutez un secret Colab `GITHUB_TOKEN` (icône 🔑 à gauche), lecture seule suffit.
REPO_DIR = "/content/clipgod"
WORK_DIR = "/content/clipgod_work"
""", form=True),
        code("gpu", """
#@title 1️⃣ Vérification du GPU
import subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True, check=False)
if gpu.returncode != 0:
    raise RuntimeError("❌ Aucun GPU. Exécution › Modifier le type d'exécution › GPU T4, puis relancez tout.")
print("✅ GPU :", gpu.stdout.strip())
""", form=True),
        code("mount", """
#@title 2️⃣ Montage de Google Drive
import os

from google.colab import drive as gdrive

gdrive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/ClipGod"
os.makedirs(DRIVE_ROOT, exist_ok=True)
print("✅ Drive monté :", DRIVE_ROOT)
""", form=True),
        code("clone", """
#@title 3️⃣ Récupération du code ClipGod
import os
import subprocess


def _git(*args, cwd=None):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} a échoué : {proc.stderr.strip()}")
    return proc.stdout.strip()

token = ""
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN") or ""
except Exception:  # noqa: BLE001 — secret absent ou accès refusé : repo public
    token = ""
auth_url = REPO_URL.replace("https://", f"https://x-access-token:{token}@") if token else REPO_URL

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    _git("fetch", "--depth", "1", auth_url, BRANCH, cwd=REPO_DIR)
    _git("reset", "--hard", "FETCH_HEAD", cwd=REPO_DIR)
else:
    _git("clone", "--depth", "1", "--branch", BRANCH, auth_url, REPO_DIR)
print("✅ ClipGod", BRANCH, "@", _git("rev-parse", "--short", "HEAD", cwd=REPO_DIR))
""", form=True),
        code("install", """
#@title 4️⃣ Installation des dépendances (≈ 1-2 min la première fois)
!bash /content/clipgod/scripts/colab_setup.sh {"--with-tts" if VOICEOVER else ""}
""", form=True),
        code("init", """
#@title 5️⃣ Initialisation
import os
import sys

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if "core.worker" in sys.modules:
    print("ℹ️ Code déjà importé : après une mise à jour du repo, faites Exécution › Redémarrer la session.")

from core import worker
from core.config import load_config
from core.drive import get_drive
from core.logger import setup_logging

cfg = load_config(os.path.join(REPO_DIR, "config.yaml"))
setup_logging(cfg, force=True)
drive = get_drive(cfg)          # Colab -> dossier monté /content/drive/MyDrive/ClipGod
drive.ensure_structure()
print("✅ Prêt :", drive)
""", form=True),
        code("queue", """
#@title 6️⃣ Jobs en attente
import pandas as pd

rows = worker.queue_status(drive)
if not rows:
    print("📭 Aucun job en attente. Sur le laptop : clipgod prepare --source <url>")
else:
    display(pd.DataFrame(rows))
""", form=True),
        code("run", """
#@title 7️⃣ Traitement (scan puis render)
stages = {"scan": ("scan",), "scan+render": ("scan", "render"), "render": ("render",)}[STAGES]
overrides = {}
if WHISPER_MODEL != "auto":
    overrides["scan"] = {"whisper": {"model": WHISPER_MODEL}}
if VOICEOVER:
    overrides["render"] = {"voiceover": {"enabled": True}}

results = worker.run_queue(
    drive, cfg,
    workdir=WORK_DIR,
    stages=stages,
    max_jobs=MAX_JOBS or None,
    only_job=ONLY_JOB or None,
    overrides=overrides,
)
if results:
    cols = [c for c in ("job_id", "status", "video_id", "n_moments", "best_score", "clips", "duration_s", "error")
            if any(c in r for r in results)]
    display(pd.DataFrame(results)[cols])
else:
    print("📭 Rien à traiter.")
""", form=True),
        code("preview", """
#@title 8️⃣ Aperçu des moments et du premier clip
for r in results:
    if r.get("status") != "done" or not r.get("moments"):
        continue
    doc = drive.read_json(r["moments"])
    print(f"\\n🎬 {doc['video_id']} — scorer {doc['scorer']['name']} — signaux : {', '.join(doc['signals_available'])}")
    display(pd.DataFrame([
        {"#": m["rank"], "début": m["start"], "fin": m["end"], "score": m["score"],
         "raison": m["reason"], "hooks": " | ".join(m["hooks"])}
        for m in doc["moments"]
    ]))
    clips = r.get("clips") or []
    print(f"🎞️ {len(clips)} clip(s) dans ClipGod/clips/ :", ", ".join(c.split("/")[-1] for c in clips))
    if clips:
        from base64 import b64encode

        from IPython.display import HTML
        first = drive.root / clips[0]
        data = b64encode(first.read_bytes()).decode()
        display(HTML(f'<video width=270 controls src="data:video/mp4;base64,{data}"></video>'))
""", form=True),
        code("flush", """
#@title 9️⃣ Synchroniser Drive avant de fermer
# Les écritures sur le Drive monté sont asynchrones : on force l'envoi avant la fin de session.
gdrive.flush_and_unmount()
print("✅ Résultats envoyés sur Drive. Sur le laptop : clipgod sync")
""", form=True),
    ]


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def render_json(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


TARGETS = {"notebooks/clipgod_colab.ipynb": lambda: notebook(colab_cells())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="vérifie sans écrire")
    args = parser.parse_args(argv)
    stale = []
    for rel, build in TARGETS.items():
        path = REPO / rel
        content = render_json(build())
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(rel)
        else:
            path.write_text(content, encoding="utf-8")
            print(f"écrit : {rel}")
    if stale:
        print("Notebooks pas à jour (lancez python scripts/build_notebooks.py) :", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":  # script utilitaire (pas de logique métier)
    sys.exit(main())
