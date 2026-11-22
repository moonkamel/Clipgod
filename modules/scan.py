"""Module 3 — scan (CLOUD GPU).

RAPPEL : ce module doit rester importable tel quel depuis le notebook ET depuis le CLI.
Pas de `if __name__ == "__main__"`, pas d'effet de bord à l'import, imports lourds
(faster_whisper, librosa, cv2) faits paresseusement dans les fonctions.

À implémenter (étape 3).
"""

from __future__ import annotations


def scan(video_path: str, output_dir: str, config: dict) -> list[dict]:
    """Analyse multi-signaux d'une vidéo -> liste de moments {start, end, score, hooks, reason, signals}."""
    raise NotImplementedError("scan : à implémenter (étape 3).")
