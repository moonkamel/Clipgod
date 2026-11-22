"""Module 6 — monetize (laptop) : règles des campagnes de clipping et validation des clips.

Partie implémentée à l'étape 9 (utilisée par ``publish``) : **validation** d'un clip contre des règles
(durée min/max, hook dans la 1re seconde, vraie transformation — pas de simple re-upload, format vertical).
Les clips refusés sont consignés dans ``data/rejected.json`` avec leurs raisons.

À venir (étape 10) : scraping Whop Content Rewards (CPM, budget, règles par campagne), classement $/heure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import path
from core.logger import get_logger
from core.storage import JsonStore, utc_now_iso

log = get_logger(__name__)

# Transformations qui comptent comme un vrai travail éditorial (au moins 2 en plus du recadrage vertical).
EDITORIAL = ("word_subtitles", "hook_overlay", "facecam_split", "voiceover")


@dataclass
class Rules:
    min_duration_seconds: float = 7.0
    max_duration_seconds: float = 30.0
    hook_within_seconds: float = 1.0
    require_transformation: bool = True
    min_editorial_transformations: int = 2
    vertical_only: bool = True
    campaign: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Rules:
        base = dict(((cfg.get("monetize") or {}).get("rules_default")) or {})
        base.update(overrides or {})
        known = {k: v for k, v in base.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Validation:
    clip_id: str
    ok: bool
    reasons: list[str] = field(default_factory=list)


def validate_clip(meta: Mapping[str, Any], rules: Rules, clip_path: str | Path | None = None) -> Validation:
    """Vérifie un clip (métadonnées du sidecar render) contre les règles. Ne lit pas la vidéo elle-même."""
    clip_id = str(meta.get("clip_id") or (Path(clip_path).stem if clip_path else "?"))
    reasons: list[str] = []
    if clip_path is not None:
        p = Path(clip_path)
        if not p.is_file() or p.stat().st_size == 0:
            reasons.append("fichier vidéo absent ou vide")
    duration = float(meta.get("duration") or 0)
    if duration < rules.min_duration_seconds:
        reasons.append(f"trop court ({duration:.1f}s < {rules.min_duration_seconds:g}s)")
    if duration >= rules.max_duration_seconds:
        reasons.append(f"trop long ({duration:.1f}s, max strictement < {rules.max_duration_seconds:g}s)")
    hook_at = meta.get("hook_at_seconds")
    if not meta.get("hook"):
        reasons.append("aucun hook affiché")
    elif hook_at is None or float(hook_at) > rules.hook_within_seconds:
        reasons.append(f"hook après {rules.hook_within_seconds:g}s")
    transformations = set(meta.get("transformations") or [])
    if rules.require_transformation:
        editorial = transformations & set(EDITORIAL)
        if "vertical_reframe" not in transformations or len(editorial) < rules.min_editorial_transformations:
            reasons.append(
                f"transformation insuffisante (simple re-upload ?) : {sorted(transformations) or 'aucune'} — "
                f"il faut le recadrage vertical + {rules.min_editorial_transformations} parmi {', '.join(EDITORIAL)}"
            )
    if rules.vertical_only:
        w, h = int(meta.get("width") or 0), int(meta.get("height") or 0)
        if not (w and h and h > w):
            reasons.append(f"format non vertical ({w}x{h})")
    return Validation(clip_id, not reasons, reasons)


def record_rejection(cfg: Mapping[str, Any], validation: Validation, *, campaign: str | None = None) -> None:
    """Ajoute (ou met à jour) le clip dans ``data/rejected.json``."""
    store = JsonStore(path(cfg, "rejected"), lambda: {"rejected": {}})
    with store.transaction() as data:
        data["rejected"][validation.clip_id] = {
            "reasons": validation.reasons, "campaign": campaign, "rejected_at": utc_now_iso(),
        }
    log.warning("Clip %s refusé : %s", validation.clip_id, " ; ".join(validation.reasons))


def validate_many(cfg: Mapping[str, Any], clips: Sequence[tuple[str, Mapping[str, Any], Path]],
                  rules: Rules | None = None) -> tuple[list[str], list[Validation]]:
    """Valide une liste ``(clip_id, meta, chemin)``. Retourne (ids acceptés, validations refusées)."""
    rules = rules or Rules.from_config(cfg)
    accepted, rejected = [], []
    for clip_id, meta, clip_path in clips:
        result = validate_clip({"clip_id": clip_id, **meta}, rules, clip_path)
        if result.ok:
            accepted.append(clip_id)
        else:
            record_rejection(cfg, result, campaign=rules.campaign)
            rejected.append(result)
    return accepted, rejected
