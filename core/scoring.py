"""Moteur de scoring ClipGod — partagé par ``modules/scan`` (inférence) et ``modules/learn`` (entraînement).

Contenu :
- :data:`FEATURES` : liste ordonnée des signaux (contrat commun scan <-> learn <-> model.pkl).
- :func:`robust_normalize` : normalisation 0..1 par vidéo (percentiles, robuste aux outliers).
- :class:`WeightedScorer` : score 0-100 = moyenne pondérée des signaux disponibles.
- :class:`ModelScorer` : mélange score pondéré + prédiction du GradientBoostingRegressor de ``data/model.pkl``,
  avec une confiance croissante avec le nombre d'exemples d'entraînement.
- :func:`load_scorer` : charge ``model.pkl`` s'il existe (sinon pondérations par défaut).
- :func:`generate_hooks` / :func:`build_reason` : accroches < 60 caractères et explication, via templates
  locaux (aucun LLM, aucune API).

Dépendances : numpy (+ joblib seulement si un model.pkl est présent). Importable partout.

Format de ``model.pkl`` (écrit par ``learn``, lu ici) — dict joblib :
    {"format": "clipgod-model/1", "features": [...], "model": <estimator sklearn>,
     "weights": {feature: poids}, "n_samples": int, "target": "views_24h",
     "reference_log_target": [log1p(vues) triées des clips d'entraînement], "trained_at": iso}
Le modèle prédit ``log1p(vues)`` à partir des signaux normalisés d'un moment.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger

log = get_logger(__name__)

MODEL_FORMAT = "clipgod-model/1"

FEATURES: tuple[str, ...] = (
    "audio_loudness_peak",
    "audio_loudness_delta",
    "text_hype_keywords",
    "text_unfinished_then_burst",
    "text_repetition",
    "text_speech_rate",
    "visual_motion",
    "visual_scene_changes",
    "most_replayed",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "audio_loudness_peak": 0.20,
    "audio_loudness_delta": 0.10,
    "text_hype_keywords": 0.20,
    "text_unfinished_then_burst": 0.10,
    "text_repetition": 0.05,
    "text_speech_rate": 0.05,
    "visual_motion": 0.10,
    "visual_scene_changes": 0.05,
    "most_replayed": 0.20,
}

FEATURE_LABELS = {
    "fr": {
        "audio_loudness_peak": "pic audio",
        "audio_loudness_delta": "explosion sonore",
        "text_hype_keywords": "mots de hype",
        "text_unfinished_then_burst": "phrase coupée puis explosion",
        "text_repetition": "répétitions",
        "text_speech_rate": "débit de parole",
        "visual_motion": "action rapide",
        "visual_scene_changes": "changements de scène",
        "most_replayed": "most replayed",
    },
    "en": {
        "audio_loudness_peak": "audio peak",
        "audio_loudness_delta": "sound burst",
        "text_hype_keywords": "hype words",
        "text_unfinished_then_burst": "cut-off then burst",
        "text_repetition": "repetitions",
        "text_speech_rate": "fast speech",
        "visual_motion": "fast action",
        "visual_scene_changes": "scene changes",
        "most_replayed": "most replayed",
    },
}


# ---------------------------------------------------------------------------
# Normalisation / vecteurs
# ---------------------------------------------------------------------------
def robust_normalize(values: np.ndarray, low_pct: float = 5.0, high_pct: float = 99.0) -> np.ndarray:
    """Ramène ``values`` dans [0, 1] entre les percentiles ``low_pct`` et ``high_pct``.

    Un signal constant (ou vide) renvoie des zéros : il ne discrimine aucun moment.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def vectorize(signals: Mapping[str, float], features: Sequence[str] = FEATURES) -> list[float]:
    """Dict de signaux -> vecteur ordonné (signal absent = 0). Utilisé par learn pour construire X."""
    return [float(signals.get(name, 0.0) or 0.0) for name in features]


def normalize_weights(weights: Mapping[str, float], available: Iterable[str] | None = None) -> dict[str, float]:
    """Garde les poids >= 0 des signaux disponibles et les renormalise à une somme de 1."""
    allowed = set(available) if available is not None else set(weights)
    kept = {k: max(0.0, float(v)) for k, v in weights.items() if k in allowed}
    total = sum(kept.values())
    if total <= 0:
        return {k: 1.0 / len(kept) for k in kept} if kept else {}
    return {k: v / total for k, v in kept.items()}


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
class WeightedScorer:
    """Score 0-100 = 100 x moyenne pondérée des signaux normalisés (0..1).

    Les signaux indisponibles pour une vidéo (ex. pas de most-replayed sur une VOD Twitch) sont exclus et
    les poids restants renormalisés : une vidéo n'est pas pénalisée pour un signal qui n'existe pas.
    """

    name = "weighted"

    def __init__(self, weights: Mapping[str, float] | None = None):
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            merged.update({k: float(v) for k, v in weights.items()})
        self.weights = merged

    def effective_weights(self, available: Iterable[str] | None = None) -> dict[str, float]:
        return normalize_weights(self.weights, available)

    def heuristic(self, X: np.ndarray, features: Sequence[str], available: Iterable[str] | None = None) -> np.ndarray:
        w = self.effective_weights(available if available is not None else features)
        vec = np.array([w.get(f, 0.0) for f in features], dtype=np.float64)
        if X.size == 0:
            return np.zeros(len(X))
        return 100.0 * np.clip(np.asarray(X, dtype=np.float64) @ vec, 0.0, 1.0)

    def score(self, X: np.ndarray, features: Sequence[str], available: Iterable[str] | None = None) -> np.ndarray:
        return self.heuristic(X, features, available)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "weights": {k: round(v, 4) for k, v in self.weights.items()}}


class ModelScorer(WeightedScorer):
    """Mélange score heuristique et prédiction du modèle appris.

    ``score = (1 - a) * heuristique + a * percentile(prédiction vs clips historiques)``
    avec ``a = blend * min(1, n_samples / full_trust_samples)`` : peu d'historique -> peu d'influence.
    Les poids heuristiques viennent du bundle (importances du modèle) s'ils existent.
    """

    name = "model"

    def __init__(self, bundle: Mapping[str, Any], *, blend: float = 0.5, full_trust_samples: int = 50):
        super().__init__(bundle.get("weights"))
        self.bundle = bundle
        self.model = bundle["model"]
        self.features: list[str] = list(bundle.get("features") or FEATURES)
        self.n_samples = int(bundle.get("n_samples", 0))
        ref = np.asarray(bundle.get("reference_log_target") or [], dtype=np.float64)
        self.reference = np.sort(ref)
        self.alpha = float(np.clip(blend, 0.0, 1.0)) * min(1.0, self.n_samples / max(1, full_trust_samples))

    def predict_log_views(self, X: np.ndarray, features: Sequence[str]) -> np.ndarray:
        index = {f: i for i, f in enumerate(features)}
        cols = [X[:, index[f]] if f in index else np.zeros(len(X)) for f in self.features]
        return np.asarray(self.model.predict(np.column_stack(cols)), dtype=np.float64)

    def _percentile(self, preds: np.ndarray) -> np.ndarray:
        if self.reference.size:
            return 100.0 * np.searchsorted(self.reference, preds, side="right") / self.reference.size
        return 100.0 * robust_normalize(preds, 0, 100)

    def score(self, X: np.ndarray, features: Sequence[str], available: Iterable[str] | None = None) -> np.ndarray:
        heur = self.heuristic(X, features, available)
        if self.alpha <= 0 or len(X) == 0:
            return heur
        try:
            pct = self._percentile(self.predict_log_views(X, features))
        except Exception as exc:  # noqa: BLE001 — un modèle cassé ne doit jamais bloquer un scan
            log.warning("Prédiction du modèle impossible (%s) : score heuristique seul.", exc)
            return heur
        return np.clip((1.0 - self.alpha) * heur + self.alpha * pct, 0.0, 100.0)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "n_samples": self.n_samples,
            "alpha": round(self.alpha, 3),
            "target": self.bundle.get("target"),
            "trained_at": self.bundle.get("trained_at"),
        })
        return info


def _model_path(config: Mapping[str, Any]) -> Path | None:
    scan_cfg = config.get("scan") or {}
    explicit = scan_cfg.get("model_path")
    raw = explicit or (config.get("paths") or {}).get("model")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        root = (config.get("_meta") or {}).get("root")
        p = (Path(root) if root else Path.cwd()) / p
    return p


def load_scorer(config: Mapping[str, Any]) -> WeightedScorer:
    """Scorer à utiliser pour un scan : ModelScorer si model.pkl valide, sinon WeightedScorer.

    ``model.pkl`` est un pickle : ne chargez que le vôtre (produit par ``clipgod learn``).
    """
    scan_cfg = config.get("scan") or {}
    defaults = WeightedScorer(scan_cfg.get("default_weights"))
    path = _model_path(config)
    if path is None or not path.is_file():
        log.info("Pas de model.pkl : pondérations par défaut.", extra={"model_path": str(path)})
        return defaults
    try:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, Mapping) or bundle.get("format") != MODEL_FORMAT or "model" not in bundle:
            raise ValueError(f"format inattendu (attendu {MODEL_FORMAT})")
    except Exception as exc:  # noqa: BLE001
        log.warning("model.pkl ignoré (%s) : pondérations par défaut.", exc, extra={"model_path": str(path)})
        return defaults
    scorer = ModelScorer(
        bundle,
        blend=float(scan_cfg.get("model_blend", 0.5)),
        full_trust_samples=int(scan_cfg.get("model_full_trust_samples", 50)),
    )
    log.info("model.pkl chargé (%d clips, alpha=%.2f).", scorer.n_samples, scorer.alpha,
             extra={"model_path": str(path)})
    return scorer


# ---------------------------------------------------------------------------
# Explication
# ---------------------------------------------------------------------------
def contributions(signals: Mapping[str, float], weights: Mapping[str, float]) -> list[tuple[str, float]]:
    """Contribution de chaque signal (poids x valeur), triée décroissante."""
    items = [(k, float(weights.get(k, 0.0)) * float(v)) for k, v in signals.items()]
    return sorted((it for it in items if it[1] >= 0.005), key=lambda kv: kv[1], reverse=True)


def build_reason(signals: Mapping[str, float], weights: Mapping[str, float], *,
                 keywords: Sequence[str] = (), lang: str = "fr", top: int = 3) -> str:
    """``"pic audio (0.94) · mots de hype : « no way », « wtf » · most replayed (0.81)"``."""
    labels = FEATURE_LABELS.get(lang, FEATURE_LABELS["en"])
    parts = []
    for name, _ in contributions(signals, weights)[:top]:
        label = labels.get(name, name)
        if name == "text_hype_keywords" and keywords:
            quoted = ", ".join(f"« {k} »" if lang == "fr" else f'"{k}"' for k in list(dict.fromkeys(keywords))[:3])
            parts.append(f"{label} : {quoted}" if lang == "fr" else f"{label}: {quoted}")
        else:
            parts.append(f"{label} ({signals[name]:.2f})")
    return " · ".join(parts) if parts else ("signaux faibles" if lang == "fr" else "weak signals")


# ---------------------------------------------------------------------------
# Accroches (templates locaux)
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "police": ("cop", "cops", "police", "wanted", "chase", "siren", "sirens", "swat", "arrest", "flic", "flics",
               "poursuite", "stars", "étoiles"),
    "crash": ("crash", "crashed", "boom", "explode", "exploded", "explosion", "plane", "helicopter", "heli",
              "accident", "boum"),
    "fail": ("fail", "died", "dead", "dying", "oh no", "nooo", "noo", "rip", "mort", "raté"),
    "money": ("money", "bank", "heist", "million", "cash", "robbery", "braquage", "argent", "fric"),
    "funny": ("lol", "lmao", "haha", "hahaha", "funny", "bruh", "mdr", "ptdr"),
    "shock": ("no way", "wtf", "what", "omg", "oh my god", "insane", "bro", "c'est pas possible", "quoi", "putain"),
}

HOOK_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "police": ("He tried to outrun the cops… 🚔", "5 stars in 10 seconds 💀", "{streamer} vs the entire police force",
                   "The cops were NOT ready for this 🚨", "Worst getaway in GTA 6 history"),
        "crash": ("This crash is unreal 💥", "{streamer} did NOT see that coming", "GTA 6 physics are insane 😳",
                  "Wait for the crash… 💥"),
        "fail": ("This went horribly wrong 💀", "{streamer} instantly regretted this", "Biggest fail in GTA 6 😭",
                 "He was so close… 💀"),
        "money": ("The heist went completely wrong 💰", "{streamer} almost got away with it", "Easiest money in GTA 6? 💸"),
        "funny": ("I can't stop laughing at this 😂", "{streamer} lost it 😂", "GTA 6 NPCs are unhinged 😂"),
        "shock": ("Wait for it… 😳", "Nobody expected this 😳", "{streamer} couldn't believe it", "This is why GTA 6 is insane"),
        "most_replayed": ("Everyone rewatched this part 👀", "The most replayed moment of the stream 👀"),
        "visual_motion": ("The craziest GTA 6 moment yet 🔥", "Watch till the end 🔥"),
        "audio": ("{streamer} completely lost it 😱", "The loudest moment of the stream 🔊"),
        "generic": ("Wait for it… 😳", "GTA 6 is something else 🔥", "{streamer} was NOT ready for this", "Watch till the end 👀"),
        "quote": ('"{quote}" 💀', '"{quote}" 😳', '{streamer}: "{quote}"'),
    },
    "fr": {
        "police": ("Il a voulu semer les flics… 🚔", "5 étoiles en 10 secondes 💀", "{streamer} contre toute la police",
                   "La pire fuite de l'histoire de GTA 6"),
        "crash": ("Ce crash est irréel 💥", "{streamer} ne l'a pas vu venir", "La physique de GTA 6 est folle 😳"),
        "fail": ("Ça a très mal tourné 💀", "{streamer} a regretté direct", "Le plus gros fail de GTA 6 😭"),
        "money": ("Le braquage a dérapé 💰", "{streamer} a failli s'en sortir", "L'argent le plus facile de GTA 6 ? 💸"),
        "funny": ("Je n'arrête pas de rire 😂", "{streamer} est mort de rire 😂", "Les PNJ de GTA 6 sont fous 😂"),
        "shock": ("Attends la fin… 😳", "Personne ne s'attendait à ça 😳", "{streamer} n'en revient pas"),
        "most_replayed": ("Tout le monde a revu ce passage 👀", "Le moment le plus revu du live 👀"),
        "visual_motion": ("Le moment le plus fou de GTA 6 🔥", "Regarde jusqu'au bout 🔥"),
        "audio": ("{streamer} a complètement craqué 😱", "Le moment le plus fort du live 🔊"),
        "generic": ("Attends la fin… 😳", "GTA 6 c'est autre chose 🔥", "{streamer} n'était pas prêt"),
        "quote": ("« {quote} » 💀", "« {quote} » 😳", "{streamer} : « {quote} »"),
    },
}

_SIGNAL_CATEGORY = {
    "most_replayed": "most_replayed",
    "visual_motion": "visual_motion",
    "visual_scene_changes": "visual_motion",
    "audio_loudness_peak": "audio",
    "audio_loudness_delta": "audio",
}
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w']){re.escape(phrase)}(?![\w'])", text) is not None


def detect_categories(text: str) -> list[str]:
    """Catégories thématiques présentes dans le texte, de la plus à la moins représentée."""
    low = text.lower()
    counts = {cat: sum(_contains(low, kw) for kw in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    return [c for c, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if n > 0]


def _fit(text: str, max_chars: int) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    return text if 0 < len(text) <= max_chars else None


def pick_quote(text: str, keywords: Sequence[str] = (), *, max_words: int = 6, min_words: int = 2) -> str | None:
    """Courte citation percutante : la phrase la plus courte contenant un mot de hype, sinon la plus courte."""
    sentences = [s.strip(" \"'«»") for s in re.split(r"(?<=[.!?…])\s+|\n", text) if s.strip()]
    candidates = []
    for s in sentences:
        words = _WORD_RE.findall(s)
        if not min_words <= len(words) <= max_words * 2:
            continue
        if len(words) > max_words:
            s = " ".join(s.split()[:max_words]).rstrip(",;:") + "…"
        has_kw = any(_contains(s.lower(), k.lower()) for k in keywords)
        candidates.append((0 if has_kw else 1, len(s), s))
    if not candidates:
        return None
    candidates.sort()
    quote = candidates[0][2]
    return quote.upper() if candidates[0][0] == 0 and len(quote) <= 30 else quote


def is_latin_text(text: str, threshold: float = 0.9) -> bool:
    """True si les lettres du texte sont (presque) toutes latines (accents compris)."""
    import unicodedata

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    return latin / len(letters) >= threshold


def generate_hooks(
    *,
    transcript: str = "",
    signals: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
    keywords: Sequence[str] = (),
    streamer: str | None = None,
    n: int = 3,
    max_chars: int = 60,
    lang: str = "en",
    seed: str | int = 0,
) -> list[str]:
    """Génère ``n`` accroches distinctes de ``max_chars`` caractères max, déterministes pour un ``seed`` donné.

    Diversité voulue pour l'A/B test : (1) une citation du transcript, (2) un template de la catégorie
    thématique détectée, (3) un template lié au signal dominant (most replayed, action, audio), puis génériques.
    Une citation dans une autre écriture que l'alphabet latin (hindi, arabe, russe…) n'est jamais utilisée :
    les hooks visent une audience anglophone ou francophone.
    """
    templates = HOOK_TEMPLATES.get(lang, HOOK_TEMPLATES["en"])
    seed_int = int(hashlib.sha256(str(seed).encode()).hexdigest()[:12], 16)
    rng = random.Random(seed_int)
    who = (streamer or "").lstrip("@").strip() or ("He" if lang == "en" else "Il")
    hooks: list[str] = []

    def add(template: str, **fmt: str) -> bool:
        text = _fit(template.format(streamer=who, **fmt), max_chars)
        if text and text.lower() not in {h.lower() for h in hooks}:
            hooks.append(text)
            return True
        return False

    def add_from(category: str) -> None:
        options = list(templates.get(category, ()))
        rng.shuffle(options)
        for tpl in options:
            if "{quote}" in tpl:
                continue
            if add(tpl):
                return

    quote = pick_quote(transcript, keywords)
    if quote and not is_latin_text(quote):
        quote = None
    if quote:
        options = list(templates["quote"])
        rng.shuffle(options)
        for tpl in options:
            if add(tpl, quote=quote):
                break

    for cat in detect_categories(transcript)[:2]:
        if len(hooks) >= n:
            break
        add_from(cat)

    if signals and len(hooks) < n:
        for name, _ in contributions(signals, weights or DEFAULT_WEIGHTS):
            cat = _SIGNAL_CATEGORY.get(name)
            if cat:
                add_from(cat)
                break

    guard = 0
    while len(hooks) < n and guard < 10:
        add_from("generic")
        guard += 1
    return hooks[:n]


def score_to_label(score: float) -> str:
    """Étiquette lisible d'un score 0-100 (logs / rapport)."""
    if math.isnan(score):
        return "?"
    return "🔥 viral" if score >= 80 else "✅ fort" if score >= 60 else "🟡 moyen" if score >= 40 else "⚪ faible"
