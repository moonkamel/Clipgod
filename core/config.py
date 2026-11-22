"""Chargement et validation de config.yaml.

Utilisé à l'identique par le CLI local (laptop) et par les notebooks cloud (Colab/Kaggle) :
aucune dépendance hors stdlib + PyYAML (python-dotenv est optionnel).

La configuration est un simple ``dict`` (c'est ce que reçoivent ``scan()`` et ``render()``),
enrichi d'une clé ``_meta`` : ``config_path``, ``root`` (dossier du projet) et ``runtime``
(``local`` | ``colab`` | ``kaggle``).

Ordre de priorité (le dernier gagne) :
    1. config.yaml
    2. variables d'environnement ``CLIPGOD__SECTION__CLE=valeur`` (ex. ``CLIPGOD__SCAN__WHISPER__MODEL=medium``)
    3. ``overrides`` passés à :func:`load_config` (dict imbriqué ou clés pointées ``"scan.whisper.model"``)

Les secrets ne sont JAMAIS dans config.yaml : la config ne contient que le *nom* de la variable
d'environnement, lu via :func:`get_secret`.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "CLIPGOD__"
CONFIG_ENV_VAR = "CLIPGOD_CONFIG"
DEFAULT_FILENAME = "config.yaml"
REQUIRED_SECTIONS = ("paths", "drive", "scan", "render")
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_MISSING: Any = object()


class ConfigError(Exception):
    """Configuration absente, illisible ou invalide (message destiné à l'utilisateur)."""


# ---------------------------------------------------------------------------
# Environnement d'exécution
# ---------------------------------------------------------------------------
def detect_runtime() -> str:
    """Retourne ``"colab"``, ``"kaggle"`` ou ``"local"``."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    return "local"


# ---------------------------------------------------------------------------
# Localisation + chargement
# ---------------------------------------------------------------------------
def find_config(path: str | os.PathLike | None = None) -> Path:
    """Trouve config.yaml : argument explicite > $CLIPGOD_CONFIG > ./config.yaml > racine du repo."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    elif os.environ.get(CONFIG_ENV_VAR):
        candidates.append(Path(os.environ[CONFIG_ENV_VAR]).expanduser())
    else:
        candidates.append(Path.cwd() / DEFAULT_FILENAME)
        candidates.append(Path(__file__).resolve().parents[1] / DEFAULT_FILENAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise ConfigError(f"config.yaml introuvable (cherché : {tried}). Utilisez --config <chemin>.")


def load_config(
    path: str | os.PathLike | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    load_env_file: bool = True,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Charge, fusionne et valide la configuration. Lève :class:`ConfigError` si invalide."""
    config_path = find_config(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} : YAML invalide — {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{config_path} : lecture impossible — {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} : la racine doit être un mapping YAML, pas {type(raw).__name__}.")

    if load_env_file:
        load_dotenv_file(config_path.parent / ".env")

    cfg: dict[str, Any] = copy.deepcopy(raw)
    apply_env_overrides(cfg, os.environ if environ is None else environ)
    if overrides:
        apply_overrides(cfg, overrides)

    cfg["_meta"] = {
        "config_path": str(config_path),
        "root": str(config_path.parent),
        "runtime": detect_runtime(),
    }
    validate_config(cfg)
    return cfg


def load_dotenv_file(path: Path) -> None:
    """Charge un fichier .env sans écraser les variables déjà définies. Silencieux si absent."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Repli minimal (notebook sans python-dotenv) : KEY=VALUE, # commentaires, guillemets simples/doubles.
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Fusion / overrides
# ---------------------------------------------------------------------------
def deep_merge(base: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Fusionne récursivement ``other`` dans une copie de ``base`` (les listes sont remplacées)."""
    result = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _match_key(mapping: Mapping[str, Any], segment: str) -> str:
    """Retrouve la clé existante correspondant à ``segment`` (insensible à la casse et à ``-``/``_``)."""
    wanted = _norm_key(segment)
    for existing in mapping:
        if isinstance(existing, str) and _norm_key(existing) == wanted:
            return existing
    return segment


def set_in(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """``set_in(cfg, "scan.whisper.model", "medium")`` — crée les sections intermédiaires si besoin."""
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ConfigError("Clé de configuration vide.")
    node = cfg
    for part in parts[:-1]:
        key = _match_key(node, part)
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[_match_key(node, parts[-1])] = value


def apply_overrides(cfg: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Applique des overrides : clés pointées (``"a.b": 1``) ou dict imbriqué (``{"a": {"b": 1}}``)."""
    for key, value in overrides.items():
        if "." in key:
            set_in(cfg, key, value)
        elif isinstance(value, Mapping) and isinstance(cfg.get(_match_key(cfg, key)), dict):
            existing = _match_key(cfg, key)
            cfg[existing] = deep_merge(cfg[existing], value)
        else:
            cfg[_match_key(cfg, key)] = copy.deepcopy(value)


def apply_env_overrides(cfg: dict[str, Any], environ: Mapping[str, str]) -> None:
    """``CLIPGOD__SCAN__WHISPER__MODEL=medium`` -> ``cfg["scan"]["whisper"]["model"] = "medium"``.

    La valeur est interprétée en YAML (``true``, ``12``, ``[a, b]`` sont typés).
    """
    for name, raw_value in environ.items():
        if not name.startswith(ENV_PREFIX) or name == ENV_PREFIX:
            continue
        dotted = ".".join(name[len(ENV_PREFIX):].split("__"))
        try:
            value = yaml.safe_load(raw_value) if raw_value != "" else ""
        except yaml.YAMLError:
            value = raw_value
        set_in(cfg, dotted, value)


# ---------------------------------------------------------------------------
# Accès
# ---------------------------------------------------------------------------
def get(cfg: Mapping[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """``get(cfg, "scan.whisper.model")`` ; lève :class:`ConfigError` si absent et sans ``default``."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping):
            node = _MISSING
            break
        node = node.get(_match_key(node, part), _MISSING)
        if node is _MISSING:
            break
    if node is _MISSING:
        if default is _MISSING:
            raise ConfigError(f"Clé de configuration manquante : {dotted_key}")
        return default
    return node


def project_root(cfg: Mapping[str, Any]) -> Path:
    meta = cfg.get("_meta") or {}
    return Path(meta.get("root") or Path.cwd())


def resolve_path(cfg: Mapping[str, Any], value: str | os.PathLike) -> Path:
    """Résout un chemin relatif par rapport à la racine du projet (dossier de config.yaml)."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else project_root(cfg) / path


def path(cfg: Mapping[str, Any], name: str) -> Path:
    """Chemin absolu d'une entrée de la section ``paths`` (ex. ``path(cfg, "model")``)."""
    return resolve_path(cfg, get(cfg, f"paths.{name}"))


def get_env(name: str, *, required: bool = False, hint: str | None = None) -> str | None:
    """Lit une variable d'environnement (valeur vide = absente)."""
    value = os.environ.get(name)
    if value is not None and value.strip() == "":
        value = None
    if value is None and required:
        msg = f"Variable d'environnement {name} manquante."
        if hint:
            msg += f" {hint}"
        msg += " Renseignez-la dans .env (voir .env.example)."
        raise ConfigError(msg)
    return value


def get_secret(cfg: Mapping[str, Any], env_key_path: str, *, required: bool = True) -> str | None:
    """Lit un secret dont le NOM de variable est dans la config.

    ``get_secret(cfg, "discover.env.youtube_api_key")`` -> valeur de ``$YOUTUBE_API_KEY``.
    """
    var_name = get(cfg, env_key_path)
    if not isinstance(var_name, str) or not var_name:
        raise ConfigError(f"{env_key_path} doit contenir un nom de variable d'environnement.")
    return get_env(var_name, required=required, hint=f"(référencée par {env_key_path})")


def job_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Sous-ensemble de config embarqué dans un job Drive (ce dont le notebook a besoin)."""
    keep = ("project", "scan", "render", "logging", "pipeline")
    return {k: copy.deepcopy(cfg[k]) for k in keep if k in cfg}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _num(cfg: Mapping[str, Any], key: str, errors: list[str], *, minimum: float | None = None,
         maximum: float | None = None) -> float | None:
    value = get(cfg, key, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{key} doit être un nombre (reçu {value!r}).")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{key} doit être >= {minimum} (reçu {value}).")
    if maximum is not None and value > maximum:
        errors.append(f"{key} doit être <= {maximum} (reçu {value}).")
    return float(value)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Vérifie la cohérence de la config. Rassemble toutes les erreurs avant de lever."""
    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if not isinstance(cfg.get(section), dict):
            errors.append(f"Section obligatoire manquante ou invalide : {section}")
    if errors:
        raise ConfigError("Configuration invalide :\n  - " + "\n  - ".join(errors))

    folders = get(cfg, "drive.folders", {})
    for name in ("jobs", "sources", "moments", "clips", "jobs_done"):
        if not isinstance(folders, dict) or not folders.get(name):
            errors.append(f"drive.folders.{name} est requis.")
    if not get(cfg, "drive.root_folder", None):
        errors.append("drive.root_folder est requis.")
    backend = get(cfg, "drive.backend", "auto")
    if backend not in {"auto", "api", "local"}:
        errors.append(f"drive.backend doit valoir auto | api | local (reçu {backend!r}).")

    wmin = _num(cfg, "scan.windows.min_seconds", errors, minimum=1)
    wmax = _num(cfg, "scan.windows.max_seconds", errors, minimum=1)
    if wmin is not None and wmax is not None and wmin > wmax:
        errors.append("scan.windows.min_seconds doit être <= scan.windows.max_seconds.")
    _num(cfg, "scan.windows.step_seconds", errors, minimum=0.5)
    _num(cfg, "scan.top_k_moments", errors, minimum=1)
    _num(cfg, "scan.hook_max_chars", errors, minimum=10)
    model = get(cfg, "scan.whisper.model", "auto")
    if model not in {"auto", "large-v3", "large-v2", "medium", "small", "base", "tiny", "distil-large-v3"}:
        errors.append(f"scan.whisper.model inconnu : {model!r}.")
    weights = get(cfg, "scan.default_weights", {})
    if isinstance(weights, dict):
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                errors.append(f"scan.default_weights.{key} doit être un nombre >= 0.")

    _num(cfg, "render.width", errors, minimum=16)
    _num(cfg, "render.height", errors, minimum=16)
    gp = _num(cfg, "render.layout.gameplay_ratio", errors, minimum=0, maximum=1)
    fc = _num(cfg, "render.layout.facecam_ratio", errors, minimum=0, maximum=1)
    if gp is not None and fc is not None and abs(gp + fc - 1.0) > 1e-6:
        errors.append("render.layout.gameplay_ratio + facecam_ratio doit valoir 1.0.")
    _num(cfg, "render.subtitles.y_position_ratio", errors, minimum=0, maximum=1)
    _num(cfg, "render.loudness_lufs", errors, minimum=-70, maximum=0)

    _num(cfg, "publish.max_posts_per_day", errors, minimum=1)
    rmin = _num(cfg, "monetize.rules_default.min_duration_seconds", errors, minimum=0)
    rmax = _num(cfg, "monetize.rules_default.max_duration_seconds", errors, minimum=1)
    if rmin is not None and rmax is not None and rmin > rmax:
        errors.append("monetize.rules_default : min_duration_seconds > max_duration_seconds.")

    level = str(get(cfg, "logging.level", "INFO")).upper()
    if level not in _VALID_LOG_LEVELS:
        errors.append(f"logging.level invalide : {level!r}.")

    if errors:
        raise ConfigError("Configuration invalide :\n  - " + "\n  - ".join(errors))
