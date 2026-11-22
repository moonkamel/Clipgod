"""Logging centralisé ClipGod.

- Fichier ``logs/clipgod.log`` : une ligne JSON par événement (rotation), exploitable par jq/pandas.
- Console : rich si disponible (laptop), sinon StreamHandler simple (notebook minimal).
- Contexte structuré via ``extra={...}`` : toutes les clés non standard sont sérialisées dans le JSON.

Fonctionne à l'identique sur le laptop et dans le notebook (pas de dépendance obligatoire hors stdlib).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_LOGGER = "clipgod"
_STANDARD_ATTRS = set(vars(logging.LogRecord("x", logging.INFO, "", 0, "", (), None))) | {"message", "asctime"}
_configured = False


class JsonFormatter(logging.Formatter):
    """Formate chaque record en une ligne JSON (ts UTC ISO-8601, level, logger, msg + extras)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ExtrasTextFormatter(logging.Formatter):
    """Format texte lisible ; ajoute les extras en ``clé=valeur`` à la fin."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and not k.startswith("_")}
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


def _console_handler(level: int) -> logging.Handler:
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=level, show_path=False, rich_tracebacks=True, markup=False, log_time_format="%H:%M:%S"
        )
        handler.setFormatter(_ExtrasTextFormatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(_ExtrasTextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    return handler


def setup_logging(
    config: Mapping[str, Any] | None = None,
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    json_format: bool | None = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure le logger ``clipgod`` (idempotent sauf ``force=True``).

    Les paramètres explicites priment sur la section ``logging`` de la config.
    Un ``log_file`` relatif est résolu depuis la racine du projet (``config["_meta"]["root"]``).
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER)
    log_cfg: Mapping[str, Any] = (config or {}).get("logging") or {}

    resolved_level = level if level is not None else log_cfg.get("level", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = logging.getLevelName(resolved_level.upper())
        if not isinstance(resolved_level, int):
            resolved_level = logging.INFO

    if _configured and not force:
        logger.setLevel(resolved_level)
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(resolved_level)
    logger.propagate = False

    if console:
        logger.addHandler(_console_handler(resolved_level))

    file_value = log_file if log_file is not None else log_cfg.get("file", "logs/clipgod.log")
    if file_value:
        file_path = Path(file_value).expanduser()
        if not file_path.is_absolute():
            root = ((config or {}).get("_meta") or {}).get("root")
            file_path = (Path(root) if root else Path.cwd()) / file_path
        use_json = json_format if json_format is not None else bool(log_cfg.get("json", True))
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=int(float(log_cfg.get("rotate_mb", 10)) * 1024 * 1024),
                backupCount=int(log_cfg.get("backups", 5)),
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                JsonFormatter()
                if use_json
                else _ExtrasTextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
            )
            logger.addHandler(fh)
        except OSError as exc:  # disque en lecture seule, quota… : on continue en console seule
            logger.warning("Fichier de log indisponible (%s) : logs console uniquement.", exc)

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """``get_logger("scan")`` -> logger ``clipgod.scan`` (hérite des handlers de ``clipgod``)."""
    if not name or name == ROOT_LOGGER:
        return logging.getLogger(ROOT_LOGGER)
    if name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    # "modules.scan" / "core.drive" (passés via __name__) -> "clipgod.scan" / "clipgod.drive"
    if name.startswith(("modules.", "core.")):
        name = name.split(".", 1)[1]
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def safe_extra(**context: Any) -> dict[str, Any]:
    """Rend un dict utilisable comme ``extra=`` (renomme les clés réservées de LogRecord)."""
    return {(f"ctx_{k}" if k in _STANDARD_ATTRS else k): v for k, v in context.items()}


@contextmanager
def log_step(logger: logging.Logger, step: str, **context: Any) -> Iterator[dict[str, Any]]:
    """Journalise début / fin / erreur d'une étape avec sa durée.

    >>> with log_step(log, "transcribe", video_id="abc") as ctx:
    ...     ctx["segments"] = 42        # ajouté au log de fin
    """
    extra_out: dict[str, Any] = {}
    start = time.perf_counter()
    logger.info("▶ %s", step, extra=safe_extra(step=step, event="start", **context))
    try:
        yield extra_out
    except Exception:
        logger.exception(
            "✖ %s", step,
            extra=safe_extra(step=step, event="error", duration_s=round(time.perf_counter() - start, 3), **context),
        )
        raise
    logger.info(
        "✔ %s", step,
        extra=safe_extra(step=step, event="done", duration_s=round(time.perf_counter() - start, 3),
                         **context, **extra_out),
    )
