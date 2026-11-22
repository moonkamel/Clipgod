"""Suivi du quota YouTube Data API (10 000 unités/jour, remis à zéro à minuit heure du Pacifique).

Partagé par ``discover`` (search.list = 100 unités, videos/channels/playlistItems.list = 1) et ``publish``
(videos.insert = 1600 unités). Persisté dans ``data/state.json`` -> survit aux redémarrages du CLI.

    quota = YouTubeQuota.from_config(cfg)
    if quota.can_spend(100): ...; quota.spend(100, "search.list")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.logger import get_logger
from core.storage import StateStore

log = get_logger(__name__)
PACIFIC = ZoneInfo("America/Los_Angeles")

COSTS = {"search.list": 100, "videos.list": 1, "channels.list": 1, "playlistItems.list": 1, "videos.insert": 1600}


class QuotaExceeded(RuntimeError):
    """Le quota YouTube du jour ne permet pas l'appel demandé."""


class YouTubeQuota:
    def __init__(self, state_path: str | Path, *, daily_limit: int = 10000, safety_margin: int = 0,
                 now: Callable[[], datetime] | None = None):
        self.state = StateStore(state_path)
        self.daily_limit = int(daily_limit)
        self.safety_margin = int(safety_margin)
        self._now = now or (lambda: datetime.now(PACIFIC))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> YouTubeQuota:
        from core.config import get, path

        return cls(path(cfg, "state"), daily_limit=int(get(cfg, "publish.youtube_quota.daily_units", 10000)),
                   safety_margin=int(get(cfg, "publish.youtube_quota.safety_margin", 0)))

    def today(self) -> str:
        return self._now().astimezone(PACIFIC).strftime("%Y-%m-%d")

    def _entry(self, data: dict[str, Any]) -> dict[str, Any]:
        entry = data.get("youtube_quota") or {}
        if entry.get("day") != self.today():
            entry = {"day": self.today(), "used": 0, "by_kind": {}}
        return entry

    def used(self) -> int:
        return int(self._entry(self.state.load())["used"])

    def remaining(self) -> int:
        return max(0, self.daily_limit - self.safety_margin - self.used())

    def can_spend(self, units: int) -> bool:
        return units <= self.remaining()

    def spend(self, units: int, kind: str) -> int:
        """Enregistre une consommation. Lève :class:`QuotaExceeded` si elle dépasse le reste du jour."""
        with self.state.transaction() as data:
            entry = self._entry(data)
            if units > self.daily_limit - self.safety_margin - int(entry["used"]):
                raise QuotaExceeded(f"Quota YouTube insuffisant pour {kind} ({units} u) : reste "
                                    f"{max(0, self.daily_limit - self.safety_margin - int(entry['used']))} u "
                                    "(remise à zéro à minuit, heure du Pacifique).")
            entry["used"] = int(entry["used"]) + int(units)
            entry["by_kind"][kind] = int(entry["by_kind"].get(kind, 0)) + int(units)
            data["youtube_quota"] = entry
            return int(entry["used"])
