import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Copie config.yaml dans un projet temporaire isolé (pas de .env, pas d'env CLIPGOD__)."""
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    for name in list(__import__("os").environ):
        if name.startswith("CLIPGOD"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    """Chaque test logge dans son tmp_path, jamais dans logs/ du dépôt."""
    import logging

    from core import logger as L

    monkeypatch.setenv("CLIPGOD__LOGGING__FILE", str(tmp_path / "test.log"))
    L._configured = False
    yield
    root = logging.getLogger(L.ROOT_LOGGER)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    L._configured = False
