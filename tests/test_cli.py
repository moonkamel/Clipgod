"""Smoke tests du squelette CLI : chaque commande est enregistrée et documentée."""

import pytest
from typer.testing import CliRunner

from clipgod import app

COMMANDS = ["discover", "prepare", "scan", "render", "sync", "publish", "monetize", "learn", "watch", "run"]
runner = CliRunner()


def test_root_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


@pytest.mark.parametrize("cmd", COMMANDS)
def test_command_help(cmd):
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, result.output


def test_scan_and_render_are_importable_without_heavy_deps():
    # scan/render doivent s'importer sans faster-whisper/opencv installés (imports paresseux).
    from modules.render import render
    from modules.scan import scan

    assert callable(scan) and callable(render)
