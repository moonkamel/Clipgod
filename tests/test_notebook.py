import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "clipgod_colab.ipynb"


def test_notebook_is_up_to_date_with_generator():
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "build_notebooks.py"), "--check"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_notebook_structure_and_code_parses():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    sources = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(line for line in "".join(cell["source"]).splitlines()
                        if not line.lstrip().startswith(("!", "%")))
        ast.parse(src)                                   # pas d'erreur de syntaxe
    assert "drive.mount" in sources["mount"]
    assert "colab_setup.sh" in sources["install"]
    assert "worker.run_queue" in sources["run"]
    assert "flush_and_unmount" in sources["flush"]


def test_notebook_has_no_secret():
    text = NB.read_text()
    for marker in ("ghp_", "github_pat_", "AIza", "ya29."):
        assert marker not in text
