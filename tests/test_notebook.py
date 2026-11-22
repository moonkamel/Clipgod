import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "clipgod_colab.ipynb"
NB_KAGGLE = REPO / "notebooks" / "clipgod_kaggle.ipynb"


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


def test_notebooks_have_no_secret():
    for nb in (NB, NB_KAGGLE):
        text = nb.read_text()
        for marker in ("ghp_", "github_pat_", "AIza", "ya29.", "refresh_token"):
            assert marker not in text


def _sources(path):
    nb = json.loads(path.read_text())
    return nb, {c["id"]: "".join(c["source"]) for c in nb["cells"]}


def test_kaggle_notebook_structure():
    nb, src = _sources(NB_KAGGLE)
    assert nb["metadata"]["kaggle"]["accelerator"] == "nvidiaTeslaT4"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is True
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code = "\n".join(line for line in "".join(cell["source"]).splitlines()
                             if not line.lstrip().startswith(("!", "%")))
            ast.parse(code)
            assert "#@param" not in code and "google.colab" not in code
    assert "CLIPGOD_GDRIVE_TOKEN_JSON" in src["secrets"] and "UserSecretsClient" in src["secrets"]
    assert "token = GITHUB_TOKEN" in src["clone"]
    assert src["run"].split("\n", 1)[1] == _sources(NB)[1]["run"].split("\n", 1)[1]   # même logique que Colab
    assert "/kaggle/working" in src["params"] and 'WORK_DIR = "/tmp' in src["params"]


def test_clone_cells_never_persist_token():
    for path in (NB, NB_KAGGLE):
        clone = _sources(path)[1]["clone"]
        assert 'set-url", "origin", REPO_URL' in clone and 'replace(token, "***")' in clone
