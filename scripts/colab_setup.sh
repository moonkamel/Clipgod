#!/usr/bin/env bash
# =============================================================================
#  ClipGod — installation des dépendances cloud (Colab / Kaggle, GPU T4)
#
#  Usage (depuis la racine du repo cloné) :   bash scripts/colab_setup.sh [--with-tts] [--quiet]
#
#  - idempotent : relançable sans risque (rapide si tout est déjà installé) ;
#  - n'installe QUE ce qui manque au runtime : Colab fournit déjà torch, numpy, pandas, sklearn… ;
#  - Piper (voix off) est optionnel et non bloquant : son échec n'empêche pas scan/render.
# =============================================================================
set -euo pipefail

WITH_TTS=0
PIP_QUIET="-q"
for arg in "$@"; do
  case "$arg" in
    --with-tts) WITH_TTS=1 ;;
    --verbose)  PIP_QUIET="" ;;
    --quiet)    PIP_QUIET="-q" ;;
    *) echo "Option inconnue : $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }

step "Environnement"
python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU : /'
else
  warn "Aucun GPU détecté. Colab : Exécution > Modifier le type d'exécution > GPU T4."
fi

step "ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null
fi
ok "$(ffmpeg -version | head -n1)"

step "Polices (sous-titres dans toutes les écritures)"
# Montserrat n'a que l'alphabet latin : sans ces polices, le hindi, l'arabe, le russe… s'affichent en carrés.
# libass bascule automatiquement sur Noto pour chaque caractère absent de Montserrat.
if ! fc-list 2>/dev/null | grep -qi "NotoSansDevanagari"; then
  { apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y fonts-noto-core >/dev/null 2>&1; } || warn "fonts-noto-core indisponible : écritures non latines en carrés."
fi
fc-list 2>/dev/null | grep -qi "NotoSansDevanagari" && ok "polices Noto installées (devanagari, arabe, cyrillique…)"

step "Paquets Python (ClipGod + extra [cloud])"
# Bornes minimales seulement : pip garde les versions déjà fournies par Colab (torch, numpy, pandas…).
python3 -m pip install $PIP_QUIET --upgrade pip >/dev/null
python3 -m pip install $PIP_QUIET -e ".[cloud]"
ok "clipgod installé en mode éditable"

if [[ "$WITH_TTS" == "1" ]]; then
  step "Piper TTS (optionnel)"
  if python3 -m pip install $PIP_QUIET -e ".[tts]"; then ok "piper-tts installé"
  else warn "piper-tts indisponible pour ce Python : la voix off sera désactivée."; fi
fi

step "Vérification"
python3 - <<'PY'
import importlib, sys
ok = True
for mod in ("faster_whisper", "ctranslate2", "cv2", "librosa", "soundfile", "yt_dlp", "core.worker", "modules.scan"):
    try:
        m = importlib.import_module(mod)
        print(f"  ✔ {mod} {getattr(m, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ✖ {mod} : {exc}")
try:
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    print(f"  ✔ CUDA visible par ctranslate2 : {n} GPU" if n else "  ⚠ ctranslate2 ne voit aucun GPU (CPU = lent)")
except Exception as exc:  # noqa: BLE001
    print(f"  ⚠ ctranslate2 CUDA : {exc}")
sys.exit(0 if ok else 1)
PY
ok "Installation terminée"
