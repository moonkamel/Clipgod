<#
  ClipGod — routine quotidienne « Top 5 GTA 6 du jour » (Windows / PowerShell).

  1. met à jour ClipGod (git pull) ;
  2. récupère les clips traités la veille sur Colab (clipgod sync) ;
  3. monte le best-of du jour s'il y a assez de nouveaux moments (clipgod bestof) ;
  4. repère ce qui buzze et prépare les VODs d'origine pour le prochain passage Colab
     (clipgod discover --trending --prepare N, téléchargement fait par le notebook) ;
  5. ouvre Colab : il ne reste qu'à cliquer sur Exécution > Tout exécuter.

  Lancement manuel :   powershell -ExecutionPolicy Bypass -File scripts\daily.ps1
  Tâche planifiée :    powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
  Journal :            logs\daily_<date>.txt
#>
param(
    [int]$Prepare = 2,          # VODs d'origine à préparer chaque jour
    [switch]$NoColab,           # ne pas ouvrir Colab à la fin
    [switch]$NoPull             # ne pas faire git pull
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$log = Join-Path "logs" ("daily_{0}.txt" -f (Get-Date -Format "yyyy-MM-dd"))
Start-Transcript -Path $log -Append | Out-Null

try {
    $env:PYTHONIOENCODING = "utf-8"
    $activate = Join-Path $root ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) { throw "Environnement .venv introuvable : suivez docs\INSTALLATION.md." }
    . $activate

    Write-Host "`n=== 1/4 Mise à jour ===" -ForegroundColor Cyan
    if (-not $NoPull) { git pull --ff-only }

    Write-Host "`n=== 2/4 Clips de la veille (Drive -> PC) ===" -ForegroundColor Cyan
    clipgod sync

    Write-Host "`n=== 3/4 Top 5 du jour ===" -ForegroundColor Cyan
    clipgod bestof --no-open
    if ($LASTEXITCODE -ne 0) { Write-Host "Pas de best-of aujourd'hui (pas assez de nouveaux moments)." -ForegroundColor Yellow }
    else { Start-Process (Join-Path $root "output\bestof") }

    Write-Host "`n=== 4/4 Ce qui buzze -> VODs à traiter ===" -ForegroundColor Cyan
    clipgod discover --trending --prepare $Prepare

    if (-not $NoColab) {
        $url = python -c "from core.config import load_config; from modules.prepare import colab_url; print(colab_url(load_config(None)))"
        if ($url) {
            Write-Host "`nColab s'ouvre : Exécution > Tout exécuter (reste sur l'onglet jusqu'à la fin)." -ForegroundColor Green
            Start-Process $url
        }
    }
}
catch {
    Write-Host "Erreur : $_" -ForegroundColor Red
}
finally {
    Stop-Transcript | Out-Null
}
