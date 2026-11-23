<#
  Installe (ou remplace) la tâche planifiée Windows « ClipGod quotidien ».
  Usage : powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 [-At 09:15] [-Prepare 2]
  Suppression : Unregister-ScheduledTask -TaskName "ClipGod quotidien" -Confirm:$false
#>
param(
    [string]$At = "09:15",
    [int]$Prepare = 2
)
$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\daily.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Prepare $Prepare" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
# StartWhenAvailable : si le PC était éteint à l'heure prévue, la tâche part au démarrage suivant.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "ClipGod quotidien" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Sync, Top 5 GTA 6 du jour, tendances et préparation des VODs, puis ouverture de Colab." -Force | Out-Null
Write-Host "Tâche « ClipGod quotidien » installée : tous les jours à $At." -ForegroundColor Green
Write-Host "Tester maintenant : Start-ScheduledTask -TaskName 'ClipGod quotidien'"
