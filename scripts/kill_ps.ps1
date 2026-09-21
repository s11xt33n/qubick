# Остановить процессы PowerShell, в командной строке которых есть подстрока (кроме самого себя)
param([string]$Match)
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*$Match*" -and $_.ProcessId -ne $PID -and $_.CommandLine -notlike "*kill_ps.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId)" }
