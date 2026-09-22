# Остановить runner'ы, в командной строке которых есть заданная подстрока
param([string]$Match)
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='cmd.exe'" |
    Where-Object { $_.CommandLine -like "*$Match*" -and $_.CommandLine -like "*qubik.experiments.runner*" } |
    ForEach-Object { taskkill /F /T /PID $_.ProcessId 2>$null | Out-Null; "killed $($_.ProcessId)" }
