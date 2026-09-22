# Сколько процессов экспериментов ещё работает (runner'ы и barren). 0 — всё досчитано.
$n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "qubik\.experiments\.(runner|barren)" -and $_.CommandLine -notmatch "multiprocessing" }).Count
"ACTIVE_JOBS=$n"
