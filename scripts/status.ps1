# Короткая сводка состояния прогона (для мониторинга по SSH)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location D:\qhnn\results
$o = Get-CimInstance Win32_OperatingSystem
$g = (& nvidia-smi -i 1 --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits) -replace " ", ""
$cpu = [math]::Round((Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object { $_.Name -notmatch "_Total" } | Measure-Object PercentProcessorTime -Average).Average)
$counts = foreach ($n in "tabular","shots","ablation","sweep","vision","vision_q12","init") {
    $c = 0
    foreach ($f in Get-ChildItem "$n.*jsonl" -EA SilentlyContinue) {
        if ($f.Name -notlike "*_history*") { $c += (Get-Content $f.FullName | Measure-Object -Line).Lines }
    }
    "$n=$c"
}
$feat = @(Get-ChildItem features\*.npz -ErrorAction SilentlyContinue).Name -join "+"
$barren = if (Test-Path barren.csv) { "done" } else { "no" }
$errs = @(Get-ChildItem *.err | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-4) -and (Select-String -Path $_.FullName -Pattern "MemoryError|BrokenProcessPool|Traceback" -Quiet) }).Name -join ","
"{0} py={1} cpu={2}% gpu(t,util,W)={3} commitFreeGB={4} | {5} | feat={6} barren={7} | newErr={8}" -f (Get-Date -Format HH:mm), @(Get-Process python -EA SilentlyContinue).Count, $cpu, $g, [math]::Round($o.FreeVirtualMemory/1MB,1), ($counts -join " "), $feat, $barren, $errs
