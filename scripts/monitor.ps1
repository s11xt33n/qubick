# Журнал нагрузки сервера каждые 15 с: results\monitor.csv (для сайта и итогов прогона)
$f = "D:\qhnn\results\monitor.csv"
if (-not (Test-Path $f)) { "time,gpu_temp,gpu_util,gpu_power,gpu_mem,cpu_load,python_procs" | Out-File $f -Encoding utf8 }
while ($true) {
    $g = (& nvidia-smi -i 1 --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits) -replace " ", ""
    $c = [math]::Round((Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object { $_.Name -notmatch "_Total" } | Measure-Object PercentProcessorTime -Average).Average)
    $n = @(Get-Process python -ErrorAction SilentlyContinue).Count
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$g,$c,$n" | Out-File $f -Append -Encoding utf8
    Start-Sleep 15
}
