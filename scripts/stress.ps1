# Короткий стресс-тест сервера (~5-7 минут) с мониторингом температуры и загрузки.
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES = "1"; $env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results\stress | Out-Null

# мониторинг: GPU (V100) и CPU каждые 5 с
$mon = Start-Job -ScriptBlock {
    param($dir)
    "time,gpu_temp,gpu_util,gpu_power,gpu_mem,cpu_load" | Out-File "$dir\monitor.csv" -Encoding utf8
    while ($true) {
        $g = (& nvidia-smi -i 1 --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits) -replace " ", ""
        $c = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average)
        "$(Get-Date -Format HH:mm:ss),$g,$c" | Out-File "$dir\monitor.csv" -Append -Encoding utf8
        Start-Sleep 5
    }
} -ArgumentList (Resolve-Path results\stress).Path

$t0 = Get-Date
"== 1. speed (CPU vs V100 vs PennyLane)"
& $py -m qubik.experiments.speed --devices cpu cuda --qubits 2 4 6 8 10 12 14 16 --pl-max-qubits 12 --repeats 3 --out results\stress\speed.csv
"== 2. CPU x60 + GPU barren одновременно"
$gpu = Start-Process -FilePath $py -ArgumentList "-m qubik.experiments.barren --device cuda --qubits 12 14 16 --layers 5 20 --samples 1000 --out results\stress\barren.csv" -NoNewWindow -PassThru -RedirectStandardOutput results\stress\barren.log
& $py -m qubik.experiments.runner configs\ablation.yaml --workers 60 --limit 180
$gpu.WaitForExit()
Get-Content results\stress\barren.log
Stop-Job $mon; Remove-Job $mon
"== всего: $([math]::Round(((Get-Date) - $t0).TotalSeconds)) с"
