# Полная серия экспериментов на сервере — все 80 потоков CPU + Tesla V100.
#   .venv-cpu — PyTorch без CUDA для CPU-процессов
#   .venv     — PyTorch + CUDA 12.6 для задач на V100
# Все серии можно перезапускать: посчитанные запуски пропускаются.
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES = "1"; $env:PYTHONIOENCODING = "utf-8"
Set-Location D:\qhnn
$cpu_py = "D:\qhnn\.venv-cpu\Scripts\python.exe"
$gpu_py = "D:\qhnn\.venv\Scripts\python.exe"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File results\run_all.log -Append -Encoding utf8 }
function Run($py, $a, $log) {
    Start-Process $py -ArgumentList $a -NoNewWindow -PassThru `
        -RedirectStandardOutput "results\$log.log" -RedirectStandardError "results\$log.err"
}
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File D:\qhnn\scripts\monitor.ps1"
$base = "configs\tabular.yaml configs\shots.yaml configs\ablation.yaml configs\sweep.yaml"

Log "START: CPU x60 (tabular/shots/ablation/sweep<=8q) + GPU x6 (sweep 10-12q) + GPU features"
$cpu = Run $cpu_py "-m qubik.experiments.runner $base --workers 60 --max-qubits 8" "main_cpu"
$gpu = Run $gpu_py "-m qubik.experiments.runner $base --workers 6 --min-qubits 10 --device cuda --tag .gpu" "main_gpu"
& $gpu_py -m qubik.features mnist fashion pneumonia breast --device cuda --n-train 10000 --n-test 2000 *> results\features.log
Log "features done -> vision CPU x20"
$vis = Run $cpu_py "-m qubik.experiments.runner configs\vision.yaml --workers 20" "vision"
Log "barren (GPU)"
& $gpu_py -m qubik.experiments.barren --device cuda --qubits 2 4 6 8 10 12 14 16 --layers 1 5 20 50 --samples 2000 *> results\barren.log
Log "barren done"
$gpu.WaitForExit(); Log "GPU runner done"
$cpu.WaitForExit(); Log "CPU runner done"
# освободившиеся ядра — на досчёт vision
$vis2 = Run $cpu_py "-m qubik.experiments.runner configs\vision.yaml --workers 40 --tag .b" "vision2"
$vis.WaitForExit(); $vis2.WaitForExit(); Log "vision done"
Log "speed (на свободной машине)"
& $gpu_py -m qubik.experiments.speed --devices cpu cuda --qubits 2 4 6 8 10 12 14 16 --pl-max-qubits 12 *> results\speed.log
& $gpu_py -m qubik.experiments.analyze *> results\analyze.log
Log "ALL DONE"
