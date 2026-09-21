# Полная серия экспериментов на сервере (2x Xeon E5-2698 v4 + Tesla V100).
# powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-Workers 36]
param([int]$Workers = 36)
$ErrorActionPreference = "Continue"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results | Out-Null

function Log($msg) { "$(Get-Date -Format 'HH:mm:ss') $msg" | Tee-Object -FilePath results\run_all.log -Append }

Log "features (GPU)"
& $py -m qhnn.features mnist fashion pneumonia --device cuda --n-train 10000 --n-test 2000 *>> results\features.log

Log "speed"
& $py -m qhnn.experiments.speed --devices cpu cuda --qubits 2 4 6 8 10 12 14 *>> results\speed.log
Log "barren (GPU)"
& $py -m qhnn.experiments.barren --device cuda --qubits 2 4 6 8 10 12 14 --layers 1 5 20 --samples 2000 *>> results\barren.log

foreach ($cfg in "tabular", "shots", "ablation", "sweep", "vision") {
    Log "runner $cfg"
    & $py -m qhnn.experiments.runner "configs\$cfg.yaml" --workers $Workers *>> "results\$cfg.log"
}
Log "analyze"
& $py -m qhnn.experiments.analyze *>> results\analyze.log
Log "done"
