# Финализация прогона на сервере: замер скорости на свободной машине, финальный анализ,
# последняя выгрузка на сайт, остановка служебных циклов. Лог: results\finalize.log
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"; $env:OPENBLAS_NUM_THREADS = "1"; $env:QHNN_HIGH_PRIORITY = "1"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES = "1"
Set-Location D:\qhnn
$log = "D:\qhnn\results\finalize.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
$gpu = "D:\qhnn\.venv\Scripts\python.exe"; $cpu = "D:\qhnn\.venv-cpu\Scripts\python.exe"

Log "stop publish loop"
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*scripts\publish.ps1*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Log "speed benchmark"
& $gpu -m qubik.experiments.speed --devices cpu cuda --qubits 2 4 6 8 10 12 14 16 --pl-max-qubits 12 --out D:\qhnn\results\speed.csv *> D:\qhnn\results\speed.log
Log "analyze"
& $cpu -m qubik.experiments.analyze *> D:\qhnn\results\analyze.log
Log "export"
& $cpu -m qubik.experiments.export_site --out D:\qhnn\site\data *> D:\qhnn\results\export.log

Log "final publish"
$key = "$env:USERPROFILE\.ssh\id_qhnn_publish"; $vps = "qhnn@195.133.75.140"
$json = Get-ChildItem D:\qhnn\site\data\*.json | ForEach-Object { $_.FullName }
& scp -q -o BatchMode=yes -i $key @json "${vps}:/var/www/qhnn/data/"
$png = Get-ChildItem D:\qhnn\site\figures\*.png -EA SilentlyContinue | ForEach-Object { $_.FullName }
if ($png) { & scp -q -o BatchMode=yes -i $key @png "${vps}:/var/www/qhnn/figures/" }
Log "publish rc=$LASTEXITCODE"

Log "stop monitor"
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*scripts\monitor.ps1*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Log "FINALIZE DONE"
