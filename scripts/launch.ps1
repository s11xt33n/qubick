# Запуск runner'а отдельным процессом (переживает обрыв SSH).
# Ограничиваем потоки BLAS/OpenMP: иначе каждый процесс резервирует буферы под все 80 потоков.
param([string]$Config, [int]$Workers = 20, [string]$Device = "cpu", [string]$Tag = "", [string]$Log, [string]$Extra = "")
$py = if ($Device -eq "cuda") { "D:\qhnn\.venv\Scripts\python.exe" } else { "D:\qhnn\.venv-cpu\Scripts\python.exe" }
$tagArg = if ($Tag) { "--tag $Tag" } else { "" }
$env_ = "set PYTHONIOENCODING=utf-8&& set OPENBLAS_NUM_THREADS=1&& set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set CUDA_DEVICE_ORDER=PCI_BUS_ID&& set CUDA_VISIBLE_DEVICES=1"
$cmd = "cmd /c $env_&& $py -m qhnn.experiments.runner $Config --workers $Workers --device $Device $tagArg $Extra > D:\qhnn\results\$Log.log 2> D:\qhnn\results\$Log.err"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory='D:\qhnn'}
"$Log rc=$($r.ReturnValue) pid=$($r.ProcessId)"
