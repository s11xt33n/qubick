# Автообновление сайта: каждую минуту — JSON с результатами, каждые 10 минут — PNG-графики.
param([int]$Interval = 60)
$env:PYTHONIOENCODING = "utf-8"; $env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"; $env:QHNN_HIGH_PRIORITY = "1"
Set-Location D:\qhnn
$py = "D:\qhnn\.venv-cpu\Scripts\python.exe"
$key = "$env:USERPROFILE\.ssh\id_qhnn_publish"
$vps = "qhnn@195.133.75.140"
$log = "D:\qhnn\results\publish.log"
$i = 0
while ($true) {
    if ($i % 10 -eq 0) { & $py -m qhnn.experiments.analyze *> D:\qhnn\results\analyze.log }
    & $py -m qhnn.experiments.export_site --out D:\qhnn\site\data *> $null
    $json = Get-ChildItem D:\qhnn\site\data\*.json | ForEach-Object { $_.FullName }
    & scp -q -o BatchMode=yes -i $key @json "${vps}:/var/www/qhnn/data/"
    $rc = $LASTEXITCODE
    if ($i % 10 -eq 0) {
        $png = Get-ChildItem D:\qhnn\site\figures\*.png -EA SilentlyContinue | ForEach-Object { $_.FullName }
        if ($png) { & scp -q -o BatchMode=yes -i $key @png "${vps}:/var/www/qhnn/figures/" }
    }
    "$(Get-Date -Format HH:mm:ss) publish rc=$rc" | Out-File $log -Append -Encoding utf8
    $i++
    Start-Sleep $Interval
}
