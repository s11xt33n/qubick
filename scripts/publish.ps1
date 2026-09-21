# Автообновление сайта: раз в минуту экспорт результатов в JSON и отправка на VPS.
# Работает до завершения всех экспериментов (ALL DONE) + одна финальная выгрузка.
param([int]$Interval = 60)
$env:PYTHONIOENCODING = "utf-8"
Set-Location D:\qhnn
$key = "$env:USERPROFILE\.ssh\id_qhnn_publish"
$dst = "qhnn@195.133.75.140:/var/www/qhnn/data/"
$log = "D:\qhnn\results\publish.log"
while ($true) {
    & D:\qhnn\.venv-cpu\Scripts\python.exe -m qhnn.experiments.export_site --out D:\qhnn\site\data *> $null
    $files = Get-ChildItem D:\qhnn\site\data\*.json | ForEach-Object { $_.FullName }
    & scp -q -o BatchMode=yes -i $key @files $dst
    "$(Get-Date -Format HH:mm:ss) publish rc=$LASTEXITCODE" | Out-File $log -Append -Encoding utf8
    if (Select-String -Path D:\qhnn\results\run_all.log -Pattern "ALL DONE" -Quiet) { break }
    Start-Sleep $Interval
}
