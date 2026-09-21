# Последняя строка и первые ошибки каждого активного лога
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Set-Location D:\qhnn\results
foreach ($f in "vision_q12", "init", "vision_e", "vision_d", "vision", "main_cpu", "main_gpu", "barren_init") {
    if (-not (Test-Path "$f.log")) { continue }
    $last = Get-Content "$f.log" -Tail 1 -EA SilentlyContinue
    $err = Select-String -Path "$f.log", "$f.err" -Pattern "OpenBLAS|MemoryError|BrokenProcessPool|Traceback" -EA SilentlyContinue | Select-Object -First 1
    $e = if ($err) { " | ERR: " + $err.Line.Substring(0, [Math]::Min(90, $err.Line.Length)) } else { "" }
    "{0,-12} {1}{2}" -f $f, ($last -replace '\s+', ' ').Substring(0, [Math]::Min(110, ($last -replace '\s+', ' ').Length)), $e
}
