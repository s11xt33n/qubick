# Повысить приоритет GPU-процессов (runner'ы с --device cuda, признаки, barren и их дочерние
# процессы), чтобы они получали CPU в первую очередь и не заставляли V100 простаивать.
param([string]$Class = "High")
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$roots = $all | Where-Object { $_.CommandLine -match "--device cuda" }
$ids = New-Object System.Collections.Generic.HashSet[int]
$queue = New-Object System.Collections.Generic.Queue[int]
foreach ($r in $roots) { [void]$ids.Add($r.ProcessId); $queue.Enqueue($r.ProcessId) }
while ($queue.Count) {
    $p = $queue.Dequeue()
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $p })) {
        if ($ids.Add($c.ProcessId)) { $queue.Enqueue($c.ProcessId) }
    }
}
$n = 0
foreach ($id in $ids) { try { (Get-Process -Id $id).PriorityClass = $Class; $n++ } catch {} }
"GPU processes set to ${Class}: $n"
