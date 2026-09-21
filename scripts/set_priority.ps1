# Приоритет для процессов (и их дочерних), в командной строке которых есть подстрока
param([string]$Match, [string]$Class = "High")
$all = Get-CimInstance Win32_Process
$ids = New-Object System.Collections.Generic.HashSet[int]
$q = New-Object System.Collections.Generic.Queue[int]
foreach ($r in ($all | Where-Object { $_.CommandLine -like "*$Match*" -and $_.CommandLine -notlike "*set_priority*" })) { [void]$ids.Add($r.ProcessId); $q.Enqueue($r.ProcessId) }
while ($q.Count) { $p = $q.Dequeue(); foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $p })) { if ($ids.Add($c.ProcessId)) { $q.Enqueue($c.ProcessId) } } }
$n = 0; foreach ($id in $ids) { try { (Get-Process -Id $id).PriorityClass = $Class; $n++ } catch {} }
"$Match -> ${Class}: $n processes"
