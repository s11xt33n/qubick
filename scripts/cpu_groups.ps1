$p = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object { $_.Name -notmatch '_Total' }
"instances: " + $p.Count
$p | Group-Object { $_.Name.Split(',')[0] } | ForEach-Object {
    "group {0}: {1} LPs, avg {2}%" -f $_.Name, $_.Count, [math]::Round(($_.Group | Measure-Object PercentProcessorTime -Average).Average)
}
