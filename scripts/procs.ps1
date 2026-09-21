[Console]::OutputEncoding = [Text.Encoding]::UTF8
Get-Date -Format HH:mm:ss
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" | ForEach-Object {
    $c = $_.CommandLine -replace '^.*?(python|powershell)(\.exe)?"?\s*', ''
    $p = Get-Process -Id $_.ProcessId -EA SilentlyContinue
    "{0,6} par={1,6} cpu={2,7:N0}s  {3}" -f $_.ProcessId, $_.ParentProcessId, $p.CPU, $c.Substring(0, [Math]::Min(70, $c.Length))
}
"--- run_all.log"; Get-Content D:\qhnn\results\run_all.log -EA SilentlyContinue
