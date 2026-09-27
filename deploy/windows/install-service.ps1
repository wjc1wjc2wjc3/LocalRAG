# LocalRAG Windows 部署：注册为开机自启的后台任务
# 无需额外软件（用系统自带的“计划任务”），但需**以管理员身份**运行 PowerShell。
#
# 安装：  powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1
# 指定端口： ...\install-service.ps1 -Port 8070
# 卸载：  ...\install-service.ps1 -Uninstall
param(
    [string]$InstallDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$Port = "8070",
    [string]$DbPath = "",
    [string]$AuditPath = "",
    [switch]$Uninstall
)

$taskName = "LocalRAG"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "[ok] 已卸载计划任务 $taskName"
    exit 0
}

if ([string]::IsNullOrEmpty($DbPath))    { $DbPath    = Join-Path $InstallDir "localrag.db" }
if ([string]::IsNullOrEmpty($AuditPath)) { $AuditPath = Join-Path $InstallDir "audit.jsonl" }

# 优先用虚拟环境里的 Python
$venvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $py = $venvPy } else { $py = "py.exe" }

# 注意：全局选项（--db / --audit）必须写在子命令 serve 之前
$arg = "-m localrag.cli --db `"$DbPath`" --audit `"$AuditPath`" serve --host 127.0.0.1 --port $Port"

$action   = New-ScheduledTaskAction -Execute $py -Argument $arg -WorkingDirectory $InstallDir
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "LocalRAG API (offline, auditable RAG)" -Force | Out-Null

Write-Host "[ok] 已注册计划任务 $taskName"
Write-Host "     Python : $py"
Write-Host "     工作目录: $InstallDir"
Write-Host "     服务地址: http://127.0.0.1:$Port"
Write-Host "     数据库  : $DbPath"
Write-Host "     审计日志: $AuditPath"
Write-Host ""
Write-Host "HTTP API 需要可选依赖，若未安装请先执行："
Write-Host "     $py -m pip install fastapi uvicorn"
Write-Host "手动启动任务： Start-ScheduledTask -TaskName $taskName"
