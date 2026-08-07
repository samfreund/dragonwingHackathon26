#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8001,
    [string]$StorageRoot = "received"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Venv = Join-Path $PSScriptRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -m venv $Venv
}
& $Python -m pip install -q -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python -m stream_transport.server `
    --host $HostAddress --port $Port --storage-root $StorageRoot
