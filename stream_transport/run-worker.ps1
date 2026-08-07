#Requires -Version 5.1
<#
    Runs the phone-query worker.

    Deliberately a separate script from run.ps1, and deliberately a different
    interpreter. The server needs only websockets and lives in
    stream_transport\.venv; the worker loads ask.py, which pulls in the query
    router and the NPU reader, and those only exist in npu_qa\.venv. Running
    the worker on the server's venv gets as far as constructing HybridQA and
    then dies at the first question with ModuleNotFoundError: numpy.

    Both processes must be running for a phone query to be answered. Without
    the worker, queries are accepted and stored but sit at "pending" forever.
#>
[CmdletBinding()]
param(
    [string]$StorageRoot = "received",
    [double]$PollSeconds = 0.25
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $RepoRoot "npu_qa\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Expected the NPU virtualenv at $Python. Create it before running the worker; " +
          "ask.py cannot load its router or reader without it."
}

Set-Location $RepoRoot
& $Python -m stream_transport.laptop_worker `
    --storage-root $StorageRoot --poll-seconds $PollSeconds
