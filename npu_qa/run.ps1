#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper for the NPU QA sample -- arch-aware.

.DESCRIPTION
    onnxruntime-qnn (the Hexagon NPU execution provider) only has a
    win-arm64 wheel. An emulated x86 Python running under Prism on ARM64
    Windows reports machine()=AMD64 and cannot install or reach the NPU
    with it. On ARM64 hardware this script finds (or recreates) a venv
    built from a NATIVE arm64 Python interpreter; elsewhere it falls back
    to whatever Python is on PATH (CPU-only -- use --cpu with npu_qa.py).

.EXAMPLE
    .\run.ps1
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg) { Write-Host "  [INFO]  $msg" }
function Write-Warn($msg) { Write-Host "  [WARN]  $msg" -ForegroundColor Yellow }

$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
$VenvDir = Join-Path $PSScriptRoot ".venv"

if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found: $ReqFile"
    exit 1
}

function Get-InterpreterMachine([string]$PythonExe) {
    try {
        return (& $PythonExe -c "import platform; print(platform.machine())" 2>$null).Trim()
    } catch {
        return $null
    }
}

# WMI/CIM, not [RuntimeInformation]::OSArchitecture -- the latter reflects the
# CURRENT process's own emulation state (and is flaky on PS 5.1/ARM64 CLRs),
# not the actual host CPU. Win32_Processor.Architecture 12 == ARM64.
$hostIsArm64 = (Get-CimInstance -ClassName Win32_Processor |
    Select-Object -First 1 -ExpandProperty Architecture) -eq 12

$pythonExe = $null

if ($hostIsArm64) {
    Write-Info "ARM64 host detected -- looking for a native arm64 Python ..."

    # 1. Ask the py launcher for an explicitly arm64-tagged interpreter.
    try {
        $pyList = & py -0p 2>$null
        foreach ($line in $pyList) {
            if ($line -match "-arm64\s+\*?\s*(\S+\.exe)") {
                $candidate = $Matches[1]
                if ((Get-InterpreterMachine $candidate) -eq "ARM64") {
                    $pythonExe = $candidate
                    break
                }
            }
        }
    } catch { }

    # 2. Fall back to scanning common per-user / per-machine install roots.
    if (-not $pythonExe) {
        $roots = @(
            "$env:LOCALAPPDATA\Programs\Python",
            "$env:ProgramFiles"
        )
        foreach ($root in $roots) {
            if (-not (Test-Path $root)) { continue }
            Get-ChildItem $root -Directory -Filter "Python3*-arm64" -ErrorAction SilentlyContinue |
                ForEach-Object {
                    if (-not $pythonExe) {
                        $candidate = Join-Path $_.FullName "python.exe"
                        if ((Test-Path $candidate) -and (Get-InterpreterMachine $candidate) -eq "ARM64") {
                            $pythonExe = $candidate
                        }
                    }
                }
        }
    }

    if (-not $pythonExe) {
        Write-Error @"
No native arm64 Python interpreter found, but this is an ARM64 host.
onnxruntime-qnn (the NPU execution provider) requires one.

Install one from https://www.python.org/downloads/windows/ (pick the
"Windows arm64" installer, not the regular x86-64/embeddable one), then
re-run this script.
"@
        exit 1
    }
    Write-Ok "native arm64 Python: $pythonExe"
} else {
    Write-Warn "Non-ARM64 host -- the NPU is unavailable here; falling back to any Python on PATH."
    Write-Warn "Run npu_qa.py with --cpu on this machine."
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonExe) {
        Write-Error "No Python found on PATH."
        exit 1
    }
}

# Recreate the venv if it exists but was built from the wrong architecture.
if (Test-Path $VenvDir) {
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    $venvMachine = Get-InterpreterMachine $venvPython
    $wantArm64 = $hostIsArm64
    $venvIsArm64 = ($venvMachine -eq "ARM64")
    if ($wantArm64 -ne $venvIsArm64) {
        Write-Warn ".venv architecture mismatch (found $venvMachine) -- recreating ..."
        Remove-Item -Recurse -Force $VenvDir
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv ..."
    & $pythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed."; exit 1 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists -- skipping creation"
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
Write-Info "Installing dependencies from requirements.txt ..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed."; exit 1 }
Write-Ok "All dependencies installed"

Write-Host ""
Write-Ok "Run it: $VenvDir\Scripts\python.exe npu_qa.py --question ""..."" --context ""..."""
