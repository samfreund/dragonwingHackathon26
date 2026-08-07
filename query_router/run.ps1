#Requires -Version 5.1
<#
.SYNOPSIS
    Dependency bootstrap helper for the query router sample -- arch-aware.

.DESCRIPTION
    The router LLM runs on the CPU, so this sample works on any host. The
    optional --npu path needs onnxruntime-qnn, which ships a win-arm64 wheel
    only, and an emulated x86 Python running under Prism on ARM64 Windows
    reports machine()=AMD64 and cannot use it.

    So: on ARM64 hardware this script finds (or recreates) a venv built from a
    NATIVE arm64 Python and additionally installs requirements-npu.txt.
    Elsewhere it falls back to whatever Python is on PATH and installs the
    portable CPU-only set.

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

$ReqFile    = Join-Path $PSScriptRoot "requirements.txt"
$ReqNpuFile = Join-Path $PSScriptRoot "requirements-npu.txt"
$VenvDir    = Join-Path $PSScriptRoot ".venv"

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
        Write-Warn @"
No native arm64 Python found on this ARM64 host. The router will still run
(it is CPU-only by default), but --npu will be unavailable. Install the
"Windows arm64" build from https://www.python.org/downloads/windows/ and
re-run this script to enable it.
"@
        $hostIsArm64 = $false
        $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $pythonExe) { Write-Error "No Python found on PATH."; exit 1 }
    } else {
        Write-Ok "native arm64 Python: $pythonExe"
    }
} else {
    Write-Warn "Non-ARM64 host -- installing the CPU-only set; --npu will be unavailable."
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
    $venvIsArm64 = ($venvMachine -eq "ARM64")
    if ($hostIsArm64 -ne $venvIsArm64) {
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

if ($hostIsArm64 -and (Test-Path $ReqNpuFile)) {
    Write-Info "ARM64 host -- installing the optional NPU extras ..."
    & $venvPython -m pip install -r $ReqNpuFile
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "onnxruntime-qnn install failed -- the CPU path still works; --npu will not."
    } else {
        Write-Ok "NPU extras installed (--npu available)"
    }
}

Write-Ok "All dependencies installed"

Write-Host ""
Write-Ok "Try it: $VenvDir\Scripts\python.exe router.py --self-test"
Write-Ok "Route:  $VenvDir\Scripts\python.exe router.py --question ""..."" --context ""..."""
