param(
    [string]$RunName = ('controlled-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)

$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'RunName must use letters, numbers, hyphens or underscores.'
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    throw 'Project Python environment is missing. See docs/environment.md.'
}
$ValidationDirectory = "results/$RunName-validation"
$TestDirectory = "results/$RunName-test"
$RealDirectory = "results/$RunName-real"
$SelectionFile = "$ValidationDirectory/selection.json"
$ReportFile = "docs/$RunName.md"

function Invoke-ProjectPython {
    param([string[]]$PythonArguments)
    & $ProjectPython @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment stage failed (exit $LASTEXITCODE). Completed records remain available for resume."
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython -PythonArguments @('scripts/run_controlled.py', 'validation', '--output', $ValidationDirectory)
    Invoke-ProjectPython -PythonArguments @('scripts/check_controlled_protocol.py', $ValidationDirectory)
    Invoke-ProjectPython -PythonArguments @('scripts/run_controlled.py', 'select', '--output', $ValidationDirectory,
                           '--selection', $SelectionFile)
    Invoke-ProjectPython -PythonArguments @('scripts/check_controlled_protocol.py', $ValidationDirectory,
                           '--selection', $SelectionFile)
    Invoke-ProjectPython -PythonArguments @('scripts/run_controlled.py', 'test', '--output', $TestDirectory,
                           '--selection', $SelectionFile)
    Invoke-ProjectPython -PythonArguments @('scripts/run_real.py', '--output', $RealDirectory,
                           '--validation', $ValidationDirectory, '--selection', $SelectionFile)
    Invoke-ProjectPython -PythonArguments @('scripts/summarize_controlled.py', '--validation', $ValidationDirectory,
                           '--test', $TestDirectory, '--real', $RealDirectory, '--report', $ReportFile)
    Write-Output "Complete. Read $ReportFile"
}
finally {
    Pop-Location
}
