param(
    [string]$Python = 'python',
    [string]$EnvironmentName = '.venv-clean',
    [string]$RunName = ('clean-reproduction-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$CacheDirectory = '',
    [ValidateSet('Prepare', 'Smoke', 'All')][string]$Phase = 'All',
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use a simple run name.' }
if ($EnvironmentName -notmatch '^\.venv[-a-zA-Z0-9]*$') { throw 'Use a project-local environment name.' }
$output = Join-Path $repoRoot "results/$RunName"
$projectPython = Join-Path $repoRoot "$EnvironmentName/Scripts/python.exe"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    ('> ' + $Executable + ' ' + ($Arguments -join ' ')) | Tee-Object -FilePath $script:NativeLog -Append
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $script:NativeLog -Append
    if ($LASTEXITCODE -ne 0) { throw "A reproduction stage failed with exit code $LASTEXITCODE." }
}

function Start-RunTranscript {
    param([string]$Path)
    $transcriptArguments = @{LiteralPath = $Path}
    if ((Get-Command Start-Transcript).Parameters.ContainsKey('UseMinimalHeader')) {
        $transcriptArguments['UseMinimalHeader'] = $true
    }
    Start-Transcript @transcriptArguments | Out-Null
}

Push-Location -LiteralPath $repoRoot
try {
    if ($Phase -in @('Prepare', 'All')) {
        if (Test-Path -LiteralPath $output) { throw 'Output already exists. Select a new RunName.' }
        if (Test-Path -LiteralPath (Split-Path -Parent (Split-Path -Parent $projectPython))) {
            throw 'Environment already exists. Select a new EnvironmentName.'
        }
        New-Item -ItemType Directory -Path $output | Out-Null
        Start-RunTranscript -Path (Join-Path $output 'prepare.txt')
        try {
            $script:NativeLog = Join-Path $output 'native-prepare.txt'
            & (Join-Path $PSScriptRoot 'setup_mis.ps1') -Python $Python -EnvironmentName $EnvironmentName -CacheDirectory $CacheDirectory -InstallLog (Join-Path $output 'installation.txt') -Offline:$Offline
            Invoke-Checked -Executable $projectPython -Arguments @('-I', 'scripts/inspect_clean_environment.py', '--output', "$output/environment.json")
            $freeze = & uv pip freeze --python $projectPython
            if ($LASTEXITCODE -ne 0) { throw 'Package freeze failed.' }
            $freeze | Set-Content -LiteralPath (Join-Path $output 'requirements-freeze.txt') -Encoding utf8
            Write-Output 'All installed package versions:'
            Write-Output $freeze
            Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', '-m', 'unittest', 'discover', '-s', 'tests', '-v')
            Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', 'scripts/clean_scip_smoke.py', '--output', $output)
            Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', '-m', 'mis.verify_report', "$output/smoke-scip-weak.json", "$output/smoke-scip-strong.json")
            'CPU test suite passed. GPU smoke is a separate stage.' | Set-Content -LiteralPath (Join-Path $output 'prepare-complete.txt') -Encoding utf8
        }
        finally { Stop-Transcript | Out-Null }
    }
    if ($Phase -in @('Smoke', 'All')) {
        if (-not (Test-Path -LiteralPath (Join-Path $output 'prepare-complete.txt'))) {
            throw 'Successful preparation is required before the smoke stage.'
        }
        if (Test-Path -LiteralPath (Join-Path $output 'smoke.txt')) { throw 'Smoke output exists; it will not be overwritten.' }
        Start-RunTranscript -Path (Join-Path $output 'smoke.txt')
        try {
            $script:NativeLog = Join-Path $output 'native-smoke.txt'
            Invoke-Checked -Executable $projectPython -Arguments @('-I', 'scripts/inspect_clean_environment.py', '--output', "$output/environment-smoke.json")
            Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', '-c', 'import torch; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda",requires_grad=True); x.square().sum().backward(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda); assert x.grad.tolist()==[2.0]*8')
            foreach ($mode in @('weak', 'strong')) {
                Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', '-m', 'mis.run', '--instance', 'instances/toy_overlap.json', '--definition', $mode, '--device', 'cuda', '--epochs', '200', '--seeds', '0', '1', '2', '--output', "$output/smoke-gpu-$mode.json")
            }
            Invoke-Checked -Executable $projectPython -Arguments @('-E', '-s', '-m', 'mis.verify_report', "$output/smoke-gpu-weak.json", "$output/smoke-gpu-strong.json")
            'CUDA backward, weak/strong GPU toy runs and independent report verification passed.' | Set-Content -LiteralPath (Join-Path $output 'smoke-complete.txt') -Encoding utf8
        }
        finally { Stop-Transcript | Out-Null }
    }
}
finally { Pop-Location }
