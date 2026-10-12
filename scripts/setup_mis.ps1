param(
    [string]$Python = 'python',
    [string]$EnvironmentName = '.venv-clean'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($EnvironmentName -notmatch '^\.venv[-a-zA-Z0-9]*$') {
    throw 'Use a project-local environment name such as .venv-clean.'
}
$environmentPath = Join-Path $repoRoot $EnvironmentName
if (Test-Path -LiteralPath $environmentPath) {
    throw 'Environment already exists. Select a new environment name to preserve it.'
}
Get-Command uv -ErrorAction Stop | Out-Null
& $Python -m venv --without-pip $environmentPath
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$projectPython = Join-Path $environmentPath 'Scripts/python.exe'
$cache = Join-Path $repoRoot '.cache/uv'
& uv pip install --python $projectPython --cache-dir $cache torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& uv pip install --python $projectPython --cache-dir $cache -r (Join-Path $repoRoot 'requirements-mis-support.txt')
if ($LASTEXITCODE -ne 0) { throw 'Supporting package installation failed' }
# DHG's published torch<2 bound conflicts with Blackwell support. Only the tested
# HGNNPConv operations are used; this is an explicit override, not full compatibility.
& uv pip install --python $projectPython --cache-dir $cache --no-deps dhg==0.9.5
if ($LASTEXITCODE -ne 0) { throw 'DHG installation failed' }
Write-Output "Environment prepared at $environmentPath. Run the tests before using it."
