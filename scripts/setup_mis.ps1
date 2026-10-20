param(
    [string]$Python = 'python',
    [string]$EnvironmentName = '.venv-clean',
    [string]$CacheDirectory = '',
    [string]$InstallLog = '',
    [switch]$Offline
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
function Invoke-SetupChecked {
    param([string]$Executable, [string[]]$Arguments)
    if ($InstallLog) {
        ('> ' + $Executable + ' ' + ($Arguments -join ' ')) | Tee-Object -FilePath $InstallLog -Append
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $InstallLog -Append
    }
    else { & $Executable @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "Installation command failed with exit code $LASTEXITCODE." }
}
$pythonVersion = (& $Python -I -c 'import platform; print(platform.python_version())').Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne '3.9.23') {
    throw 'This verified setup requires Python 3.9.23. Pass its executable with -Python.'
}
Invoke-SetupChecked -Executable $Python -Arguments @('-I', '-m', 'venv', '--without-pip', $environmentPath)
$projectPython = Join-Path $environmentPath 'Scripts/python.exe'
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $repoRoot '.cache/uv'
}
$cache = [System.IO.Path]::GetFullPath($CacheDirectory)
$installArguments = @('pip', 'install', '--python', $projectPython,
                      '--cache-dir', $cache, '--link-mode', 'copy')
if ($Offline) { $installArguments += '--offline' }
$supportFile = Join-Path $repoRoot 'requirements-mis-support.txt'
Invoke-SetupChecked -Executable 'uv' -Arguments @('--version')
Invoke-SetupChecked -Executable 'uv' -Arguments ($installArguments + @('--constraint', $supportFile, 'torch==2.8.0+cu128', 'torchvision==0.23.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'))
Invoke-SetupChecked -Executable 'uv' -Arguments ($installArguments + @('-r', $supportFile))
# DHG's published torch<2 bound conflicts with Blackwell support. Only the tested
# HGNNPConv operations are used; this is an explicit override, not full compatibility.
Invoke-SetupChecked -Executable 'uv' -Arguments ($installArguments + @('--no-deps', 'dhg==0.9.5'))
Invoke-SetupChecked -Executable $projectPython -Arguments @('-I', '-c', 'import site,sys; assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE; print("Isolated environment:", sys.prefix)')
Write-Output "Environment prepared at $environmentPath. Run the tests before using it."
