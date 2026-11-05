param([string]$Python = "python")
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & $Python validate_pc.py
    if ($LASTEXITCODE -ne 0) { throw 'Offline validation failed; inspect reports.' }
    & $Python pc_mock_demo.py
    if ($LASTEXITCODE -ne 0) { throw 'Mock gateway validation failed.' }
    Write-Output 'Offline validation passed. No physical robot was connected or moved.'
} finally { Pop-Location }
