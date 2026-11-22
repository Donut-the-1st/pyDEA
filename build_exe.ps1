# Builds dist\pyDEA\pyDEA.exe and, if Inno Setup is installed, dist\pyDEA-setup.exe.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller pyDEA.spec --noconfirm --clean

$iscc = @(
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($iscc) {
    & $iscc installer\pyDEA.iss
} else {
    Write-Warning 'Inno Setup 6 not found; skipping installer. Folder build is in dist\pyDEA.'
}
