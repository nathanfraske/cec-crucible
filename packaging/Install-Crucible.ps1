<#
.SYNOPSIS
    Installs cec-crucible for the current user and creates shortcuts that open
    the interactive TUI.

.DESCRIPTION
    Per-user install — NO ADMIN REQUIRED. Copies the single portable executable
    to %LOCALAPPDATA%\Programs\cec-crucible, puts it on the user PATH, and
    creates Start Menu + Desktop shortcuts that launch the interactive menu.

    The executable is self-contained: every import is a Windows system DLL.
    Vulkan / CUDA / NVML are loaded at runtime from the GPU driver and degrade
    gracefully when absent, so nothing else needs installing.

.PARAMETER InstallDir
    Override the install location.

.PARAMETER NoShortcuts
    Skip creating Start Menu / Desktop shortcuts.

.PARAMETER Uninstall
    Remove the install directory, PATH entry and shortcuts.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\Install-Crucible.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\Install-Crucible.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\cec-crucible'),
    [switch] $NoShortcuts,
    [switch] $Uninstall
)

$ErrorActionPreference = 'Stop'

$AppName   = 'cec-crucible'
$ExeName   = 'cec-crucible.exe'
$StartMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$Desktop   = [Environment]::GetFolderPath('Desktop')
$LnkName   = 'CEC Crucible.lnk'

function Write-Step { param([string]$m) Write-Host "  $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "  $m" -ForegroundColor Green }
function Write-Warn2{ param([string]$m) Write-Host "  $m" -ForegroundColor Yellow }

Write-Host ""
Write-Host " CEC CRUCIBLE " -ForegroundColor Black -BackgroundColor Magenta -NoNewline
Write-Host "  PC-build stress & validation suite"
Write-Host ""

# --- Uninstall ------------------------------------------------------------
if ($Uninstall) {
    Write-Step "Uninstalling..."

    foreach ($lnk in @((Join-Path $StartMenu $LnkName), (Join-Path $Desktop $LnkName))) {
        if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Ok "removed shortcut $lnk" }
    }

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath) {
        $kept = ($userPath -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -ne $InstallDir.TrimEnd('\')) }) -join ';'
        if ($kept -ne $userPath) {
            [Environment]::SetEnvironmentVariable('Path', $kept, 'User')
            Write-Ok "removed from user PATH"
        }
    }

    if (Test-Path $InstallDir) {
        Remove-Item $InstallDir -Recurse -Force
        Write-Ok "removed $InstallDir"
    }

    Write-Host ""
    Write-Ok "Uninstalled."
    Write-Host ""
    return
}

# --- Locate the payload ---------------------------------------------------
$srcExe = Join-Path $PSScriptRoot $ExeName
if (-not (Test-Path $srcExe)) {
    throw "$ExeName not found next to this script ($PSScriptRoot). Extract the whole zip and run the script from inside it."
}

# --- Refuse to clobber a running instance ---------------------------------
$running = Get-Process -Name ([IO.Path]::GetFileNameWithoutExtension($ExeName)) -ErrorAction SilentlyContinue
if ($running) {
    throw "$ExeName is currently running (PID $($running.Id -join ', ')). Close it and re-run the installer."
}

# --- Install --------------------------------------------------------------
Write-Step "Installing to $InstallDir"
if (-not (Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }

Copy-Item $srcExe (Join-Path $InstallDir $ExeName) -Force
foreach ($extra in @('README.md', 'LICENSE', 'RELEASE-NOTES.md')) {
    $p = Join-Path $PSScriptRoot $extra
    if (Test-Path $p) { Copy-Item $p $InstallDir -Force }
}
Write-Ok "copied $ExeName"

$destExe = Join-Path $InstallDir $ExeName

# --- PATH (user scope, idempotent) ----------------------------------------
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }
$already = $userPath -split ';' | Where-Object { $_ -and ($_.TrimEnd('\') -eq $InstallDir.TrimEnd('\')) }
if (-not $already) {
    $newPath = if ($userPath.Trim()) { "$($userPath.TrimEnd(';'));$InstallDir" } else { $InstallDir }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Ok "added to user PATH (open a NEW terminal to use 'cec-crucible')"
} else {
    Write-Ok "already on user PATH"
}

# --- Shortcuts that open the TUI ------------------------------------------
if (-not $NoShortcuts) {
    # Launch via cmd /k so the console window persists and the TUI has a real
    # terminal to draw into; bare `cec-crucible` with no args opens the menu.
    $shell = New-Object -ComObject WScript.Shell
    foreach ($dir in @($StartMenu, $Desktop)) {
        if (-not (Test-Path $dir)) { continue }
        $lnkPath = Join-Path $dir $LnkName
        $lnk = $shell.CreateShortcut($lnkPath)
        $lnk.TargetPath       = "$env:SystemRoot\System32\cmd.exe"
        $lnk.Arguments        = "/k `"`"$destExe`"`""
        $lnk.WorkingDirectory = $InstallDir
        $lnk.IconLocation     = "$destExe,0"
        $lnk.Description      = 'CEC Crucible — PC-build stress & validation suite'
        $lnk.Save()
        Write-Ok "shortcut: $lnkPath"
    }
    [Runtime.InteropServices.Marshal]::ReleaseComObject($shell) | Out-Null
}

# --- Verify ---------------------------------------------------------------
Write-Host ""
Write-Step "Verifying..."
$ver = & $destExe version 2>&1
if ($LASTEXITCODE -ne 0) { throw "installed binary failed to run: $ver" }
Write-Ok "$ver"

$gpu = & $destExe gpu-info 2>&1 | Select-Object -First 4
Write-Host ""
Write-Host "  Detected GPUs:" -ForegroundColor Cyan
$gpu | ForEach-Object { Write-Host "    $_" }

Write-Host ""
Write-Ok "Installed."
Write-Host ""
Write-Host "  Launch the interactive TUI:" -ForegroundColor White
Write-Host "    * Double-click the 'CEC Crucible' shortcut on your Desktop, or" -ForegroundColor Gray
Write-Host "    * open a NEW terminal and run:  cec-crucible" -ForegroundColor Gray
Write-Host ""
Write-Host "  A few things to try:" -ForegroundColor White
Write-Host "    cec-crucible info                 system + device identity" -ForegroundColor Gray
Write-Host "    cec-crucible run quick            ~15s CPU/RAM/storage QC" -ForegroundColor Gray
Write-Host "    cec-crucible benchmark            graphics composite score" -ForegroundColor Gray
Write-Host "    cec-crucible run worst-case --ui  everything at once, live dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "  Uninstall:  powershell -ExecutionPolicy Bypass -File Install-Crucible.ps1 -Uninstall" -ForegroundColor DarkGray
Write-Host ""
