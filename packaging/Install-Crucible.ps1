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
    # Skip the optional CPU-sensor driver (PawnIO). LibreHardwareMonitor is
    # bundled and always installed; only the kernel module is optional.
    [switch] $NoCpuSensors,
    # Uninstall without prompting about PawnIO - leave it on the machine.
    [switch] $KeepPawnIO,
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

    # Stop the sensor daemon we may have started, before deleting it out from
    # under itself. It is ours only when it lives inside our install directory;
    # a copy the operator installed separately is left alone.
    $ourLhm = Join-Path $InstallDir 'LibreHardwareMonitor'
    Get-Process LibreHardwareMonitor -ErrorAction SilentlyContinue | ForEach-Object {
        $path = try { $_.MainModule.FileName } catch { '' }
        if ($path -and $path.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase)) {
            $_.CloseMainWindow() | Out-Null
            Start-Sleep -Milliseconds 800
            if (-not $_.HasExited) { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
            Write-Ok "stopped bundled LibreHardwareMonitor"
        } else {
            Write-Warn2 "left a LibreHardwareMonitor running from $path (not ours)"
        }
    }

    if (Test-Path $InstallDir) {
        Remove-Item $InstallDir -Recurse -Force
        Write-Ok "removed $InstallDir"
    }

    # Settings live outside the install directory on purpose (they follow the
    # technician between machines), so removing the directory does not remove
    # them. Leaving them behind would be a lie about "fully uninstalled".
    $cfgDir = Join-Path $env:APPDATA 'cec-crucible'
    if (Test-Path $cfgDir) {
        Remove-Item $cfgDir -Recurse -Force
        Write-Ok "removed settings at $cfgDir"
    }

    # An ETW session outlives the process that created it. A run killed
    # mid-flight can leave ours recording in the kernel, and uninstalling the
    # tool that owns it would strand it there permanently.
    $sessions = (& logman query -ets 2>&1 | Out-String) -split "`r?`n" |
        Where-Object { $_ -match '^(cec-crucible|crucible-)\S*' } |
        ForEach-Object { ($_ -split '\s+')[0] }
    foreach ($sn in $sessions) {
        & logman stop $sn -ets 2>&1 | Out-Null
        Write-Ok "stopped leftover ETW session $sn"
    }

    # PawnIO is a kernel driver we may have installed. Offer to take it with us:
    # other tools (FanControl, LibreHardwareMonitor installed separately) use it
    # too, so removing it unasked could break something the operator relies on.
    if (-not $KeepPawnIO) {
        $pawnUninst = Join-Path ${env:ProgramFiles} 'PawnIO\uninstall.exe'
        if (Test-Path $pawnUninst) {
            Write-Host ""
            Write-Host "  PawnIO (the CPU-sensor kernel module) is still installed." -ForegroundColor White
            Write-Host "  Other tools may use it - FanControl and LibreHardwareMonitor do." -ForegroundColor Gray
            $ans = Read-Host "    Remove PawnIO as well? [y/N]"
            if ($ans -match '^(y|yes)$') {
                Start-Process -FilePath $pawnUninst -ArgumentList '/S' -Wait -Verb RunAs -ErrorAction SilentlyContinue
                if (Test-Path $pawnUninst) { Write-Warn2 "PawnIO uninstaller ran but files remain; remove manually if you want it gone" }
                else { Write-Ok "removed PawnIO" }
            } else {
                Write-Host "    Left installed." -ForegroundColor DarkGray
            }
        }
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
foreach ($extra in @('README.md', 'LICENSE', 'RELEASE-NOTES.md', 'THIRD-PARTY-NOTICES.md')) {
    $p = Join-Path $PSScriptRoot $extra
    if (Test-Path $p) { Copy-Item $p $InstallDir -Force }
}
Write-Ok "copied $ExeName"

# PresentMon ships alongside so --presentmon works with no extra setup. It is
# Intel's MIT-licensed tool, unmodified; see THIRD-PARTY-NOTICES.md. Optional —
# everything else works without it.
$pm = Join-Path $PSScriptRoot 'PresentMon.exe'
if (Test-Path $pm) {
    Copy-Item $pm (Join-Path $InstallDir 'PresentMon.exe') -Force
    Write-Ok "copied PresentMon.exe (enables --presentmon)"
}

# Directories, which the file loop above cannot carry. Missing this is why an
# install once produced a working binary with no bundled sensor daemon beside
# it: LibreHardwareMonitor sat in the extracted archive and never arrived.
foreach ($sub in @('LibreHardwareMonitor', 'licenses')) {
    $src = Join-Path $PSScriptRoot $sub
    if (Test-Path $src) {
        $dst = Join-Path $InstallDir $sub
        if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
        Copy-Item $src $dst -Recurse -Force
        Write-Ok "copied $sub\"
    }
}

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


# --- CPU package power + die temperature ----------------------------------
#
# LibreHardwareMonitor ships in this package (MPL-2.0) and was copied in with
# everything else above; cec-crucible configures and starts it itself. What it
# still needs is PawnIO, a signed kernel module, because package power and die
# temperature live in model-specific registers no user-mode code can read.
#
# PawnIO is NOT redistributed here - its signed setup has no stated licence, so
# the installer downloads it from the official URL and verifies a pinned hash.
# See licenses/THIRD-PARTY-BOUNDARY.md.
#
# Offered, never forced: this puts a kernel module on the machine, and that is a
# decision somebody should make rather than absorb as a side effect.
$PawnIoUrl    = 'https://github.com/namazso/PawnIO.Setup/releases/download/2.2.0/PawnIO_setup.exe'
$PawnIoSha256 = '1f519a22e47187f70a1379a48ca604981c4fcf694f4e65b734aaa74a9fba3032'

$pawnPresent = Test-Path (Join-Path ${env:ProgramFiles} 'PawnIO\PawnIOLib.dll')
if ($pawnPresent) {
    Write-Ok "PawnIO already installed - CPU package power will be available"
} elseif (-not $NoCpuSensors) {
    Write-Host ""
    Write-Host "  Optional: CPU package power + die temperature" -ForegroundColor White
    Write-Host "    Needs PawnIO, a signed, sandboxed kernel module - those values" -ForegroundColor Gray
    Write-Host "    live in registers no user-mode code can read. It replaces the old" -ForegroundColor Gray
    Write-Host "    WinRing0 driver that Defender now flags." -ForegroundColor Gray
    Write-Host "    Downloaded from the official release and hash-verified; it is not" -ForegroundColor Gray
    Write-Host "    redistributed in this package." -ForegroundColor Gray
    Write-Host "    Everything else - GPU power, SSD health, board zones, per-core" -ForegroundColor Gray
    Write-Host "    clocks - already works without it." -ForegroundColor Gray
    Write-Host ""
    $ans = Read-Host "    Download and install PawnIO now? [y/N]"
    if ($ans -match '^(y|yes)$') {
        $tmp = Join-Path $env:TEMP 'PawnIO_setup.exe'
        try {
            Write-Step "downloading PawnIO ..."
            Invoke-WebRequest -Uri $PawnIoUrl -OutFile $tmp -UseBasicParsing
            $got = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
            if ($got -ne $PawnIoSha256) {
                # Refuse rather than warn: this one runs in ring 0.
                Write-Warn2 "PawnIO hash mismatch - NOT installing."
                Write-Warn2 "  expected $PawnIoSha256"
                Write-Warn2 "  got      $got"
                Remove-Item $tmp -Force -ErrorAction SilentlyContinue
            } else {
                Write-Ok "hash verified"
                Write-Step "installing PawnIO (expect an elevation prompt) ..."
                Start-Process -FilePath $tmp -ArgumentList '/S' -Wait -Verb RunAs
                Remove-Item $tmp -Force -ErrorAction SilentlyContinue
                if (Test-Path (Join-Path ${env:ProgramFiles} 'PawnIO\PawnIOLib.dll')) {
                    Write-Ok "PawnIO installed"
                } else {
                    Write-Warn2 "PawnIO does not appear to have installed; run 'cec-crucible sensors' to check"
                }
            }
        } catch {
            Write-Warn2 "could not fetch PawnIO: $($_.Exception.Message)"
            Write-Warn2 "  install it later with:  winget install -e --id namazso.PawnIO"
        }
    } else {
        Write-Host "    Skipped. 'cec-crucible sensors' will tell you how to add it later." -ForegroundColor DarkGray
    }
}
if (Test-Path (Join-Path $InstallDir 'LibreHardwareMonitor\LibreHardwareMonitor.exe')) {
    Write-Ok "LibreHardwareMonitor bundled (MPL-2.0) - started automatically when needed"
}

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
Write-Host "    cec-crucible sensors              what this machine can measure" -ForegroundColor Gray
Write-Host "    cec-crucible run worst-case --ui  everything at once, live dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "  Uninstall (removes files, PATH, shortcuts, settings, and offers to" -ForegroundColor DarkGray
Write-Host "  remove PawnIO):" -ForegroundColor DarkGray
Write-Host "    powershell -ExecutionPolicy Bypass -File Install-Crucible.ps1 -Uninstall" -ForegroundColor DarkGray
Write-Host ""
