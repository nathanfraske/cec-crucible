<#
.SYNOPSIS
    Assemble the two release archives.

.DESCRIPTION
    Produces:

      cec-crucible-<ver>-win-x64.zip           installer release
      cec-crucible-<ver>-win-x64-portable.zip  portable release

    Both carry the same payload. The difference is intent, and it is worth
    keeping separate:

      * The installer copies into %LOCALAPPDATA%, puts itself on PATH, makes
        shortcuts, offers the CPU-sensor driver, and can uninstall all of it.
      * The portable archive runs from wherever it is extracted and touches
        nothing outside its own folder except the settings file. It is what goes
        on a USB stick to a customer site, or onto a machine that must be left
        exactly as it was found.

    The portable archive deliberately omits Install-Crucible.ps1 so there is no
    way to half-install from it by accident.

.PARAMETER SkipBuild
    Use the existing target/release binary instead of rebuilding.
#>
[CmdletBinding()]
param(
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'
$root    = Split-Path -Parent $PSScriptRoot
$vendor  = Join-Path $PSScriptRoot 'vendor'
$stage   = Join-Path $PSScriptRoot 'stage'
$portage = Join-Path $PSScriptRoot 'stage-portable'

function Step($m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  $m" -ForegroundColor Green }

# --- Version straight from the workspace, never typed twice ---------------
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version\s*=\s*"(.+)"' |
        Select-Object -First 1).Matches[0].Groups[1].Value
Write-Host ""
Write-Host " CEC CRUCIBLE " -ForegroundColor Black -BackgroundColor Magenta -NoNewline
Write-Host "  building release $ver"
Write-Host ""

# --- Third-party components, verified -------------------------------------
Step "verifying pinned third-party components ..."
& (Join-Path $PSScriptRoot 'Fetch-ThirdParty.ps1')
if ($LASTEXITCODE -ne 0) { throw "third-party components unavailable; refusing to build a release" }

# --- Our binary -----------------------------------------------------------
$exe = Join-Path $root 'target\release\cec-crucible.exe'
if (-not $SkipBuild) {
    Step "cargo build --release ..."
    Push-Location $root
    try {
        & cargo build --release -p crucible-cli --features "tui,gpu"
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    } finally { Pop-Location }
}
if (-not (Test-Path $exe)) { throw "no binary at $exe" }

# --- Common payload -------------------------------------------------------
function Build-Payload([string]$dir, [bool]$withInstaller) {
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    New-Item -ItemType Directory -Force $dir | Out-Null

    Copy-Item $exe (Join-Path $dir 'cec-crucible.exe')
    foreach ($f in @('RELEASE-NOTES.md','THIRD-PARTY-NOTICES.md')) {
        Copy-Item (Join-Path $PSScriptRoot $f) $dir
    }
    foreach ($f in @('README.md','LICENSE')) {
        Copy-Item (Join-Path $root $f) $dir -ErrorAction SilentlyContinue
    }

    # PresentMon rides along; it is MIT and unmodified. Sourced from vendor/,
    # NOT from the staging directory: an earlier version of this script read it
    # from `stage` and then deleted `stage` on the next build, destroying the
    # only copy on disk. Build inputs must never live where build outputs go.
    $pm = Get-ChildItem $vendor -Filter 'presentmon-*.exe' -ErrorAction SilentlyContinue |
          Select-Object -First 1
    if ($pm) { Copy-Item $pm.FullName (Join-Path $dir 'PresentMon.exe') }
    else { Write-Host "  WARNING: PresentMon not in vendor/ - run Fetch-ThirdParty.ps1" -ForegroundColor Yellow }

    # Licence texts travel with the binaries they cover, not just as a URL in a
    # notices file - MPL-2.0 requires the notice to survive redistribution.
    $lic = Join-Path $dir 'licenses'
    New-Item -ItemType Directory -Force $lic | Out-Null
    Copy-Item (Join-Path $root 'third_party\LICENSES\*') $lic
    Copy-Item (Join-Path $root 'third_party\README.md') (Join-Path $lic 'THIRD-PARTY-BOUNDARY.md')

    # LibreHardwareMonitor, unpacked into its own subdirectory so the boundary
    # is visible on disk and the uninstaller can remove exactly it.
    $lhmZip = Get-ChildItem $vendor -Filter 'librehardwaremonitor-*.zip' | Select-Object -First 1
    if ($lhmZip) {
        $lhmDir = Join-Path $dir 'LibreHardwareMonitor'
        New-Item -ItemType Directory -Force $lhmDir | Out-Null
        Expand-Archive -Path $lhmZip.FullName -DestinationPath $lhmDir -Force
    }

    if ($withInstaller) {
        Copy-Item (Join-Path $PSScriptRoot 'Install-Crucible.ps1') $dir
        Copy-Item (Join-Path $PSScriptRoot 'INSTALL.cmd') $dir
    } else {
        # A portable archive that can half-install itself is not portable.
        @"
cec-crucible - portable

Run cec-crucible.exe. Nothing is installed, nothing is added to PATH, and no
shortcuts are created.

Two things are written outside this folder, both removable:
  * settings, at %APPDATA%\cec-crucible\settings.conf
  * PawnIO, only if you opt in to CPU package power (see below)

CPU package power and die temperature need a signed kernel module (PawnIO)
because those values live in registers no user-mode code can read. Portable mode
will NOT install it. Everything else - GPU power, SSD health, board zones,
per-core clocks - works with no driver at all.

To get CPU power on a portable run, install PawnIO once on the machine:
    winget install -e --id namazso.PawnIO
LibreHardwareMonitor is already in this folder and cec-crucible starts it itself.

Uninstall: delete this folder, and %APPDATA%\cec-crucible if you want the
settings gone too.
"@ | Set-Content (Join-Path $dir 'PORTABLE-README.txt') -Encoding UTF8
    }
}

Step "staging installer payload ..."
Build-Payload $stage $true
Step "staging portable payload ..."
Build-Payload $portage $false

# --- Archives -------------------------------------------------------------
$zipInstall  = Join-Path $PSScriptRoot "cec-crucible-$ver-win-x64.zip"
$zipPortable = Join-Path $PSScriptRoot "cec-crucible-$ver-win-x64-portable.zip"
foreach ($z in @($zipInstall, $zipPortable)) {
    if (Test-Path $z) { Remove-Item $z -Force }
}
Step "compressing ..."
Compress-Archive -Path "$stage\*"   -DestinationPath $zipInstall  -CompressionLevel Optimal
Compress-Archive -Path "$portage\*" -DestinationPath $zipPortable -CompressionLevel Optimal

Write-Host ""
foreach ($z in @($zipInstall, $zipPortable)) {
    $i = Get-Item $z
    Ok ("{0,-46} {1,7:N1} MB  sha256 {2}" -f $i.Name, ($i.Length/1MB), (Get-FileHash $z -Algorithm SHA256).Hash.Substring(0,16).ToLower())
}
Write-Host ""
