<#
.SYNOPSIS
    Assemble the two release archives.

.DESCRIPTION
    Produces:

      cec-crucible-<ver>-win-x64-setup.exe      installer
      cec-crucible-<ver>-win-x64-portable.zip   portable

    Both carry the same payload. The difference is intent, and it is worth
    keeping separate:

      * The installer is a real setup executable - nothing to unpack, an entry
        in Add/Remove Programs, and an uninstaller that removes everything it
        created. Silent install for imaging a bench:
            cec-crucible-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
      * The portable archive runs from wherever it is extracted and touches
        nothing outside its own folder except the settings file. It is what goes
        on a USB stick to a customer site, or onto a machine that must be left
        exactly as it was found.

    The portable archive deliberately contains no installer of any kind, so
    there is no way to half-install from it by accident.

    Also emits winget manifests under packaging/winget/, with the real installer
    hash filled in. Generated rather than hand-maintained because a manifest
    whose hash does not match the artifact is worse than no manifest: winget
    refuses the install and blames the network.

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
        # Nothing extra: the payload IS the installer's input. Inno Setup
        # compiles it into a single executable, so there is no loose installer
        # script for an operator to run by mistake.
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

# --- Installer ------------------------------------------------------------
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup not found. Install it with:  winget install -e --id JRSoftware.InnoSetup"
}
Step "compiling installer ..."
$setupExe = Join-Path $PSScriptRoot "cec-crucible-$ver-win-x64-setup.exe"
if (Test-Path $setupExe) { Remove-Item $setupExe -Force }
& $iscc /Qp "/DAppVersion=$ver" "/DPayload=stage" (Join-Path $PSScriptRoot 'cec-crucible.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed" }

# A second copy under a version-free name, uploaded alongside. It is what makes
#   .../releases/latest/download/cec-crucible-setup.exe
# a URL that keeps working, so the install one-liner in the README does not go
# stale the moment the next version ships. Same bytes, different name.
$setupStable = Join-Path $PSScriptRoot 'cec-crucible-setup.exe'
Copy-Item $setupExe $setupStable -Force

# --- Portable archive -----------------------------------------------------
$zipPortable = Join-Path $PSScriptRoot "cec-crucible-$ver-win-x64-portable.zip"
if (Test-Path $zipPortable) { Remove-Item $zipPortable -Force }
Step "compressing portable ..."
Compress-Archive -Path "$portage\*" -DestinationPath $zipPortable -CompressionLevel Optimal

# --- winget manifests -----------------------------------------------------
# So `winget install CriticalErrorComputing.Crucible` works once these are
# merged into microsoft/winget-pkgs. Generated here because the hash has to be
# the hash of the artifact we just built - see the note in the header.
Step "writing winget manifests ..."
$pkgId    = 'CriticalErrorComputing.Crucible'
$setupSha = (Get-FileHash $setupExe -Algorithm SHA256).Hash.ToUpper()
$relUrl   = "https://github.com/nathanfraske/cec-crucible/releases/download/v$ver/$(Split-Path $setupExe -Leaf)"
$wgDir    = Join-Path $PSScriptRoot "winget\$pkgId\$ver"
New-Item -ItemType Directory -Force $wgDir | Out-Null

@"
# yaml-language-server: `$schema=https://aka.ms/winget-manifest.version.1.6.0.schema.json
PackageIdentifier: $pkgId
PackageVersion: $ver
DefaultLocale: en-US
ManifestType: version
ManifestVersion: 1.6.0
"@ | Set-Content (Join-Path $wgDir "$pkgId.yaml") -Encoding UTF8

@"
# yaml-language-server: `$schema=https://aka.ms/winget-manifest.installer.1.6.0.schema.json
PackageIdentifier: $pkgId
PackageVersion: $ver
InstallerType: inno
Scope: user
InstallModes:
  - interactive
  - silent
  - silentWithProgress
UpgradeBehavior: install
ReleaseDate: $(Get-Date -Format 'yyyy-MM-dd')
Installers:
  - Architecture: x64
    InstallerUrl: $relUrl
    InstallerSha256: $setupSha
ManifestType: installer
ManifestVersion: 1.6.0
"@ | Set-Content (Join-Path $wgDir "$pkgId.installer.yaml") -Encoding UTF8

@"
# yaml-language-server: `$schema=https://aka.ms/winget-manifest.defaultLocale.1.6.0.schema.json
PackageIdentifier: $pkgId
PackageVersion: $ver
PackageLocale: en-US
Publisher: Critical Error Computing
PublisherUrl: https://github.com/nathanfraske
PublisherSupportUrl: https://github.com/nathanfraske/cec-crucible/issues
PackageName: CEC Crucible
PackageUrl: https://github.com/nathanfraske/cec-crucible
License: MIT
LicenseUrl: https://github.com/nathanfraske/cec-crucible/blob/master/LICENSE
Copyright: Copyright (c) Critical Error Computing
ShortDescription: PC-build stress, validation and benchmark suite for the workshop bench.
Description: |-
  A stress, validation and benchmark suite for people who build and repair PCs.
  Mission: if something is ever going to fail, make it fail in the shop.

  Loads CPU, memory, storage, GPU compute, VRAM, PCIe and the graphics pipeline,
  alone or simultaneously, and records power, temperature, clocks and drive
  health throughout. Ships with LibreHardwareMonitor and PresentMon; CPU package
  power additionally needs PawnIO, which the installer offers to fetch.
Moniker: crucible
Tags:
  - benchmark
  - burn-in
  - diagnostics
  - gpu
  - hardware
  - memory-test
  - stress-test
ReleaseNotesUrl: https://github.com/nathanfraske/cec-crucible/releases/tag/v$ver
ManifestType: defaultLocale
ManifestVersion: 1.6.0
"@ | Set-Content (Join-Path $wgDir "$pkgId.locale.en-US.yaml") -Encoding UTF8

Write-Host ""
foreach ($a in @($setupExe, $zipPortable)) {
    if (-not (Test-Path $a)) { continue }
    $i = Get-Item $a
    Ok ("{0,-46} {1,7:N1} MB" -f $i.Name, ($i.Length/1MB))
    Write-Host ("      sha256 {0}" -f (Get-FileHash $a -Algorithm SHA256).Hash.ToLower()) -ForegroundColor DarkGray
}
Ok ("{0,-46} {1,7} " -f "winget/$pkgId/$ver", "3 files")
Write-Host ""
