<#
.SYNOPSIS
    Download and verify the pinned third-party components.

.DESCRIPTION
    Reads third_party/MANIFEST.txt and fetches each artifact into
    packaging/vendor/, verifying the pinned SHA-256.

    A hash mismatch ABORTS. It never warns and continues: one of these
    components ends up running in ring 0 on a customer's machine, and silently
    accepting a substituted binary is precisely the supply-chain attack worth
    refusing. If upstream legitimately re-cut a release, the new hash is printed
    so it can be reviewed and pinned deliberately.

.PARAMETER All
    Also fetch components marked `bundle = no` (i.e. PawnIO), for testing the
    install path offline. They are still not placed in release archives.

.PARAMETER Force
    Re-download even when a verified copy is already cached.
#>
[CmdletBinding()]
param(
    [switch] $All,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$root     = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $root 'third_party\MANIFEST.txt'
$vendor   = Join-Path $PSScriptRoot 'vendor'

if (-not (Test-Path $manifest)) { throw "manifest not found: $manifest" }
New-Item -ItemType Directory -Force $vendor | Out-Null

# --- Parse the manifest into records --------------------------------------
$records = @()
$cur = @{}
foreach ($line in Get-Content $manifest) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    if ($t -match '^\s*(\w+)\s*=\s*(.+?)\s*$') {
        $k = $Matches[1]; $v = $Matches[2]
        if ($k -eq 'id' -and $cur.Count -gt 0) { $records += ,$cur; $cur = @{} }
        $cur[$k] = $v
    }
}
if ($cur.Count -gt 0) { $records += ,$cur }

Write-Host ""
Write-Host " third-party components " -ForegroundColor Black -BackgroundColor Magenta
Write-Host ""

$failed = 0
foreach ($r in $records) {
    if (-not $All -and $r.bundle -ne 'yes') {
        Write-Host ("  {0,-24} {1,-8} not bundled (installer fetches it) - skipping" -f $r.id, $r.version) -ForegroundColor DarkGray
        continue
    }

    $name = Split-Path -Leaf ([Uri]$r.url).AbsolutePath
    $dest = Join-Path $vendor "$($r.id)-$($r.version)-$name"

    if ((Test-Path $dest) -and -not $Force) {
        $have = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
        if ($have -eq $r.sha256.ToLower()) {
            Write-Host ("  {0,-24} {1,-8} cached, verified" -f $r.id, $r.version) -ForegroundColor Green
            continue
        }
        Write-Host ("  {0,-24} cached copy failed verification, refetching" -f $r.id) -ForegroundColor Yellow
    }

    Write-Host ("  {0,-24} {1,-8} downloading ..." -f $r.id, $r.version) -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $r.url -OutFile $dest -UseBasicParsing
    } catch {
        Write-Host ("  {0,-24} DOWNLOAD FAILED: {1}" -f $r.id, $_.Exception.Message) -ForegroundColor Red
        $failed++
        continue
    }

    $have = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
    if ($have -ne $r.sha256.ToLower()) {
        Write-Host ""
        Write-Host "  HASH MISMATCH for $($r.id) - refusing to use this file." -ForegroundColor Red
        Write-Host "    expected $($r.sha256)" -ForegroundColor Red
        Write-Host "    got      $have" -ForegroundColor Red
        Write-Host "  If upstream re-cut this release, review the change and pin the new" -ForegroundColor Yellow
        Write-Host "  hash in third_party/MANIFEST.txt deliberately." -ForegroundColor Yellow
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        $failed++
        continue
    }
    Write-Host ("  {0,-24} {1,-8} verified" -f $r.id, $r.version) -ForegroundColor Green
}

Write-Host ""
if ($failed -gt 0) {
    Write-Host "  $failed component(s) unavailable." -ForegroundColor Red
    exit 1
}
Write-Host "  All pinned components present and verified." -ForegroundColor Green
Write-Host ""
# Explicit: a script that falls off the end leaves $LASTEXITCODE as whatever ran
# before it, which had the release builder refusing to build after a clean fetch.
exit 0
