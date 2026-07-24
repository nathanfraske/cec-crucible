# SPDX-License-Identifier: MIT
#Requires -Version 5.1
<#
.SYNOPSIS
    cec-crucible QC burn-in gauntlet sequencer.

.DESCRIPTION
    Runs the recommended stress campaign (docs/gauntlet.md) as a series of
    short, self-contained cec-crucible invocations -- ONE PROCESS PER PHASE.
    This is deliberate: the CLI writes its report + markers only at finish(),
    so a single day-long process that BSODs at hour 23 produces zero output.
    Per-phase invocation makes every completed phase a durable checkpoint, and
    lets a WHEA (hardware-error) read bracket each phase so a corrected-error
    storm is attributed to the stimulus that caused it -- not smeared across
    the day.

    Requires a GPU-enabled binary (cargo build --release -p crucible-cli
    --features gpu, or --features cuda for the full-duplex PCIe path).

.PARAMETER Profile
    express (~2 h, the standing QC gate) | standard (~12 h, premium builds) |
    full (~24 h, SLA / mission-critical only). See docs/gauntlet.md section 8
    before choosing a longer tier -- hours 12-24 are largely insurance.

.PARAMETER SoakHours
    Override the elastic hot-soak (P4) length, in hours. 0 = profile default.

.PARAMETER Cuda
    Append --link-cuda to the PCIe/worst-case phases (needs a --features cuda
    binary; probed at runtime, falls back to wgpu if the driver is absent).

.PARAMETER Resume
    Skip phases already recorded complete in <OutDir>\manifest.json. Point
    -OutDir at an interrupted campaign to continue it.

.PARAMETER Plan
    Print the phase plan and estimated wall-clock, then exit without running.

.EXAMPLE
    .\scripts\gauntlet.ps1 -Profile express
.EXAMPLE
    .\scripts\gauntlet.ps1 -Profile full -SoakHours 12 -Cuda
.EXAMPLE
    .\scripts\gauntlet.ps1 -Profile standard -Resume -OutDir .\gauntlet-20260724-1030
#>
[CmdletBinding()]
param(
    [ValidateSet('express', 'standard', 'full')]
    [string] $Profile = 'express',

    [string] $Exe,
    [string] $OutDir,
    [string] $DeviceId,

    [int]    $SoakHours = 0,
    [int]    $MemMb = 0,
    [int]    $VramMb = 0,

    [switch] $Cuda,
    [switch] $Resume,
    [switch] $Plan,

    # Strict WHEA policy (design default): ANY WHEA event in a phase window is a
    # margin FAIL, corrected or not. Set -WheaStrict:$false to fail only on
    # fatal/uncorrected events and merely flag corrected ones.
    [bool]   $WheaStrict = $true
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Locate the binary
# ---------------------------------------------------------------------------
function Resolve-Exe {
    param([string] $Hint)
    if ($Hint) {
        if (Test-Path $Hint) { return (Resolve-Path $Hint).Path }
        throw "cec-crucible not found at -Exe '$Hint'"
    }
    $root = Split-Path -Parent $PSScriptRoot   # repo root (scripts\ is under it)
    $candidates = @(
        (Join-Path $root 'target\release\cec-crucible.exe'),
        (Join-Path $root 'target\debug\cec-crucible.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return (Resolve-Path $c).Path } }
    $onPath = Get-Command 'cec-crucible' -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "cec-crucible.exe not found. Build it (cargo build --release -p crucible-cli --features gpu) or pass -Exe."
}

# ---------------------------------------------------------------------------
# WHEA: read hardware-error events since a bracket start time.
# A *failed scan* is reported (Ok=$false) so it can never become a silent pass.
# ---------------------------------------------------------------------------
function Read-WheaSince {
    param([datetime] $Start)
    $r = [ordered]@{ Ok = $true; Total = 0; Fatal = 0; Corrected = 0 }
    try {
        $events = Get-WinEvent -FilterHashtable @{
            LogName      = 'System'
            ProviderName = 'Microsoft-Windows-WHEA-Logger'
            StartTime    = $Start
        } -ErrorAction Stop
        foreach ($e in $events) {
            $r.Total++
            if ($e.Level -le 2) { $r.Fatal++ } else { $r.Corrected++ }   # 1 Crit / 2 Err = fatal
        }
    }
    catch {
        if ($_.Exception.Message -match 'No events were found') {
            # genuinely zero events in the window -- clean
        }
        else {
            $r.Ok = $false   # could not read the log: treat as NOT a pass
        }
    }
    return $r
}

# ---------------------------------------------------------------------------
# Which subcommands write a report (and so take --out / --device-id).
# info / drives / gpu-info / version are informational and reject --out.
# ---------------------------------------------------------------------------
$script:LoadCmds = @('cpu', 'mem', 'storage', 'gpu', 'vram', 'link', 'run')
function Test-IsLoadCmd { param([string] $First) return ($script:LoadCmds -contains $First) }

# Run one CLI step (a space-delimited command string). Returns the exit code.
function Invoke-Step {
    param([string] $Exe, [string] $Command, [string] $PhaseOut)
    $tokens = $Command.Split(' ') | Where-Object { $_ -ne '' }
    $full = @() + $tokens
    if (Test-IsLoadCmd $tokens[0]) {
        $full += @('--out', $PhaseOut)
        if ($DeviceId) { $full += @('--device-id', $DeviceId) }
        if ($MemMb -gt 0 -and $tokens[0] -eq 'mem') { $full += @('--mb', "$MemMb") }
        if ($VramMb -gt 0 -and $tokens[0] -eq 'vram') { $full += @('--vram-mb', "$VramMb") }
        if ($Cuda -and (($tokens -contains 'worst-case') -or ($tokens[0] -eq 'link'))) {
            $full += '--link-cuda'
        }
    }
    Write-Host "    > cec-crucible $($full -join ' ')" -ForegroundColor DarkGray
    # Pipe the CLI's stdout to the host so it streams live (and into the
    # transcript) WITHOUT being folded into this function's return value --
    # otherwise $code becomes [stdout-lines..., exitcode] and every phase with
    # output reads as failed. Out-Host does not disturb $LASTEXITCODE.
    & $Exe @full | Out-Host
    return $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Phase table builder
# ---------------------------------------------------------------------------
function New-Phase {
    param([string] $Name, [switch] $Gate, [string[]] $Steps, [hashtable] $Cycle, [hashtable] $Soak)
    return [ordered]@{ Name = $Name; Gate = [bool]$Gate; Steps = $Steps; Cycle = $Cycle; Soak = $Soak }
}

function New-Phases {
    param([string] $Profile, [int] $SoakHours)

    $smoke = New-Phase -Name 'P0 smoke gate' -Gate -Steps @(
        'info', 'drives', 'gpu-info',
        'run quick --seconds 60', 'gpu --seconds 60', 'vram --seconds 120', 'link --seconds 60'
    )

    if ($Profile -eq 'express') {
        $soak = 1800
        if ($SoakHours -gt 0) { $soak = $SoakHours * 3600 }
        return @(
            $smoke,
            (New-Phase -Name 'E1 cold transients'   -Steps @('run anti-phase --seconds 600', 'run beat --seconds 600')),
            (New-Phase -Name 'E2 worst-case + ramp'  -Steps @('run worst-case --seconds 1200')),
            (New-Phase -Name 'E3 cold integrity'     -Steps @('mem --seconds 1200', 'vram --seconds 900')),
            (New-Phase -Name 'E4 Arrhenius dwell'    -Steps @("run cross --seconds $soak")),
            (New-Phase -Name 'E5 macro thermal cycle' -Cycle @{ Hot = 'run worst-case'; HotSec = 180; CoolSec = 120; Reps = 4 }),
            (New-Phase -Name 'E6 hot latent + cooldown' -Steps @('run worst-case --seconds 600', 'mem --seconds 150', 'vram --seconds 150'))
        )
    }

    # standard / full share structure; the elastic P4 soak + P5 cycle scale.
    if ($Profile -eq 'full') { $soakDefault = 10 * 3600; $cycleReps = 34 } else { $soakDefault = 5 * 3600; $cycleReps = 17 }
    $soak = $soakDefault
    if ($SoakHours -gt 0) { $soak = $SoakHours * 3600 }

    return @(
        $smoke,
        (New-Phase -Name 'P1 cold transient trio' -Steps @(
                'run anti-phase --seconds 900', 'run in-phase --seconds 600', 'run beat --seconds 1200')),
        (New-Phase -Name 'P2 cold integrity baseline' -Steps @(
                'mem --seconds 1800', 'vram --seconds 1200', 'run storage-cross --seconds 1500')),
        (New-Phase -Name 'P3 worst-case + ramp' -Steps @('run worst-case --seconds 1800')),
        (New-Phase -Name 'P4 steady-max hot soak' -Soak @{
                Base = 'run cross'; TotalSec = $soak; ChunkSec = 5400
                Integrity = @('mem --seconds 900', 'vram --seconds 900')
            }),
        (New-Phase -Name 'P5 macro thermal cycle' -Cycle @{ Hot = 'run worst-case'; HotSec = 240; CoolSec = 180; Reps = $cycleReps }),
        (New-Phase -Name 'P6 hot transient re-attack' -Steps @('run anti-phase --seconds 1500', 'run beat --seconds 1200')),
        (New-Phase -Name 'P7 late worst-case + hot integrity' -Steps @(
                'run worst-case --seconds 1800', 'mem --seconds 1800', 'vram --seconds 1800')),
        (New-Phase -Name 'P8 cooldown re-verify' -Steps @(
                'mem --seconds 600', 'vram --seconds 600', 'run storage-cross --seconds 600'))
    )
}

function Get-PhaseWallSec {
    param($Phase)
    $sec = 0
    if ($Phase.Steps) {
        foreach ($s in $Phase.Steps) {
            $m = [regex]::Match($s, '--seconds (\d+)')
            if ($m.Success) { $sec += [int]$m.Groups[1].Value } else { $sec += 5 }
        }
    }
    if ($Phase.Cycle) { $sec += $Phase.Cycle.Reps * ($Phase.Cycle.HotSec + $Phase.Cycle.CoolSec) }
    if ($Phase.Soak) {
        $sec += $Phase.Soak.TotalSec
        $breaks = [math]::Floor($Phase.Soak.TotalSec / $Phase.Soak.ChunkSec)
        foreach ($ig in $Phase.Soak.Integrity) {
            $m = [regex]::Match($ig, '--seconds (\d+)'); if ($m.Success) { $sec += $breaks * [int]$m.Groups[1].Value }
        }
    }
    return $sec
}

function Format-Duration { param([int] $Sec) return ('{0:d2}h{1:d2}m' -f [int][math]::Floor($Sec / 3600), [int][math]::Floor(($Sec % 3600) / 60)) }

# ---------------------------------------------------------------------------
# Run a single phase; returns a result record.
# ---------------------------------------------------------------------------
function Invoke-Phase {
    param($Exe, $Phase, $PhaseOut)
    New-Item -ItemType Directory -Force -Path $PhaseOut | Out-Null
    $start = Get-Date
    $worstExit = 0
    $ok = $true

    if ($Phase.Steps) {
        foreach ($s in $Phase.Steps) {
            $code = Invoke-Step -Exe $Exe -Command $s -PhaseOut $PhaseOut
            if ($code -eq 2) { throw "Usage error (exit 2) from: $s -- this is a script/args bug, not a hardware fault." }
            if ($code -ne 0) { $worstExit = 1; $ok = $false }
        }
    }
    elseif ($Phase.Cycle) {
        $c = $Phase.Cycle
        for ($i = 1; $i -le $c.Reps; $i++) {
            Write-Host "    cycle $i/$($c.Reps): hot $($c.HotSec)s / cool $($c.CoolSec)s" -ForegroundColor DarkGray
            $code = Invoke-Step -Exe $Exe -Command "$($c.Hot) --seconds $($c.HotSec)" -PhaseOut $PhaseOut
            if ($code -eq 2) { throw "Usage error (exit 2) in cycle: $($c.Hot)" }
            if ($code -ne 0) { $worstExit = 1; $ok = $false }
            if ($i -lt $c.Reps) { Start-Sleep -Seconds $c.CoolSec }   # idle = the cool half of the ΔT swing
        }
    }
    elseif ($Phase.Soak) {
        $s = $Phase.Soak
        $elapsed = 0
        while ($elapsed -lt $s.TotalSec) {
            $chunk = [math]::Min($s.ChunkSec, $s.TotalSec - $elapsed)
            $code = Invoke-Step -Exe $Exe -Command "$($s.Base) --seconds $chunk" -PhaseOut $PhaseOut
            if ($code -eq 2) { throw "Usage error (exit 2) in soak: $($s.Base)" }
            if ($code -ne 0) { $worstExit = 1; $ok = $false }
            $elapsed += $chunk
            if ($elapsed -lt $s.TotalSec) {
                foreach ($ig in $s.Integrity) {   # hot re-verify vs the cold baseline
                    $code = Invoke-Step -Exe $Exe -Command $ig -PhaseOut $PhaseOut
                    if ($code -ne 0) { $worstExit = 1; $ok = $false }
                }
            }
        }
    }

    $whea = Read-WheaSince -Start $start
    return [ordered]@{
        Name      = $Phase.Name
        Gate      = $Phase.Gate
        Exit      = $worstExit
        KernelOk  = $ok
        WheaOk    = $whea.Ok
        WheaTotal = $whea.Total
        WheaFatal = $whea.Fatal
        WheaCorr  = $whea.Corrected
        Start     = $start.ToString('o')
        End       = (Get-Date).ToString('o')
        Out       = $PhaseOut
    }
}

function Test-PhaseFailed {
    param($R)
    if (-not $R.KernelOk) { return $true }
    if ($R.WheaFatal -gt 0) { return $true }
    if ($WheaStrict -and $R.WheaTotal -gt 0) { return $true }
    return $false
}

# ===========================================================================
# Main
# ===========================================================================
$Exe = Resolve-Exe -Hint $Exe
if (-not $OutDir) {
    $stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $OutDir = Join-Path (Get-Location) "gauntlet-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$phases = New-Phases -Profile $Profile -SoakHours $SoakHours
$totalSec = ($phases | ForEach-Object { Get-PhaseWallSec $_ } | Measure-Object -Sum).Sum

Write-Host ''
Write-Host "cec-crucible GAUNTLET  --  profile: $Profile" -ForegroundColor Cyan
Write-Host ("  binary : {0}" -f $Exe)
Write-Host ("  out    : {0}" -f $OutDir)
Write-Host ("  est.   : ~{0} active ({1} phases){2}" -f (Format-Duration $totalSec), $phases.Count, $(if ($Cuda) { ', CUDA link' } else { '' }))
Write-Host ''

if ($Plan) {
    foreach ($ph in $phases) {
        $tag = if ($ph.Gate) { '[HARD GATE] ' } else { '' }
        Write-Host ("  {0,-38} ~{1}" -f "$tag$($ph.Name)", (Format-Duration (Get-PhaseWallSec $ph))) -ForegroundColor Yellow
        if ($ph.Steps) { foreach ($s in $ph.Steps) { Write-Host "      cec-crucible $s" -ForegroundColor DarkGray } }
        if ($ph.Cycle) { Write-Host "      $($ph.Cycle.Reps)x [ $($ph.Cycle.Hot) $($ph.Cycle.HotSec)s -> idle $($ph.Cycle.CoolSec)s ]" -ForegroundColor DarkGray }
        if ($ph.Soak) { Write-Host "      soak $($ph.Soak.Base) for $(Format-Duration $ph.Soak.TotalSec), hot mem+vram every $(Format-Duration $ph.Soak.ChunkSec)" -ForegroundColor DarkGray }
    }
    Write-Host ''
    Write-Host 'Plan only (-Plan): nothing was run.' -ForegroundColor Cyan
    return
}

# Manifest (resume support)
$manifestPath = Join-Path $OutDir 'manifest.json'
$completed = @()
if ($Resume -and (Test-Path $manifestPath)) {
    try {
        $prev = Get-Content $manifestPath -Raw | ConvertFrom-Json
        if ($prev.completed) { $completed = @($prev.completed) }
        Write-Host ("Resuming: {0} phase(s) already complete." -f $completed.Count) -ForegroundColor Cyan
    }
    catch { Write-Warning "Could not read manifest for resume; starting fresh." }
}

Start-Transcript -Path (Join-Path $OutDir 'gauntlet.log') -Append | Out-Null
$results = @()
$campaignStart = Get-Date
$aborted = $false

try {
    $idx = 0
    foreach ($ph in $phases) {
        $idx++
        if ($completed -contains $ph.Name) {
            Write-Host ("[{0}/{1}] SKIP (done) {2}" -f $idx, $phases.Count, $ph.Name) -ForegroundColor DarkGray
            continue
        }
        $wall = Format-Duration (Get-PhaseWallSec $ph)
        Write-Host ''
        Write-Host ("[{0}/{1}] {2}  (~{3})" -f $idx, $phases.Count, $ph.Name, $wall) -ForegroundColor Green

        $safe = ($ph.Name -replace '[^\w]+', '-')
        $phaseOut = Join-Path $OutDir $safe
        $r = Invoke-Phase -Exe $Exe -Phase $ph -PhaseOut $phaseOut
        $results += $r

        $failed = Test-PhaseFailed $r
        $wheaStr = if (-not $r.WheaOk) { 'WHEA:UNREAD' } elseif ($r.WheaTotal -eq 0) { 'WHEA:clean' } else { "WHEA:$($r.WheaTotal)(fatal $($r.WheaFatal))" }
        $verdict = if ($failed) { 'FAIL' } elseif (-not $r.WheaOk) { 'INCONCLUSIVE' } else { 'PASS' }
        $color = if ($failed) { 'Red' } elseif (-not $r.WheaOk) { 'Yellow' } else { 'Green' }
        Write-Host ("  -> {0}   exit={1}  {2}" -f $verdict, $r.Exit, $wheaStr) -ForegroundColor $color

        $completed += $ph.Name
        $manifest = [ordered]@{
            campaign  = "gauntlet-$Profile"
            device_id = $DeviceId
            started   = $campaignStart.ToString('o')
            profile   = $Profile
            completed = $completed
            results   = $results
        }
        $manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

        if ($ph.Gate -and $failed) {
            Write-Host ''
            Write-Host "HARD GATE FAILED at $($ph.Name) -- aborting campaign. Pull the unit." -ForegroundColor Red
            $aborted = $true
            break
        }
    }

    # -----------------------------------------------------------------------
    # Campaign summary (kept inside the try so it lands in gauntlet.log too)
    # -----------------------------------------------------------------------
    $elapsed = (New-TimeSpan -Start $campaignStart -End (Get-Date))
$anyFail = $false; $anyInconclusive = $false
Write-Host ''
Write-Host '==================== CAMPAIGN SUMMARY ====================' -ForegroundColor Cyan
foreach ($r in $results) {
    $failed = Test-PhaseFailed $r
    if ($failed) { $anyFail = $true }
    if (-not $r.WheaOk) { $anyInconclusive = $true }
    $v = if ($failed) { 'FAIL' } elseif (-not $r.WheaOk) { 'INCONCL' } else { 'PASS' }
    $c = if ($failed) { 'Red' } elseif (-not $r.WheaOk) { 'Yellow' } else { 'Green' }
    Write-Host ("  {0,-8} {1,-38} exit={2} whea={3}" -f $v, $r.Name, $r.Exit, $r.WheaTotal) -ForegroundColor $c
}
Write-Host ('  elapsed: {0:hh\:mm\:ss}   reports: {1}' -f $elapsed, $OutDir)

$final = 'PASS'; $finalColor = 'Green'; $exitCode = 0
if ($anyFail -or $aborted) { $final = 'FAIL'; $finalColor = 'Red'; $exitCode = 1 }
elseif ($anyInconclusive) { $final = 'INCONCLUSIVE (WHEA unread -- not a clean pass)'; $finalColor = 'Yellow'; $exitCode = 1 }
Write-Host ''
Write-Host ("CAMPAIGN VERDICT: {0}" -f $final) -ForegroundColor $finalColor
Write-Host '=========================================================' -ForegroundColor Cyan
}
finally {
    Stop-Transcript | Out-Null
}
exit $exitCode
