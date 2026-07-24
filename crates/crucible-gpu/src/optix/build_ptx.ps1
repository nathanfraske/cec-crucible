# SPDX-License-Identifier: MIT
#
# Regenerate path_tracer.ptx from path_tracer.cu. Run only when the .cu changes;
# the resulting PTX is committed and shipped inside the binary (include_bytes!),
# so BUILD needs nvcc + the OptiX headers but the TARGET needs only the driver.
#
# Prerequisites (bench/dev machine only):
#   - CUDA toolkit (nvcc)         winget install -e --id Nvidia.CUDA
#   - OptiX headers (public repo, NOT committed — proprietary license):
#         git clone --depth 1 https://github.com/NVIDIA/optix-dev.git
#     placed as a sibling of the cec-crucible repo. Only used here for -I.
#   - MSVC build tools (cl.exe) — already required by the Rust MSVC toolchain.
#
# nvcc uses cl.exe as its host compiler, so we run inside the VS dev environment.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Adjust these if your install paths differ.
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$nvcc = (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$optix = Resolve-Path (Join-Path $here "..\..\..\..\..\optix-dev\include") -ErrorAction SilentlyContinue

if (-not $optix) { throw "OptiX headers not found. Clone https://github.com/NVIDIA/optix-dev.git as a sibling of the repo." }

$cu = Join-Path $here "path_tracer.cu"
$ptx = Join-Path $here "path_tracer.ptx"

# sm_86 PTX (Ampere) JITs forward to Blackwell; -diag-suppress 20044 silences the
# benign 'extern __constant__ params' OptiX-idiom warning.
$cmd = "`"$vcvars`" >nul 2>&1 && `"$nvcc`" -ptx -I `"$optix`" -arch=sm_86 -diag-suppress 20044 `"$cu`" -o `"$ptx`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "nvcc failed ($LASTEXITCODE)" }
Write-Output ("OK: {0} ({1} bytes)" -f $ptx, (Get-Item $ptx).Length)
