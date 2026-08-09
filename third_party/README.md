# Third-party components

cec-crucible is MIT. These components are not, and the boundary between them and
our code is deliberate and mechanical:

* **We never link to any of them.** No headers, no static libraries, no shared
  objects loaded into our address space.
* We talk to LibreHardwareMonitor over **HTTP on loopback**, and to PresentMon
  by **spawning it and reading the CSV it writes**. LibreHardwareMonitor talks to
  PawnIO by **device IOCTL** — we never touch PawnIO ourselves.
* That makes every one of these an independent program aggregated with ours on a
  distribution medium, not a derived work. GPL-2 §2 says so explicitly ("mere
  aggregation of another work not based on the Program … does not bring the other
  work under the scope of this License").

Keeping that boundary is the contract. If any future change links against one of
these, this file is wrong and the licensing has to be re-examined before shipping.

---

## Bundled in our releases

### LibreHardwareMonitor 0.9.6 — MPL-2.0

* Source: <https://github.com/LibreHardwareMonitor/LibreHardwareMonitor>
* Artifact: `LibreHardwareMonitor.zip` from the v0.9.6 release
* SHA-256: `086d9f1b5a99e643edc2cfaaac16051685b551e4c5ac0b32a57c58c0e529c001`
* Licence text: `LICENSES/MPL-2.0.txt`

Provides CPU package power and die temperature, which live in model-specific
registers no user-mode code can read. MPL-2.0 is file-level copyleft: we may
redistribute the binaries provided the notices survive and the source of the
covered files stays available (the upstream URL above discharges this). It does
not reach our code, which shares no files with it.

The **.NET Framework** build is used deliberately over the .NET 10 build: 4.8 is
present on every Windows 10/11 install, so a technician's machine needs no
runtime download.

### PresentMon (Intel) — MIT

* Source: <https://github.com/GameTechDev/PresentMon>
* Licence text: `LICENSES/PresentMon-MIT.txt`

Unmodified. MIT, so redistribution is unencumbered.

---

## NOT bundled — fetched at install time

### PawnIO 2.2.0 — see below, this one is not simple

* Source (driver): <https://github.com/namazso/PawnIO> — **GPL-2.0-or-later**
* Signed setup: <https://github.com/namazso/PawnIO.Setup> — **no stated licence**
* Installer URL: `https://github.com/namazso/PawnIO.Setup/releases/download/2.2.0/PawnIO_setup.exe`
* SHA-256: `1f519a22e47187f70a1379a48ca604981c4fcf694f4e65b734aaa74a9fba3032`
* Licence text: `LICENSES/GPL-2.0.txt` (covers the source, not necessarily the
  signed build)

**Why this one is downloaded rather than shipped.** The driver *source* is
GPL-2.0-or-later, which we could redistribute freely by also shipping the source.
But the thing that is actually useful — the **signed** setup — lives in a
separate repository with **no licence file at all**, and winget classifies it as
`Proprietary (Freeware)`. Freeware customarily means free to *use*, not free to
*redistribute*.

An unsigned driver built from the GPL source will not load on a normal machine,
so "just build it ourselves" is not a route to a shippable product.

Unstated redistribution terms are not the same as permissive ones, and this tool
goes out to customers. So the installer **downloads it from the official URL and
verifies the pinned SHA-256 above** — the operator's machine fetches it from
upstream, exactly as if they had run `winget install namazso.PawnIO` themselves.
Functionally identical for the technician; we are not redistributing it.

**To bundle it properly:** the project invites licensing questions at
`admin@namazso.eu`. With written permission to redistribute the signed setup,
bundling becomes a one-line change — drop the file into `packaging/vendor/` and
flip `PAWNIO_BUNDLED` in `Build-Release.ps1`. The fetch path stays as the
fallback.

---

## Keeping upstream current

Versions and hashes are pinned in `MANIFEST.txt`. `packaging/Fetch-ThirdParty.ps1`
downloads and verifies against it, and refuses to proceed on a hash mismatch —
a silent substitution of a component that runs in ring 0 is exactly the supply
chain attack worth refusing.

To move a version: bump it in `MANIFEST.txt`, run the fetch script (it prints
the new hash on mismatch), paste the hash in, re-run, and commit.
