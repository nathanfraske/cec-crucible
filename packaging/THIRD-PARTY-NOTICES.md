# Third-party components bundled with cec-crucible

## PresentMon (`PresentMon.exe`)

Intel's open-source frame-timing tool, used by the optional `--presentmon` flag
to capture ETW frame data (displayed frames, GPU busy, CPU busy/wait, display
latency, dropped frames) alongside a presenting run.

* Upstream: <https://github.com/GameTechDev/PresentMon>
* Version bundled: **2.5.1**
* Licence: **MIT**

It is shipped unmodified, purely for convenience so a bench machine needs no
extra setup. cec-crucible runs it as a separate process and never links against
it; the suite works fine without it (the flag simply reports that no capture was
made). Delete `PresentMon.exe` from the install directory if you would rather
supply your own copy — `--presentmon-path`, the `CRUCIBLE_PRESENTMON`
environment variable and `PATH` are all still honoured.

Note that an ETW real-time session requires elevation, so PresentMon relaunches
itself as administrator (one UAC prompt) when `--presentmon` is used.

### PresentMon licence

```
Copyright (C) 2017-2024 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

cec-crucible itself is MIT licensed — see `LICENSE`.


## Optional CPU sensor daemons (NOT bundled)

cec-crucible reads CPU package power and die temperature from a sensor daemon
rather than shipping a kernel driver of its own. Neither of these is included in
this package; the installer offers to fetch them via `winget`, and both are
removable independently.

**LibreHardwareMonitor** — Mozilla Public License 2.0.
<https://github.com/LibreHardwareMonitor/LibreHardwareMonitor>
Preferred backend. cec-crucible writes two keys into its config file
(`runWebServerMenuItem`, `listenerIp=127.0.0.1`) and reads its local `/metrics`
endpoint. The listener is bound to loopback deliberately.

**PawnIO** — <https://pawnio.eu/> · <https://github.com/namazso/PawnIO>
The signed, sandboxed kernel module LibreHardwareMonitor 0.9.5+ uses in place of
WinRing0. Required for CPU power/temperature; without it LHM runs but reports no
CPU sensors.

**HWiNFO** — <https://www.hwinfo.com/> (freeware, not redistributed).
Supported as an alternative backend via its documented shared-memory interface,
for machines that already run it with Shared Memory Support enabled.

### Why no driver is bundled

CPU package power and die temperature live in model-specific registers readable
only from ring 0. The driver most monitoring tools have historically used,
WinRing0, carries a privilege-escalation CVE, sits on Microsoft's
vulnerable-driver blocklist, and is now flagged by Defender as
`VulnerableDriver:WinNT/Winring0`. Shipping it would leave an exploitable driver
on every machine this tool touches. PawnIO is a deliberate, removable choice the
operator makes.
