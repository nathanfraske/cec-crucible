; cec-crucible installer (Inno Setup 6)
;
; Produces a single self-contained cec-crucible-<ver>-setup.exe carrying the
; whole payload: our binary, LibreHardwareMonitor, PresentMon, and the licence
; texts. No archive to unpack, a real Add/Remove Programs entry, and an
; uninstaller that removes everything we created.
;
; Silent install, for imaging a bench or scripting a rollout:
;     cec-crucible-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
;     cec-crucible-setup.exe /VERYSILENT /MERGETASKS="cpusensors"   also PawnIO
;     cec-crucible-setup.exe /VERYSILENT /TASKS=""                  nothing extra
; Silent uninstall:
;     "%LOCALAPPDATA%\Programs\cec-crucible\unins000.exe" /VERYSILENT
;
; Per-user by default (PrivilegesRequired=lowest): a technician should not need
; a domain admin to put a QC tool on a bench machine. The only thing that wants
; elevation is PawnIO, and its own installer asks.
;
; NOTE for editors: in [Code], comment with // only. A { } comment is terminated
; by the first } inside it, so a constant like {app} in prose silently ends the
; comment and the rest of the sentence is compiled as Pascal.

#define AppName      "CEC Crucible"
#define AppExe       "cec-crucible.exe"
#define AppPublisher "Critical Error Computing"
#define AppUrl       "https://github.com/nathanfraske/cec-crucible"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Payload
  #define Payload "stage"
#endif

[Setup]
AppId={{8E4C1A93-6D2F-4B7A-9C31-5F0E2D8A7B64}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
DefaultDirName={localappdata}\Programs\cec-crucible
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=.
OutputBaseFilename=cec-crucible-{#AppVersion}-win-x64-setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile={#Payload}\LICENSE
; The uninstaller is what makes this better than a zip; make it easy to find.
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExe}
ChangesEnvironment=yes
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath";   Description: "Add cec-crucible to my &PATH (run it from any terminal)"; GroupDescription: "Command line:"
; Unchecked by default. This puts a kernel module on the machine, and that
; should be a decision somebody makes rather than a default they inherit.
Name: "cpusensors";  Description: "Install &PawnIO for CPU package power and die temperature"; GroupDescription: "Sensors:"; Flags: unchecked

[Files]
Source: "{#Payload}\{#AppExe}";              DestDir: "{app}"; Flags: ignoreversion
Source: "{#Payload}\PresentMon.exe";         DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Payload}\README.md";              DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Payload}\LICENSE";                DestDir: "{app}"; Flags: ignoreversion
Source: "{#Payload}\RELEASE-NOTES.md";       DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#Payload}\THIRD-PARTY-NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; Licence texts travel with the binaries they cover - MPL-2.0 requires the
; notice to survive redistribution, and a URL in a notices file does not satisfy
; that.
Source: "{#Payload}\licenses\*";             DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
; LibreHardwareMonitor in its own directory so the boundary is visible on disk
; and the uninstaller removes exactly it.
Source: "{#Payload}\LibreHardwareMonitor\*"; DestDir: "{app}\LibreHardwareMonitor"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{group}\Release notes";    Filename: "{app}\RELEASE-NOTES.md"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Parameters: "sensors"; Description: "Show what this machine can measure"; Flags: postinstall nowait skipifsilent shellexec unchecked

[Code]
const
  PawnIoUrl    = 'https://github.com/namazso/PawnIO.Setup/releases/download/2.2.0/PawnIO_setup.exe';
  PawnIoSha256 = '1f519a22e47187f70a1379a48ca604981c4fcf694f4e65b734aaa74a9fba3032';

// ---------------------------------------------------------------------------
// PATH, user scope, idempotent.
//
// Written by hand rather than with a [Registry] entry so uninstall can remove
// exactly our entry and leave the rest of the operator's PATH alone.
// ---------------------------------------------------------------------------
function PathHas(const Existing, Dir: string): Boolean;
begin
  Result := Pos(';' + Lowercase(Dir) + ';', ';' + Lowercase(Existing) + ';') > 0;
end;

procedure AddToPath(const Dir: string);
var
  Cur: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Cur) then
    Cur := '';
  if PathHas(Cur, Dir) then Exit;
  if (Cur <> '') and (Cur[Length(Cur)] <> ';') then Cur := Cur + ';';
  RegWriteExpandStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Cur + Dir);
end;

procedure RemoveFromPath(const Dir: string);
var
  Cur, Rebuilt, Part: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Cur) then Exit;
  Rebuilt := '';
  while Cur <> '' do
  begin
    P := Pos(';', Cur);
    if P = 0 then
    begin
      Part := Cur;
      Cur := '';
    end
    else
    begin
      Part := Copy(Cur, 1, P - 1);
      Cur := Copy(Cur, P + 1, Length(Cur));
    end;
    if (Part <> '') and (CompareText(Part, Dir) <> 0) then
    begin
      if Rebuilt <> '' then Rebuilt := Rebuilt + ';';
      Rebuilt := Rebuilt + Part;
    end;
  end;
  RegWriteExpandStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Rebuilt);
end;

// ---------------------------------------------------------------------------
// PawnIO: downloaded and hash-verified, never redistributed.
//
// The driver source is GPL-2.0-or-later, but the SIGNED setup - the only build
// that will load - has no stated licence, so we fetch it from the official
// release rather than shipping it. See licenses/THIRD-PARTY-BOUNDARY.md.
//
// A hash mismatch aborts the step. This one ends up in ring 0; silently
// accepting a substituted binary is the supply-chain attack worth refusing.
// ---------------------------------------------------------------------------
function PawnIoPresent: Boolean;
begin
  Result := FileExists(ExpandConstant('{commonpf}\PawnIO\PawnIOLib.dll'));
end;

procedure InstallPawnIo;
var
  Code: Integer;
begin
  if PawnIoPresent then
  begin
    Log('PawnIO already present; nothing to do');
    Exit;
  end;
  try
    DownloadTemporaryFile(PawnIoUrl, 'PawnIO_setup.exe', PawnIoSha256, nil);
  except
    // DownloadTemporaryFile raises on a hash mismatch as well as on a network
    // failure, and both mean the same thing here: do not install it.
    Log('PawnIO download/verify failed: ' + GetExceptionMessage);
    if not WizardSilent then
      MsgBox('Could not download or verify PawnIO, so CPU package power will not'#13#10 +
             'be available. Everything else still works.'#13#10#13#10 +
             'You can add it later with:'#13#10 +
             '    winget install -e --id namazso.PawnIO',
             mbInformation, MB_OK);
    Exit;
  end;
  if not Exec(ExpandConstant('{tmp}\PawnIO_setup.exe'), '/S', '', SW_SHOW,
              ewWaitUntilTerminated, Code) then
    Log('PawnIO setup failed to launch, code ' + IntToStr(Code));
end;

// Called when each install step finishes.
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('addtopath') then
      AddToPath(ExpandConstant('{app}'));
    if WizardIsTaskSelected('cpusensors') then
      InstallPawnIo;
  end;
end;

// ---------------------------------------------------------------------------
// Uninstall: leave nothing of ours behind.
//
// Settings live in %APPDATA% deliberately (they follow a technician between
// machines), so removing the install directory does not remove them - and
// leaving them behind would be a lie about "fully uninstalled". An ETW session
// outlives the process that created it, so a run killed mid-flight can leave
// one recording in the kernel; uninstalling the tool that owns it would strand
// it there permanently.
//
// PawnIO is deliberately NOT removed: FanControl and a separately-installed
// LibreHardwareMonitor use it too, and taking it out from under them would be
// rude. It has its own entry in Add/Remove Programs.
// ---------------------------------------------------------------------------
procedure StopBundledDaemon;
var
  Code: Integer;
  Cmd: string;
begin
  // Only ours - matched by path - so a copy the operator installed separately
  // keeps running. Not wmic: it is removed on current Windows 11 builds.
  Cmd := '-NoProfile -ExecutionPolicy Bypass -Command "Get-Process LibreHardwareMonitor'
       + ' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '''
       + ExpandConstant('{app}') + '\*'' } | Stop-Process -Force"';
  Exec('powershell.exe', Cmd, '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure StopLeftoverEtwSessions;
var
  Code: Integer;
begin
  Exec(ExpandConstant('{cmd}'), '/c logman stop cec-crucible -ets', '', SW_HIDE,
       ewWaitUntilTerminated, Code);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Cfg: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    StopBundledDaemon;
    StopLeftoverEtwSessions;
  end;
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveFromPath(ExpandConstant('{app}'));
    Cfg := ExpandConstant('{userappdata}\cec-crucible');
    if DirExists(Cfg) then
      DelTree(Cfg, True, True, True);
    // LibreHardwareMonitor writes its config beside its own exe, inside the
    // install directory, which [Files] does not track because we did not put
    // it there.
    if DirExists(ExpandConstant('{app}\LibreHardwareMonitor')) then
      DelTree(ExpandConstant('{app}\LibreHardwareMonitor'), True, True, True);
    if DirExists(ExpandConstant('{app}')) then
      DelTree(ExpandConstant('{app}'), True, True, True);
  end;
end;
