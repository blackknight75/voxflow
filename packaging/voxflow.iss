; Inno Setup 6 script for VoxFlow. Built by scripts/build_installer.py, which passes:
;   /DAppVersion=<voxflow.__version__>  /DSourceDir=<dist\VoxFlow>  /DOutDir=<dist>  /DIconFile=<.ico>
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\VoxFlow"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif
#define AppURL "https://github.com/blackknight75/voxflow"
; The app's own "Launch at login" entry (HKCU): RUN_KEY / RUN_VALUE in voxflow/system.py. Only the uninstaller uses
; these, to remove the entry (Code section at the end); tests/test_packaging.py keeps them equal to the app's constants.
#define RunKey "Software\Microsoft\Windows\CurrentVersion\Run"
#define RunValue "VoxFlow"

[Setup]
; Stable id: a newer installer installs over the old version. The uninstall registry key is "<AppId>_is1" under
; HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall; packaging/winget/*.installer.yaml names it as ProductCode.
AppId={{6C0B0B5E-5A9E-4C57-9F1D-4F4F0E2A7B31}
AppName=VoxFlow
AppVersion={#AppVersion}
AppVerName=VoxFlow {#AppVersion}
AppPublisher=VoxFlow
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\Programs\VoxFlow
DefaultGroupName=VoxFlow
DisableProgramGroupPage=yes
; Per-user: never a UAC prompt and no "for me / for all users" question. /ALLUSERS on the command line still works.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
; Windows 11 (build 22000) or later; Setup refuses to start on anything older. The same floor as MinimumOSVersion
; in packaging/winget/*.installer.yaml.
MinVersion=10.0.22000
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=VoxFlow-Setup-{#AppVersion}
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\VoxFlow.exe
UninstallDisplayName=VoxFlow
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
; Wizard: folder page -> shortcuts page (its button reads "Install") -> finish. Silent: /VERYSILENT /SUPPRESSMSGBOXES.
DisableWelcomePage=yes
DisableReadyPage=yes
; Upgrade over a running copy: Restart Manager asks VoxFlow (it lives in the tray) to close and "force" ends it if it
; does not answer; otherwise the files in use would only be replaced after a reboot. It is not restarted afterwards.
CloseApplications=force
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no
; Held while Setup runs: a VoxFlow started meanwhile (e.g. right after "Install when I quit") sees it and exits with
; "Finishing the update..." instead of running from half-replaced files (voxflow/system.py, SETUP_MUTEX).
SetupMutex=VoxFlowSetupRunning,Global\VoxFlowSetupRunning
VersionInfoVersion={#AppVersion}
VersionInfoProductName=VoxFlow
VersionInfoDescription=VoxFlow installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[InstallDelete]
; An upgrade starts from an empty runtime folder: DLLs, .pyd files and *.dist-info folders of packages that an older
; version had and this one does not would otherwise pile up (and two dist-info folders of one package confuse
; importlib.metadata). VoxFlow is closed by then (PrepareToInstall, Restart Manager). Nothing of the user's is in it.
; Normally the folder is already gone here: KeepOldCopy (Code section) has moved it to {app}\previous, from where it
; is put back if Setup does not finish. This entry empties it when it could not be moved.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\VoxFlow"; Filename: "{app}\VoxFlow.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\VoxFlow"; Filename: "{app}\VoxFlow.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\VoxFlow.exe"; Description: "Launch VoxFlow"; Flags: nowait postinstall skipifsilent
; The in-app updater (voxflow/updater.py) runs Setup with /VERYSILENT /SUPPRESSMSGBOXES /NORESTART after VoxFlow has
; quit, and with /RELAUNCH=1 when the user chose "Install now" (or a scheduled install): VoxFlow then starts again.
; --after-update: Setup still holds its mutex at this point, so VoxFlow must not wait for it to go.
Filename: "{app}\VoxFlow.exe"; Parameters: "--after-update"; Flags: nowait; Check: RelaunchAfterUpdate

[UninstallDelete]
; written by CurStepChanged below for a winget install
Type: files; Name: "{app}\managed-by.txt"
; compiled GPU kernels (voxflow/modelhub.py): a cache of this PC's driver, not the user's data
Type: filesandordirs; Name: "{localappdata}\VoxFlow\cuda-cache"
Type: dirifempty; Name: "{localappdata}\VoxFlow"
; the in-app updater's download folder (voxflow/updater.py, download_folder): an installer of up to 1.8 GB and Setup's log
Type: filesandordirs; Name: "{%TEMP|{localappdata}\Temp}\VoxFlow-update"
; the old program files an interrupted upgrade set aside (KeepOldCopy below)
Type: filesandordirs; Name: "{app}\previous"

; The uninstaller removes what the installer put in {app}, plus the app's "Launch at login" entry (below). The user's
; profile (%APPDATA%\VoxFlow: settings, history, dictionary, logs) and downloaded models (the Hugging Face cache) are
; not touched. "Launch at login" is the app's own setting (Settings > General); the installer adds no autostart entry
; of its own: there is no Registry section, and Setup never writes to the Run key.

[Code]
// A running VoxFlow keeps its files locked. It is asked to quit the normal way first ("VoxFlow.exe --quit": pending
// saves are written and apps it muted get their sound back; it waits up to 5 s). Only a copy that is still running
// after that is ended, and only one started from this folder: a portable or Scoop copy elsewhere is left alone.
procedure StopRunningCopy;
var
  ResultCode: Integer;
  Folder: String;
begin
  if not FileExists(ExpandConstant('{app}\VoxFlow.exe')) then
    Exit;
  if Exec(ExpandConstant('{app}\VoxFlow.exe'), '--quit', '', SW_HIDE, ewWaitUntilTerminated, ResultCode)
     and (ResultCode = 0) then
    Exit;
  Folder := ExpandConstant('{app}\');
  StringChangeEx(Folder, '''', '''''', True);     // a quote in the path, doubled for PowerShell's '...'
  Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Get-Process -Name VoxFlow,VoxFlow-console ' +
    '-ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith(''' + Folder + ''', ' +
    '[StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

// An upgrade that does not finish (the disk fills up, a file is locked, Setup is cancelled) must not leave a VoxFlow
// that cannot start: Setup deletes the files it had copied but does not bring back the ones it replaced, and the
// runtime folder is emptied before the first file is copied (InstallDelete). So the old program files are set aside
// first: _internal is moved (same drive: no copying, no extra space until the new files arrive) and the two small
// exe files are copied to {app}\previous. RestoreOldCopy puts them back when Setup ends without having finished;
// RemoveOldCopy deletes them when it has. A {app}\previous that is already there is what an interrupted earlier run
// left: it is the complete copy and is kept. A folder that cannot be moved (a file in it is open) is not kept;
// Setup then goes on as it did before this existed.
var
  Prepared, Installed: Boolean;

procedure KeepOldCopy;
var
  App, Prev: String;
  Tries: Integer;
begin
  App := ExpandConstant('{app}');
  Prev := App + '\previous';
  if DirExists(Prev + '\_internal') or not DirExists(App + '\_internal') then
    Exit;
  if not ForceDirectories(Prev) then
    Exit;
  FileCopy(App + '\VoxFlow.exe', Prev + '\VoxFlow.exe', False);
  FileCopy(App + '\VoxFlow-console.exe', Prev + '\VoxFlow-console.exe', False);
  for Tries := 1 to 10 do
  begin
    if RenameFile(App + '\_internal', Prev + '\_internal') then
    begin
      Log('VoxFlow: the old program files are kept in ' + Prev + ' until the new ones are complete');
      Exit;
    end;
    Sleep(500);                                    // a virus scanner still reading a DLL of the copy that just quit
  end;
  Log('VoxFlow: the old program files could not be set aside');
  DelTree(Prev, True, True, True);
end;

procedure RestoreOldCopy;
var
  App, Prev: String;
begin
  App := ExpandConstant('{app}');
  Prev := App + '\previous';
  if not DirExists(Prev + '\_internal') then
    Exit;
  DelTree(App + '\_internal', True, True, True);   // what was copied of the new version
  if not RenameFile(Prev + '\_internal', App + '\_internal') then
  begin
    Log('VoxFlow: the old program files could not be put back; they stay in ' + Prev);
    Exit;
  end;
  FileCopy(Prev + '\VoxFlow.exe', App + '\VoxFlow.exe', False);
  FileCopy(Prev + '\VoxFlow-console.exe', App + '\VoxFlow-console.exe', False);
  DelTree(Prev, True, True, True);
  Log('VoxFlow: Setup did not finish, the old program files were put back');
end;

procedure RemoveOldCopy;
begin
  DelTree(ExpandConstant('{app}\previous'), True, True, True);
end;

// An upgrade by hand: VoxFlow is asked to quit before Restart Manager would end it by force (CloseApplications).
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopRunningCopy;
  KeepOldCopy;
  Prepared := True;
  Result := '';
end;

// The in-app updater asks for a restart with /RELAUNCH=1 (a normal silent install, e.g. winget's, never starts the app).
function RelaunchAfterUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

// winget's manifest (packaging/winget/*.installer.yaml) passes /MANAGEDBY=winget. The file tells VoxFlow that winget
// owns its updates, so it never replaces itself (voxflow/updater.py, install_kind). Any other value is ignored.
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  Installed := True;                               // every file is in place: the old copy is no longer needed
  RemoveOldCopy;
  if Lowercase(ExpandConstant('{param:MANAGEDBY|}')) = 'winget' then
    SaveStringToFile(ExpandConstant('{app}\managed-by.txt'), 'winget', False)
  else
    DeleteFile(ExpandConstant('{app}\managed-by.txt'));   // installed by hand over a winget install: ours now
end;

// Setup ends. If it had started to install and did not finish, the old program files come back (KeepOldCopy), and a
// VoxFlow that quit for this update (the in-app updater's /RELAUNCH=1) is started again: the old version, which then
// says that the update was not installed (voxflow/updater.py). Not after a Setup that stopped before it prepared
// anything: {app} is not known then.
procedure DeinitializeSetup;
var
  ResultCode: Integer;
begin
  if Installed or not Prepared then
    Exit;
  RestoreOldCopy;
  if RelaunchAfterUpdate and FileExists(ExpandConstant('{app}\VoxFlow.exe')) then
    Exec(ExpandConstant('{app}\VoxFlow.exe'), '--after-update', '', SW_SHOWNORMAL, ewNoWait, ResultCode);
end;

// "Launch at login" is written by the app, so the uninstall log knows nothing about it: without this, an entry
// pointing at the deleted VoxFlow.exe would stay behind. This event exists only in the uninstaller, so an install or
// an upgrade over an older version never runs it and the user's setting survives upgrades. The entry is removed only
// if it starts a program inside this install folder: a copy run from source, or a second install elsewhere, owns
// its own entry under the same name.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Command: String;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  StopRunningCopy;
  if not RegQueryStringValue(HKEY_CURRENT_USER, '{#RunKey}', '{#RunValue}', Command) then
    Exit;
  if Pos(Lowercase(AddBackslash(ExpandConstant('{app}'))), Lowercase(Command)) > 0 then
    RegDeleteValue(HKEY_CURRENT_USER, '{#RunKey}', '{#RunValue}');
end;
