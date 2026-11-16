<#
.SYNOPSIS
    Install VoxFlow-Setup-<version>.exe silently, check what it installed, run the installed program,
    uninstall it and check what is left. For a DISPOSABLE machine: a CI runner, a VM, a spare PC.

.DESCRIPTION
    Steps (each prints one PASS / FAIL line; exit code 0 = all passed, 1 = a check failed, 2 = refused to run):

      1. Windows build >= the installer's MinVersion (voxflow.iss: 10.0.22000). Setup refuses older builds.
      2. Setup /SP- /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=..., with a timeout (the process tree is killed).
      3. The install folder holds VoxFlow.exe, VoxFlow-console.exe and unins000.exe; the uninstall registry key
         HKCU\...\Uninstall\{AppId}_is1 exists with the expected DisplayVersion; Setup wrote NO "VoxFlow" value
         to the HKCU Run key (launch at login is the app's own setting, never the installer's).
      4. The INSTALLED VoxFlow-console.exe runs with a bare environment (PATH = Windows only, no Python
         variables, a temp APPDATA): "--help", then "--selftest --fake" (stand-in models: no download, no GPU;
         it imports the UI, tries the keyboard hook and the clipboard, runs the pipeline on silence).
         The keyboard hook and the clipboard need an interactive desktop, so their result is reported and
         only fails the run with -StrictDesktop. -HubCheck adds the app's own tiny download from the model
         hub (VOXFLOW_SELFTEST_HUB=1: TLS and the cache folder as in a first run; needs internet).
      4b. -UpdatePath: the in-app update, as voxflow/updater.py runs it. Twice: the installed VoxFlow is started
         (--fake --no-hook, a temp APPDATA, HF_HUB_OFFLINE=1, VOXFLOW_UPDATES=0: no download, no update check)
         and, while it runs, the SAME installer is run again with exactly the arguments
         updater.installer_command(path, relaunch) builds:
             /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAUNCH=1|0 /LOG=<file>
         Checked: Setup exits 0 within the timeout and its log says the install succeeded; the running copy was
         closed (Restart Manager, CloseApplications=force); with /RELAUNCH=1 a new VoxFlow.exe from the install
         folder is running afterwards, with /RELAUNCH=0 none is (for 20 s). Every VoxFlow process this step
         started is stopped again.
      5. A "VoxFlow" value is written to the HKCU Run key, as the app's "Launch at login" switch would:
           -RunValue Owned    "<install folder>\VoxFlow.exe" --autostart     -> the uninstaller must remove it
           -RunValue Foreign  "<install folder>-other\VoxFlow.exe" --autostart (another copy's entry)
                                                                             -> the uninstaller must keep it
      6. unins000.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=...; the uninstaller continues in a copy
         of itself in %TEMP%, so this waits for that copy too, with a timeout.
      7. The install folder and the uninstall key are gone, the Run value is gone (Owned) or unchanged
         (Foreign), and %APPDATA%\VoxFlow (the user's settings and history) is exactly as it was. When that
         folder does not exist, a marker file is put there first so the check means something.

    The script refuses to run without -IUnderstandThisInstalls, and when VoxFlow is already installed or a
    "VoxFlow" login entry already exists: it would overwrite and then remove them.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .github\scripts\installer_smoke.ps1 `
        -Installer dist\VoxFlow-Setup-0.2.0.exe -RunValue Owned -IUnderstandThisInstalls

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .github\scripts\installer_smoke.ps1 `
        -Installer dist\VoxFlow-Setup-0.2.0.exe -RunValue Foreign -UpdatePath -IUnderstandThisInstalls
#>
[CmdletBinding()]
param(
    [string]$Installer,
    [string]$ExpectedVersion,
    [ValidateSet('Owned', 'Foreign')]
    [string]$RunValue = 'Owned',
    [string]$LogDir,
    [int]$MinBuild = 22000,
    [int]$InstallTimeoutSec = 1800,
    [int]$UninstallTimeoutSec = 600,
    [int]$AppTimeoutSec = 300,
    [switch]$HubCheck,
    [switch]$StrictDesktop,
    [switch]$SkipSelftest,
    [switch]$UpdatePath,
    [int]$UpdateTimeoutSec = 1800,
    [int]$RelaunchWaitSec = 120,
    [switch]$IUnderstandThisInstalls,
    # The real locations. Only a rehearsal of this script against a fake installer changes them.
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\VoxFlow'),
    [string]$RegistryBase = 'HKCU:\Software\Microsoft\Windows\CurrentVersion',
    [string]$ProfileDir = (Join-Path $env:APPDATA 'VoxFlow')
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$AppId = '{6C0B0B5E-5A9E-4C57-9F1D-4F4F0E2A7B31}'           # AppId in packaging/voxflow.iss
$RunName = 'VoxFlow'                                          # RUN_VALUE in voxflow/system.py
$RunKey = Join-Path $RegistryBase 'Run'
$UninstallKey = Join-Path $RegistryBase ("Uninstall\{0}_is1" -f $AppId)
$MarkerName = 'installer-smoke-marker.txt'

$script:Checks = 0
$script:Failures = New-Object System.Collections.Generic.List[string]

function Write-Info([string]$Text) { Write-Host "INFO  $Text" }

function Add-Check([string]$Name, [bool]$Ok, [string]$Detail = '') {
    $script:Checks++
    if ($Ok) {
        if ($Detail) { Write-Host "PASS  $Name ($Detail)" } else { Write-Host "PASS  $Name" }
    }
    else {
        $script:Failures.Add($Name)
        if ($Detail) { Write-Host "FAIL  $Name : $Detail" } else { Write-Host "FAIL  $Name" }
        if ($env:GITHUB_ACTIONS -eq 'true') { Write-Host "::error title=Installer test::$Name : $Detail" }
    }
}

function Stop-Refused([string]$Why) {
    Write-Host "REFUSED  $Why"
    exit 2
}

function Get-RunCommand {
    if (-not (Test-Path -LiteralPath $RunKey)) { return $null }
    $item = Get-ItemProperty -LiteralPath $RunKey
    $prop = $item.PSObject.Properties[$RunName]
    if ($null -eq $prop) { return $null }
    return [string]$prop.Value
}

function Get-Fingerprint([string]$Dir) {
    # relative path | size | last write time of every file: any change, addition or removal shows
    $lines = New-Object System.Collections.Generic.List[string]
    if (Test-Path -LiteralPath $Dir) {
        $root = (Get-Item -LiteralPath $Dir).FullName.TrimEnd('\')
        Get-ChildItem -LiteralPath $Dir -Recurse -Force -File | Sort-Object FullName | ForEach-Object {
            $lines.Add(('{0}|{1}|{2}' -f $_.FullName.Substring($root.Length), $_.Length, $_.LastWriteTimeUtc.Ticks))
        }
    }
    return ($lines -join "`n")
}

function Invoke-Program {
    # Start a program and wait at most $TimeoutSec. On a timeout the whole process tree is killed.
    # -Capture: collect stdout + stderr (console programs only: Setup hands its work to a child process).
    # -Bare: an environment with nothing of the build machine in it, and a throw-away APPDATA.
    param([string]$File, [string]$Arguments, [int]$TimeoutSec, [switch]$Capture, [switch]$Bare,
          [hashtable]$ExtraEnv = @{})
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $File
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.WorkingDirectory = $env:TEMP
    if ($Bare) {
        foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'pythonLocation',
                            'LLAMA_CPP_LIB_PATH', 'CUDA_PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM',
                            'QT_QPA_PLATFORM_PLUGIN_PATH', 'VOXFLOW_DEVICE', 'VOXFLOW_SELFTEST_HUB')) {
            $psi.EnvironmentVariables.Remove($name)
        }
        $psi.EnvironmentVariables['PATH'] = ('{0}\System32;{0}' -f $env:SystemRoot)
    }
    foreach ($key in $ExtraEnv.Keys) { $psi.EnvironmentVariables[$key] = [string]$ExtraEnv[$key] }
    if ($Capture) {
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
    }
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$proc.Start()
    $out = $null
    $err = $null
    if ($Capture) {
        $out = $proc.StandardOutput.ReadToEndAsync()
        $err = $proc.StandardError.ReadToEndAsync()
    }
    $timedOut = -not $proc.WaitForExit($TimeoutSec * 1000)
    if ($timedOut) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $proc.Id /T /F 2>&1 | Out-Null
        [void]$proc.WaitForExit(15000)
    }
    $text = ''
    if ($Capture) {
        if ($out.Wait(15000)) { $text = $out.Result }
        if ($err.Wait(15000)) { $text = $text + $err.Result }
    }
    $code = -1
    if (-not $timedOut) { $code = $proc.ExitCode }
    return [pscustomobject]@{ ExitCode = $code; TimedOut = $timedOut; Output = $text
                              Seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1) }
}

function Get-SelftestSteps([string]$Text) {
    # "[ok] label (0.12s): detail" / "[FAIL] label: reason"  ->  label = 'ok' or the failure reason
    $steps = @{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\[ok\] (.+?) \(\d+(\.\d+)?s\)') { $steps[$Matches[1]] = 'ok' }
        elseif ($line -match '^\[FAIL\] (.+?): (.*)$') { $steps[$Matches[1]] = $Matches[2] }
    }
    return $steps
}

function Get-VoxFlowProcesses([string]$ExePath) {
    # the VoxFlow.exe processes started from $ExePath (the install folder), not other copies
    $found = @()
    foreach ($p in @(Get-Process -Name 'VoxFlow' -ErrorAction SilentlyContinue)) {
        $path = $null
        try { $path = $p.Path } catch { $path = $null }
        if ($path -and ($path -ieq $ExePath)) { $found += $p }
    }
    return $found
}

function Stop-VoxFlowProcesses([string]$ExePath) {
    foreach ($p in (Get-VoxFlowProcesses $ExePath)) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop }
        catch { Write-Info "could not stop VoxFlow ($($p.Id)): $($_.Exception.Message)" }
    }
    $deadline = (Get-Date).AddSeconds(30)
    while ((@(Get-VoxFlowProcesses $ExePath).Count -gt 0) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 500 }
}

function Test-UpdatePath([bool]$Relaunch) {
    # The in-app update (voxflow/updater.py): Setup runs over a running VoxFlow with the updater's own arguments.
    $flag = 0
    if ($Relaunch) { $flag = 1 }
    $tag = "relaunch=$flag"
    $log = Join-Path $LogDir ("update-relaunch{0}.log" -f $flag)
    $appData = Join-Path $LogDir ("update-appdata-{0}" -f $flag)
    New-Item -ItemType Directory -Force -Path $appData | Out-Null
    # what the running copy and (through Setup) the relaunched one get: nothing downloads, nothing checks GitHub
    $safeEnv = @{ APPDATA = $appData; HF_HUB_CACHE = (Join-Path $LogDir 'hf-cache'); HF_HUB_OFFLINE = '1'
                  HF_HUB_DISABLE_TELEMETRY = '1'; VOXFLOW_UPDATES = '0'; VOXFLOW_SPEED_CHECK = '0'
                  PYINSTALLER_RESET_ENVIRONMENT = '1' }
    Stop-VoxFlowProcesses $exe

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = '--fake --no-hook'
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $env:TEMP
    $psi.EnvironmentVariables['PATH'] = ('{0}\System32;{0}' -f $env:SystemRoot)
    foreach ($key in $safeEnv.Keys) { $psi.EnvironmentVariables[$key] = [string]$safeEnv[$key] }
    $running = [System.Diagnostics.Process]::Start($psi)
    Start-Sleep -Seconds 15                                   # past the start-up: the tray, the lock file
    $alive = -not $running.HasExited
    Add-Check "update path ($tag): the installed VoxFlow is running before the update" $alive "pid $($running.Id)"
    if (-not $alive) {
        Write-Info "VoxFlow exited early with code $($running.ExitCode); see $appData\VoxFlow\logs"
        return
    }
    $oldPid = $running.Id

    # exactly updater.installer_command(path, relaunch, log_path)
    $arguments = ('/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAUNCH={0} /LOG="{1}"' -f $flag, $log)
    Write-Info "update path ($tag): $(Split-Path -Leaf $Installer) $arguments"
    $r = Invoke-Program -File $Installer -Arguments $arguments -TimeoutSec $UpdateTimeoutSec -Bare -ExtraEnv $safeEnv
    if ($r.TimedOut) {
        Add-Check "update path ($tag): Setup finished" $false "no exit after $UpdateTimeoutSec s; the process tree was killed"
        Stop-VoxFlowProcesses $exe
        return
    }
    Add-Check "update path ($tag): Setup finished" ($r.ExitCode -eq 0) "exit code $($r.ExitCode) after $($r.Seconds) s"
    $logText = ''
    if (Test-Path -LiteralPath $log) { $logText = [string](Get-Content -LiteralPath $log -Raw) }
    Add-Check "update path ($tag): the Setup log says the install succeeded" ($logText -match 'Installation process succeeded') $log
    $shown = [string](Get-ItemProperty -LiteralPath $UninstallKey).DisplayVersion
    Add-Check "update path ($tag): DisplayVersion is the release version" ($shown -eq $ExpectedVersion) "'$shown'"

    $closed = $running.WaitForExit(30000)
    Add-Check "update path ($tag): Setup closed the running VoxFlow" $closed "pid $oldPid"

    # Setup starts VoxFlow ("nowait") just before it exits: give it time to appear
    $wait = $RelaunchWaitSec
    if (-not $Relaunch) { $wait = 20 }
    $deadline = (Get-Date).AddSeconds($wait)
    $new = @()
    while ((Get-Date) -lt $deadline) {
        $new = @(Get-VoxFlowProcesses $exe | Where-Object { $_.Id -ne $oldPid })
        if ($Relaunch -and $new.Count -gt 0) { break }
        Start-Sleep -Milliseconds 500
    }
    $pids = ($new | ForEach-Object { $_.Id }) -join ', '
    if ($Relaunch) {
        Add-Check "update path ($tag): VoxFlow was started again" ($new.Count -gt 0) "new pid(s): '$pids' within $wait s"
    }
    else {
        Add-Check "update path ($tag): VoxFlow was NOT started again" ($new.Count -eq 0) "new pid(s): '$pids' within $wait s"
    }
    Stop-VoxFlowProcesses $exe
    Add-Check "update path ($tag): no VoxFlow process is left" (@(Get-VoxFlowProcesses $exe).Count -eq 0)
}

# ------------------------------------------------------------------------------------------- may this run?
if (-not $IUnderstandThisInstalls) {
    Stop-Refused ('this script INSTALLS and then UNINSTALLS VoxFlow for the current user and writes to the ' +
                  'HKCU Run key. Use it on a disposable machine only, and pass -IUnderstandThisInstalls.')
}
if (-not $Installer) { Stop-Refused 'pass -Installer <path to VoxFlow-Setup-<version>.exe>.' }
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) { Stop-Refused "installer not found: $Installer" }
$Installer = (Get-Item -LiteralPath $Installer).FullName
if (-not $ExpectedVersion) {
    if ((Split-Path -Leaf $Installer) -match '^VoxFlow-Setup-(.+)\.exe$') { $ExpectedVersion = $Matches[1] }
    else { Stop-Refused 'the version is not in the installer file name: pass -ExpectedVersion.' }
}
if (Test-Path -LiteralPath $UninstallKey) {
    Stop-Refused "VoxFlow is already installed for this user ($UninstallKey). This test would remove it."
}
if (Test-Path -LiteralPath $InstallDir) {
    Stop-Refused "$InstallDir already exists. This test would overwrite and then remove it."
}
$existing = Get-RunCommand
if ($null -ne $existing) {
    Stop-Refused "a '$RunName' login entry already exists ($existing). This test would overwrite and remove it."
}
if (-not $LogDir) { $LogDir = Join-Path $env:TEMP ('voxflow-installer-test-' + $RunValue.ToLower()) }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogDir = (Get-Item -LiteralPath $LogDir).FullName
$installLog = Join-Path $LogDir 'install.log'
$uninstallLog = Join-Path $LogDir 'uninstall.log'

Write-Host "== VoxFlow installer test: $(Split-Path -Leaf $Installer), login entry '$RunValue' =="
Write-Info "install folder  $InstallDir"
Write-Info "uninstall key   $UninstallKey"
Write-Info "logs            $LogDir"

$wroteRun = $null          # the Run command this script wrote (removed again at the end if it is still there)
$madeProfile = $false      # this script created $ProfileDir (removed again at the end)
$exe = Join-Path $InstallDir 'VoxFlow.exe'
$console = Join-Path $InstallDir 'VoxFlow-console.exe'
$uninstaller = Join-Path $InstallDir 'unins000.exe'

try {
    # --------------------------------------------------------------------------------------- 1. Windows build
    $os = [System.Environment]::OSVersion.Version
    $caption = ''
    try { $caption = (Get-CimInstance Win32_OperatingSystem).Caption } catch { $caption = 'Windows' }
    $buildOk = ($os.Major -gt 10) -or ($os.Major -eq 10 -and $os.Build -ge $MinBuild)
    Add-Check 'Windows build meets the installer minimum' $buildOk "$caption, build $($os.Build); minimum $MinBuild"
    if (-not $buildOk) {
        throw "Setup refuses to run on this Windows (MinVersion=10.0.$MinBuild in packaging/voxflow.iss). " +
              'In CI: use a runner image with Windows Server 2025 (build 26100) or later, e.g. runs-on: windows-2025.'
    }

    # --------------------------------------------------------------------------------------- the user profile
    if (Test-Path -LiteralPath $ProfileDir) {
        Write-Info "$ProfileDir exists: it must be unchanged at the end"
    }
    else {
        New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
        Set-Content -LiteralPath (Join-Path $ProfileDir $MarkerName) -Value 'put here by installer_smoke.ps1' -Encoding ASCII
        $madeProfile = $true
        Write-Info "$ProfileDir did not exist: created with a marker file, it must survive the uninstall"
    }
    $profileBefore = Get-Fingerprint $ProfileDir

    # --------------------------------------------------------------------------------------- 2. install
    $r = Invoke-Program -File $Installer -TimeoutSec $InstallTimeoutSec `
        -Arguments ('/SP- /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG="{0}"' -f $installLog)
    if ($r.TimedOut) {
        Add-Check 'Setup finished' $false "no exit after $InstallTimeoutSec s; the process tree was killed. See install.log"
        throw 'the installer timed out'
    }
    Add-Check 'Setup finished' ($r.ExitCode -eq 0) "exit code $($r.ExitCode) after $($r.Seconds) s"
    if ($r.ExitCode -ne 0) {
        throw "the installer failed (exit code $($r.ExitCode); 1 = Setup failed to initialise, e.g. the Windows version check). See install.log"
    }

    # --------------------------------------------------------------------------------------- 3. what it installed
    foreach ($file in @($exe, $console, $uninstaller)) {
        Add-Check "installed: $(Split-Path -Leaf $file)" (Test-Path -LiteralPath $file -PathType Leaf) $file
    }
    $keyThere = Test-Path -LiteralPath $UninstallKey
    Add-Check 'uninstall registry key exists' $keyThere $UninstallKey
    if ($keyThere) {
        $props = Get-ItemProperty -LiteralPath $UninstallKey
        $shown = $props.PSObject.Properties['DisplayVersion']
        $version = ''
        if ($null -ne $shown) { $version = [string]$shown.Value }
        Add-Check 'DisplayVersion is the release version' ($version -eq $ExpectedVersion) "'$version', expected '$ExpectedVersion'"
        $where = $props.PSObject.Properties['InstallLocation']
        if ($null -ne $where) {
            $same = ([string]$where.Value).TrimEnd('\') -ieq $InstallDir.TrimEnd('\')
            Add-Check 'InstallLocation is the per-user program folder' $same ([string]$where.Value)
        }
    }
    $afterInstall = Get-RunCommand
    Add-Check 'Setup did not create a login entry (HKCU Run value)' ($null -eq $afterInstall) ([string]$afterInstall)
    if ($null -ne $afterInstall) { throw 'the installer wrote to the Run key; stopping before this script touches it' }

    # --------------------------------------------------------------------------------------- 4. run the installed app
    $appEnv = @{ APPDATA = (Join-Path $LogDir 'appdata'); HF_HUB_CACHE = (Join-Path $LogDir 'hf-cache')
                 HF_HUB_DISABLE_TELEMETRY = '1' }
    New-Item -ItemType Directory -Force -Path $appEnv.APPDATA | Out-Null
    if (Test-Path -LiteralPath $console -PathType Leaf) {
        $r = Invoke-Program -File $console -Arguments '--help' -TimeoutSec $AppTimeoutSec -Capture -Bare -ExtraEnv $appEnv
        Set-Content -LiteralPath (Join-Path $LogDir 'help.txt') -Value $r.Output -Encoding UTF8
        $helpOk = (-not $r.TimedOut) -and ($r.ExitCode -eq 0) -and ($r.Output -match 'usage:') -and ($r.Output -match '--selftest')
        Add-Check 'installed VoxFlow-console.exe --help' $helpOk "exit code $($r.ExitCode), $($r.Seconds) s, timed out: $($r.TimedOut)"
        if (-not $helpOk) { Write-Host $r.Output }

        if (-not $SkipSelftest) {
            if ($HubCheck) { $appEnv['VOXFLOW_SELFTEST_HUB'] = '1' }
            $r = Invoke-Program -File $console -Arguments '--selftest --fake' -TimeoutSec $AppTimeoutSec -Capture -Bare -ExtraEnv $appEnv
            Set-Content -LiteralPath (Join-Path $LogDir 'selftest.txt') -Value $r.Output -Encoding UTF8
            Write-Host '----- VoxFlow-console.exe --selftest --fake'
            Write-Host $r.Output.TrimEnd()
            Write-Host '-----'
            Add-Check 'installed self-test ran to its verdict' ((-not $r.TimedOut) -and ($r.Output -match '(?m)^SELFTEST (PASSED|FAILED)')) "exit code $($r.ExitCode), $($r.Seconds) s, timed out: $($r.TimedOut)"
            $steps = Get-SelftestSteps $r.Output
            $required = @('UI imports', 'speech model load', 'cleanup model load')
            if ($HubCheck) { $required += 'model hub download' }
            foreach ($label in $required) {
                $state = 'did not run'
                if ($steps.ContainsKey($label)) { $state = $steps[$label] }
                Add-Check "installed self-test: $label" ($state -eq 'ok') $state
            }
            foreach ($label in @('keyboard hook', 'clipboard')) {       # these need an interactive desktop
                $state = 'did not run'
                if ($steps.ContainsKey($label)) { $state = $steps[$label] }
                if ($StrictDesktop) { Add-Check "installed self-test: $label" ($state -eq 'ok') $state }
                elseif ($state -eq 'ok') { Write-Info "installed self-test: $label ok" }
                else {
                    Write-Host "WARN  installed self-test: $label : $state (not counted: needs an interactive desktop; -StrictDesktop counts it)"
                    if ($env:GITHUB_ACTIONS -eq 'true') { Write-Host "::warning title=Installer test::self-test '$label' : $state" }
                }
            }
        }
    }

    # --------------------------------------------------------------------------------------- 4b. the in-app update
    if ($UpdatePath) {
        Test-UpdatePath $true
        Test-UpdatePath $false
        foreach ($file in @($exe, $console, $uninstaller)) {
            Add-Check "after the update: $(Split-Path -Leaf $file)" (Test-Path -LiteralPath $file -PathType Leaf) $file
        }
    }

    # --------------------------------------------------------------------------------------- 5. the login entry
    if ($RunValue -eq 'Owned') { $target = $exe }
    else { $target = Join-Path ($InstallDir.TrimEnd('\') + '-other') 'VoxFlow.exe' }
    $command = '"{0}" --autostart' -f $target
    if (-not (Test-Path -LiteralPath $RunKey)) { New-Item -Path $RunKey -Force | Out-Null }
    New-ItemProperty -LiteralPath $RunKey -Name $RunName -Value $command -PropertyType String -Force | Out-Null
    $wroteRun = $command
    Add-Check "login entry written ($RunValue)" ((Get-RunCommand) -eq $command) $command

    # --------------------------------------------------------------------------------------- 6. uninstall
    if (-not (Test-Path -LiteralPath $uninstaller -PathType Leaf)) { throw 'unins000.exe is missing: cannot uninstall' }
    $r = Invoke-Program -File $uninstaller -TimeoutSec $UninstallTimeoutSec `
        -Arguments ('/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG="{0}"' -f $uninstallLog)
    if ($r.TimedOut) {
        Add-Check 'uninstaller finished' $false "no exit after $UninstallTimeoutSec s; the process tree was killed. See uninstall.log"
        throw 'the uninstaller timed out'
    }
    Add-Check 'uninstaller finished' ($r.ExitCode -eq 0) "exit code $($r.ExitCode) after $($r.Seconds) s"
    # unins000.exe hands over to a copy of itself in %TEMP% (_iu*.tmp) that does the work, so that the program
    # folder can be deleted: wait until no such process is left and the folder is gone.
    $deadline = (Get-Date).AddSeconds($UninstallTimeoutSec)
    $quietSince = $null
    while ((Get-Date) -lt $deadline) {
        $busy = @(Get-Process | Where-Object { $_.ProcessName -like '_iu*' -or $_.ProcessName -like 'unins0*' })
        if ($busy.Count -eq 0) {
            if (-not (Test-Path -LiteralPath $InstallDir)) { break }
            if ($null -eq $quietSince) { $quietSince = Get-Date }
            if (((Get-Date) - $quietSince).TotalSeconds -ge 20) { break }     # nothing is working on it any more
        }
        else { $quietSince = $null }
        Start-Sleep -Milliseconds 500
    }
    $busy = @(Get-Process | Where-Object { $_.ProcessName -like '_iu*' -or $_.ProcessName -like 'unins0*' })
    Add-Check 'no uninstaller process is left running' ($busy.Count -eq 0) (($busy | ForEach-Object { $_.ProcessName }) -join ', ')

    # --------------------------------------------------------------------------------------- 7. what is left
    $dirGone = -not (Test-Path -LiteralPath $InstallDir)
    $left = ''
    if (-not $dirGone) {
        $files = @(Get-ChildItem -LiteralPath $InstallDir -Recurse -Force | Select-Object -First 20 | ForEach-Object { $_.FullName })
        $left = "left behind (first 20): " + ($files -join '; ')
    }
    Add-Check 'program folder is gone' $dirGone $left
    Add-Check 'uninstall registry key is gone' (-not (Test-Path -LiteralPath $UninstallKey)) $UninstallKey
    $afterUninstall = Get-RunCommand
    if ($RunValue -eq 'Owned') {
        Add-Check 'the uninstaller removed its own login entry' ($null -eq $afterUninstall) ([string]$afterUninstall)
    }
    else {
        Add-Check "the uninstaller kept another copy's login entry" ($afterUninstall -eq $command) ("now: '$afterUninstall'")
    }
    $profileAfter = Get-Fingerprint $ProfileDir
    Add-Check 'the user profile folder is untouched' ((Test-Path -LiteralPath $ProfileDir) -and ($profileAfter -eq $profileBefore)) $ProfileDir
}
catch {
    Add-Check 'test completed' $false $_.Exception.Message
}
finally {
    # put back what this script itself added (never anything else)
    try { Stop-VoxFlowProcesses $exe } catch { Write-Host "WARN  could not stop VoxFlow: $($_.Exception.Message)" }
    try {
        if ($null -ne $wroteRun -and (Get-RunCommand) -eq $wroteRun) {
            Remove-ItemProperty -LiteralPath $RunKey -Name $RunName -ErrorAction SilentlyContinue
            Write-Info 'removed the login entry this test wrote'
        }
        if ($madeProfile -and (Test-Path -LiteralPath $ProfileDir)) {
            $marker = Join-Path $ProfileDir $MarkerName
            if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker -Force }
            if (@(Get-ChildItem -LiteralPath $ProfileDir -Force).Count -eq 0) { Remove-Item -LiteralPath $ProfileDir -Force }
        }
    }
    catch { Write-Host "WARN  clean-up: $($_.Exception.Message)" }
    if (Test-Path -LiteralPath $InstallDir) {
        Write-Host "WARN  $InstallDir is still there (left for inspection). Remove it with its unins000.exe or by hand."
    }
}

if ($script:Failures.Count -eq 0) {
    Write-Host "RESULT: PASS ($($script:Checks) checks, login entry '$RunValue')"
    exit 0
}
Write-Host "RESULT: FAIL ($($script:Failures.Count) of $($script:Checks) checks failed: $($script:Failures -join '; '))"
exit 1
