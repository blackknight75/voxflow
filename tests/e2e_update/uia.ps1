# Windows UI Automation driver for the desktop update-UI tests (test_update_desktop.py). It finds controls the way a
# screen reader does (the UIA tree of the process, by Name) and acts only through UIA patterns: NO synthetic input
# (no SendInput / mouse_event / keybd_event / SendKeys), the cursor never moves. Output: one JSON object, written to
# -OutFile (or stdout).
#   -Action dump                      every element of the process's windows (name, type, automation id, rect, ...)
#   -Action find    -Name <n>         the first enabled on-screen element called <n> (or found=false); -Type
#                                     ControlType.Button (etc.) narrows it to that control type
#   -Action invoke  -Name <n>         InvokePattern.Invoke() on it
#   -Action focused                   the element that has the keyboard focus (AutomationElement.FocusedElement)
#   -Action windows                   the process's top-level windows (a popup menu is one of them)
#   -Action toast   -Name <text>      read-only search of the other top-level windows for an element whose name
#                                     contains <text> (where Windows shows a notification); with -Invoke, also
#                                     InvokePattern.Invoke() on it (or its nearest ancestor that has the pattern)
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$Action,
    [string]$Name = '',
    [int]$TimeoutMs = 5000,
    [string]$OutFile = '',
    [switch]$Invoke,
    [string]$Type = ''
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -ReferencedAssemblies UIAutomationClient, UIAutomationTypes, WindowsBase @'
using System;
using System.Reflection;
using System.Runtime.InteropServices;
public static class VfUia {
    [DllImport("UIAutomationCore.dll")]
    static extern int UiaGetPropertyValue(SafeHandle hnode, int propertyId, [MarshalAs(UnmanagedType.Struct)] out object value);
    // UIA_FullDescriptionPropertyId (Windows 10 1703+): Qt puts QAccessible::Description there; the managed client
    // predates it, so it is read through the element's node with the flat UIA API (still read-only)
    public static string FullDescription(System.Windows.Automation.AutomationElement e) {
        var f = typeof(System.Windows.Automation.AutomationElement).GetField("_hnode", BindingFlags.NonPublic | BindingFlags.Instance);
        var h = f == null ? null : f.GetValue(e) as SafeHandle;
        if (h == null) return null;
        object v;
        if (UiaGetPropertyValue(h, 30159, out v) != 0) return null;
        return v as string;
    }
}
'@
$A = [System.Windows.Automation.AutomationElement]
$TS = [System.Windows.Automation.TreeScope]
$True_ = [System.Windows.Automation.Condition]::TrueCondition

function Describe($e) {
    $c = $e.Current
    $r = $c.BoundingRectangle
    $rect = $null
    if (-not $r.IsEmpty) { $rect = @([int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height) }
    $pats = @($e.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName -replace 'PatternIdentifiers.Pattern', '' })
    @{ name = $c.Name; type = $c.ControlType.ProgrammaticName; id = $c.AutomationId; cls = $c.ClassName;
       enabled = $c.IsEnabled; focusable = $c.IsKeyboardFocusable; focused = $c.HasKeyboardFocus;
       offscreen = $c.IsOffscreen; rect = $rect; help = $c.HelpText; pid = $c.ProcessId; patterns = $pats;
       description = $(try { [VfUia]::FullDescription($e) } catch { "error: $_" }) }
}

function Windows-Of([int]$procId) {
    $cond = New-Object System.Windows.Automation.PropertyCondition($A::ProcessIdProperty, $procId)
    @($A::RootElement.FindAll($TS::Children, $cond))
}

function Find-Named([string]$n) {
    $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
    do {
        foreach ($w in Windows-Of $ProcessId) {
            $cond = New-Object System.Windows.Automation.PropertyCondition($A::NameProperty, $n)
            foreach ($e in @($w.FindAll($TS::Subtree, $cond))) {
                if ($Type -and $e.Current.ControlType.ProgrammaticName -ne $Type) { continue }
                if ($e.Current.IsEnabled -and -not $e.Current.IsOffscreen) { return $e }
            }
        }
        Start-Sleep -Milliseconds 150
    } while ((Get-Date) -lt $deadline)
    return $null
}

function Invoke-It($e) {
    $p = $null
    if ($e.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$p)) { $p.Invoke(); return 'InvokePattern' }
    return ''
}

$res = @{ action = $Action; name = $Name }
switch ($Action) {
    'dump' {
        $items = @()
        foreach ($w in Windows-Of $ProcessId) {
            foreach ($e in @($w.FindAll($TS::Subtree, $True_))) { $items += Describe $e }
        }
        $res.items = $items
    }
    'windows' { $res.items = @(Windows-Of $ProcessId | ForEach-Object { Describe $_ }) }
    'find' {
        $e = Find-Named $Name
        $res.found = $null -ne $e
        if ($e) { $res.element = Describe $e }
    }
    'invoke' {
        $e = Find-Named $Name
        $res.found = $null -ne $e
        if ($e) { $res.element = Describe $e; $res.done = Invoke-It $e }
    }
    'focused' {
        $e = $A::FocusedElement
        $res.found = $null -ne $e
        if ($e) { $res.element = Describe $e }
    }
    'toast' {
        $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
        $hit = $null
        $scanned = @()
        do {
            foreach ($w in @($A::RootElement.FindAll($TS::Children, $True_))) {
                try {
                    if ($w.Current.ProcessId -eq $ProcessId) { continue }
                    if (-not ($scanned -contains $w.Current.ClassName)) { $scanned += $w.Current.ClassName }
                    foreach ($e in @($w.FindAll($TS::Subtree, $True_))) {
                        if ($e.Current.Name -and $e.Current.Name.Contains($Name)) { $hit = $e; break }
                    }
                } catch { }
                if ($hit) { break }
            }
            if (-not $hit) { Start-Sleep -Milliseconds 300 }
        } while (-not $hit -and (Get-Date) -lt $deadline)
        $res.found = $null -ne $hit
        $res.scanned_classes = $scanned
        if ($hit) {
            $res.element = Describe $hit
            $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
            $chain = @(); $cur = $hit
            for ($i = 0; $i -lt 6 -and $cur; $i++) { $chain += Describe $cur; $cur = $walker.GetParent($cur) }
            $res.ancestors = $chain
            if ($Invoke) {
                $cur = $hit; $done = ''
                for ($i = 0; $i -lt 6 -and $cur -and -not $done; $i++) {
                    $done = Invoke-It $cur
                    if (-not $done) { $cur = $walker.GetParent($cur) }
                }
                $res.done = $done
            }
        }
    }
    default { throw "unknown action $Action" }
}
$json = $res | ConvertTo-Json -Depth 6 -Compress
if ($OutFile) { [System.IO.File]::WriteAllText($OutFile, $json, (New-Object System.Text.UTF8Encoding($false))) } else { $json }
