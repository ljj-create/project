param([string]$hwnd)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class DesktopHook {
  [DllImport("user32.dll", SetLastError=true)]
  public static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string win);
  [DllImport("user32.dll")]
  public static extern bool SetParent(IntPtr hWndChild, IntPtr hWndNewParent);
}
"@

$last = [IntPtr]::Zero
$target = [IntPtr]::Zero
$workerw = [IntPtr]::Zero
while ($true) {
  $workerw = [DesktopHook]::FindWindowEx([IntPtr]::Zero, $workerw, "WorkerW", $null)
  if ($workerw -eq [IntPtr]::Zero) { break }
  $last = $workerw
  $shell = [DesktopHook]::FindWindowEx($workerw, [IntPtr]::Zero, "SHELLDLL_DefView", $null)
  if ($shell -ne [IntPtr]::Zero) {
    $next = [DesktopHook]::FindWindowEx([IntPtr]::Zero, $workerw, "WorkerW", $null)
    if ($next -ne [IntPtr]::Zero) { $target = $next; break }
  }
}

if ($target -eq [IntPtr]::Zero) { $target = $last }
if ($target -eq [IntPtr]::Zero) { exit 2 }

[DesktopHook]::SetParent([IntPtr]::new([int64]$hwnd), $target) | Out-Null
exit 0
