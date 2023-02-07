# Find all installed Visual Studio with Cmake, and only output the vsdevcmd path of the latest one
PARAM([int]$outputEnv = 0)
$latest_valid_vs = ""
function choose_latter_vs {
    param([string]$path)
    if ($global:latest_valid_vs -lt $path) {
        $cmake_path = $path + "\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path -Path $cmake_path) {
            $global:latest_valid_vs = $path
        }
    }
}

$vswherepath=[Environment]::GetEnvironmentVariable(
    "ProgramFiles(x86)") + "\Microsoft Visual Studio\Installer\vswhere.exe"
$vs_installed = & "$vswherepath" -latest -property installationPath

FOREACH ($line in $vs_installed) {
    choose_latter_vs($line.Trim())
}

if ($latest_valid_vs.Length -eq 0) {
    exit 1
}

$pathDevCmd = $latest_valid_vs + "\Common7\Tools\vsdevcmd.bat"
if ($outputEnv) {
    cmd /q /c "$pathDevCmd & set"
}
else {
    $pathDevCmd
}
