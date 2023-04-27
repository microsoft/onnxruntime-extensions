# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by cpu-Nuget Packaging Pipeline

# in the native ORT nuget package
$nuget_artifacts_dir = "$Env:BUILD_BINARIESDIRECTORY\nuget-artifacts-ort-ext"
New-Item -Path $nuget_artifacts_dir -ItemType directory

## .zip files
# unzip directly
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.zip |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$nuget_artifacts_dir"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

## .tgz files
# first extract the tar file from the tgz
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tgz |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

## .tar files
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tar |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$nuget_artifacts_dir"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

# copy android AAR.
$aars = Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter onnxruntime-extensions-android-*.aar
# file structure:
# nuget-artifact
#   onnxruntime-extensions-android
#     onnxruntime-extensions-android-x.y.z.aar  <-- this is the file we want      
#     
if ($aars.Count -eq 1) {
  $aar = $aars[0]
  $target_dir = "$nuget_artifacts_dir\onnxruntime-extensions-android-aar"
  $target_file = "$target_dir\onnxruntime-extensions.aar"  # remove '-mobile' and version info from filename
  New-Item -Path $target_dir -ItemType directory

  Write-Output "Copy-Item $($aar.FullName) $target_file"
  Copy-Item $aar.FullName $target_file
}
elseif ($aars.Count -gt 1) {
  Write-Error "Expected at most one Android .aar file but got: [$aars]"
}


Write-Output "Get-ChildItem -Directory -Path $nuget_artifacts_dir\onnxruntime-extensions-*"
$ort_dirs = Get-ChildItem -Directory -Path $nuget_artifacts_dir\onnxruntime-extensions-*
foreach ($ort_dir in $ort_dirs)
{
  # remove the last '-xxx' segment from the dir name. typically that's the architecture.
  $dirname = Split-Path -Path $ort_dir -Leaf
  $dirname = $dirname.SubString(0,$dirname.LastIndexOf('-'))
  Write-Output "Renaming $ort_dir to $dirname"
  Rename-Item -Path $ort_dir -NewName $nuget_artifacts_dir\$dirname
}

# List artifacts
"Post copy artifacts"
Get-ChildItem -Recurse $nuget_artifacts_dir\