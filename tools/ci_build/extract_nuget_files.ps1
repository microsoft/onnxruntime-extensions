# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by the Nuget Packaging Pipeline
# TODO: Setup params properly for the arguments

# arg0: Source directory that build artifacts for the various platforms are downloaded to
# arg1: Target directory that we extract the relevant files to and setup the directory structure expected by
#       /nuget/NativeNuget.nuspec
if ($args.Count -ne 2) {
  throw "Specify source and output directories respectively."
}

$artifact_download_dir = $args[0]
$nuget_sources_dir = $args[1]

# Create a directory to extract files from the various artifacts to be packed into the nuget package.
# All 'src' paths in the nuspec 'files' section for binary files should be valid once we finish extraction.
New-Item -Path $nuget_sources_dir -ItemType directory

## .zip files
# unzip directly
Get-ChildItem $artifact_download_dir -Include *.zip -Exclude onnxruntime_extensions.xcframework.*.zip |
Foreach-Object {
  $cmd = "7z.exe x $($_.FullName) -y -o$nuget_sources_dir"
  Write-Output $cmd
  Invoke-Expression -Command $cmd
}

## .tgz files
# first extract the tar file from the tgz
Get-ChildItem $artifact_download_dir -Filter *.tgz |
Foreach-Object {
  $cmd = "7z.exe x $($_.FullName) -y -o$artifact_download_dir"
  Write-Output $cmd
  Invoke-Expression -Command $cmd
}

## .tar files
Get-ChildItem $artifact_download_dir -Filter *.tar |
Foreach-Object {
  $cmd = "7z.exe x $($_.FullName) -y -o$nuget_sources_dir"
  Write-Output $cmd
  Invoke-Expression -Command $cmd
}

# process iOS xcframework
$xcframeworks = Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifacts -Filter onnxruntime_extensions.xcframework.*.zip
if ($xcframeworks.Count -eq 1) {
  $xcframework = $xcframeworks[0]
  # remove version info from filename and use required filename format
  $target_file = "$nuget_sources_dir\onnxruntime_extensions.xcframework.zip"
  New-Item -Path $target_dir -ItemType directory

  Write-Output "Copy-Item $($xcframework.FullName) $target_file"
  Copy-Item $xcframework.FullName $target_file
}
elseif ($xcframeworks.Count -ne 1) {
  Write-Error "Expected at most one onnxruntime_ios_xcframework.*.zip file but got: [$xcframeworks]"
}

# copy android AAR.

# target file structure:
# nuget-artifact
#   onnxruntime-extensions-android
#     onnxruntime-extensions-android-x.y.z.aar  <-- this is the file we want
#  to match any {version}
$aars = Get-ChildItem $artifact_download_dir -Filter onnxruntime-extensions-android-*.aar
# could be empty if no android build
if ($aars.Count -eq 1) {
  $aar = $aars[0]
  $target_dir = "$nuget_sources_dir\onnxruntime-extensions-android-aar"
  $target_file = "$target_dir\onnxruntime-extensions.aar"  # remove -version info from filename
  New-Item -Path $target_dir -ItemType directory
  Write-Output "Copy-Item $($aar.FullName) $target_file"
  Copy-Item $aar.FullName $target_file
}


Write-Output "Get-ChildItem -Directory -Path $nuget_sources_dir\onnxruntime-extensions-*"
$ort_dirs = Get-ChildItem -Directory -Path $nuget_sources_dir\onnxruntime-extensions-*
foreach ($ort_dir in $ort_dirs)
{
  # remove the last '-xxx' segment from the dir name. typically that's the architecture.
  $dirname = Split-Path -Path $ort_dir -Leaf
  $dirname = $dirname.SubString(0,$dirname.LastIndexOf('-'))
  Write-Output "Renaming $ort_dir to $dirname"
  Rename-Item -Path $ort_dir -NewName $nuget_sources_dir\$dirname
}

# List artifacts
"Post copy artifacts"
Get-ChildItem -Recurse $nuget_sources_dir\
