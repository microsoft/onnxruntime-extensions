param ($target_folder)

function Link-Copy {
    param (
        $Path,
        $Target
    )
#TODO, Add a normal mode, in which the files will be copied, instead of hard-link.
    if ( -not $(Test-Path $Path)) {
        New-Item -ItemType HardLink -Path $Path -Target $Target
    }
}

write-host "Build pip package folder in: $target_folder..."
$package_root = Join-Path $target_folder "ortcustomops"
if ( -not (Test-Path $package_root)){
    mkdir $package_root
}

Copy-Item .\setup.py $target_folder
Copy-Item .\README.md $target_folder
Copy-Item .\requirements.txt $target_folder


$pysrc_dir = Join-Path $(Get-Location) "ocos\pyfunc"
Link-Copy -Path "$package_root\_ocos.py" -Target $(Join-Path $pysrc_dir "\_ocos.py")
Link-Copy -Path "$package_root\__init__.py" -Target $(Join-Path $pysrc_dir "\__init__.py")
Link-Copy -Path "$package_root\_ortcustomops.pyd" -Target $(Join-Path $target_folder "ortcustomops.dll")
