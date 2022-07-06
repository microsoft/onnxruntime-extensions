# Build ONNX Runtime with onnxruntime-extensions for Java package

*The following step are demonstrated for Windows Platform only, the others like Linux and MacOS can be done similarly.*

> Android build was supported as well; check [here](https://onnxruntime.ai/docs/build/android.html#cross-compiling-on-windows) for arguments to build AAR package.

## Tools required
1. install visual studio 2022 (with cmake, git, desktop C++)
2. install android studio (https://developer.android.com/studio), and NDK
3. install miniconda to have Python support (for onnxruntime build)
4. OpenJDK: https://docs.microsoft.com/en-us/java/openjdk/download
		(OpenJDK 11.0.15 LTS)
5. Gradle: https://gradle.org/releases/
		(v6.9.2)

## Commands
Launch **Developer PowerShell for VS 2022** in Windows Tereminal
```
    . $home\miniconda3\shell\condabin\conda-hook.ps1
	conda activate base 
	conda create -n pyort python=3.9
	conda activate pyort
	pip install flatbuffers==2.0

	$env:JAVA_HOME="C:\Program Files\Microsoft\jdk-11.0.15.10-hotspot"
	# clone ONNXRuntime
	git clone -b rel-1.12.0 <GIT-REPO-URL> onnxruntime

	# clone onnxruntime-extensions
	git clone <URL> onnxruntime_extensions

    # build JAR package in this folder
    mkdir ortall.build
	cd ortall.build
	python ..\onnxruntime\tools\ci_build\build.py --config Release --cmake_generator "Visual Studio 7 2022" --build_java --build_dir .  --build_dir . --use_extensions --extensions_overridden_path "..\onnxruntime-extensions"
```
