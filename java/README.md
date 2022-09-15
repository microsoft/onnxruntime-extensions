# ONNXRuntime-Extensions Java/Android API and Package

This java and Android API and packaging principles were inspired by the https://github.com/microsoft/onnxruntime/tree/main/java, and openly share credits for API with the contributors in onnxruntime repo.

<br />

## Building

<br />

### Tools required
1. install visual studio 2022 (with cmake, git, desktop C++)
2. OpenJDK: https://docs.microsoft.com/en-us/java/openjdk/download
		(OpenJDK 11.0.15 LTS)
3. Gradle: https://gradle.org/releases/
		(v6.9.2)


### Build command
./build.sh **-DOCOS_BUILD_JAVA=ON**

and find onnxruntime-extensions-0.5.0.jar at out/$OS/$CMake_BUILD_TYPE/java/build/libs

<br />

## Usage
There is a Java example project checked in tutorial folder, [demo4j](../tutorials/demo4j) which provide a showcase how extensions package works with ONNXRuntime's Java API
