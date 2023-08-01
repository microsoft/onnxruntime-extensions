# Mobile Azure EP pre-build

Manual libraries that need to be prebuilt for the Azure operators on Android and iOS.
There is no simple cmake setup that works, so we prebuild as a one-off.

## Requirements:
- pkg-config 
- Android
  - Android SDK installed with NDK 25 or later 
    - You can install a package but that means you have to use `sudo` for all updates like installing an NDK
      - https://stackoverflow.com/questions/34556884/how-to-install-android-sdk-on-ubuntu
      - you still need to manually add the cmdline-tools to that package as well
      - probably easier to create a per-user install using command line tools
    - Using command line tools
      - Download the command line tools from https://developer.android.com/studio
        - Download the 'Command line tools only' and unzip
      - `mkdir ~/Android`
      - `unzip commandlinetools-linux-9477386_latest.zip`
      - `mkdir -p ~/Android/cmdline-tools/latest`
      - `mv cmdline-tools/* ~/Android/cmdline-tools/latest`
      - `export ANDROID_HOME=~/Android`
      - Add these to PATH
        - ~/Android/cmdline-tools/latest/bin
        - ~/Android/platform-tools/bin
      - `sdkmanager --list` to make sure the setup works
      - Install platform-tools and latest NDK
        - `sdkmanager --install platform-tools`
        - e.g. `sdkmanager --install ndk;25.2.9519653`        

      That should be enough to build. 
      e.g. `./build_lib.sh --android --android_api=24 --android_home=/home/me/Android --android_abi=x86_64 --android_ndk_path=/home/me/Android/ndk/25.2.9519653 --enable_cxx_tests`
      
      See Android documentation for installing a system image with `sdkmanager` and 
      creating an emulator with `avdmanager`.
- iOS
  - TBD
  
## Android build
  Export ANDROID_NDK_ROOT with the value set to the NDK path as this is used by the build script
  - e.g. export ANDROID_NDK_ROOT=~/Android/ndk/25.2.9519653
  From this directory run `./build_curl_for_android.sh`
  An architecture can optionally be specified as the first argument to limit the build to that architecture.
  Otherwise all 4 architectures (arm, arm64, x86, x86_64) will be built. 
  e.g. if you just want to build locally for the emulator you can do `./build_curl_for_android.sh x86_64`

## Android testing
  Build with `--enable_cxx_tests`. 
  This should result in the 'bin' directory of the build output having the two test executables.
  Create/start Android emulator
  Use `adb push` to copy bin, lib and data directories from the build output to the /data/local/tmp directory 
    - `adb push build/Android/bin /data/local/tmp`
      - repeat for 'lib' and 'data'
  - copy the onnxruntime shared library to the lib dir (adjust version number as needed)
    - adjust architecture as needed (most likely x86_64 for emulator and arm)
    - `adb push build/Android/Debug/_deps/onnxruntime-src/jni/x86_64/libonnxruntime.so /data/local/tmp/lib`
  - Connect to emulator
    - `adb shell`
    - `cd /data/local/tmp`
    - Add path to .so
      - export LD_LIBRARY_PATH=/data/local/tmp/lib:$LD_LIBRARY_PATH
    - Make tests executable
      - `chmod +x bin/ocos_test`
      - `chmod +x bin/extensions_test`
    - Run tests from `tmp` dir so paths to `data` are as expected
      - ./bin/ocos_test
      - ./bin/extensions_test
