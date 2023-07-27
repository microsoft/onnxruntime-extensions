Manual libraries that need to be prebuilt for the Azure operators on Android and iOS.
There is no simple cmake setup that works, so we prebuild as a one-off.

Requirements:
- pkg-config 
- Android
  - Android SDK installed with NDK 25 or later 
    - TODO: Instructions on installing android-sdk and ndk
  - export ANDROID_NDK_ROOT with the value set to the NDK path

Android:
  From this directory run `./build_curl_for_android.sh`
  An architecture can optionally be specified as the first argument to limit the build to that architecture.
  Otherwise all 4 architectures (arm, arm64, x86, x86_64) will be built. 
  e.g. if you just want to build locally for the emulator you can do `./build_curl_for_android.sh x86_64`