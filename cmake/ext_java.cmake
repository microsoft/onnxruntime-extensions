include(FindJava)
find_package(Java REQUIRED)
include(UseJava)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_package(JNI REQUIRED)
endif()

set(JAVA_ROOT ${PROJECT_SOURCE_DIR}/java)
set(JAVA_OUTPUT_TEMP ${CMAKE_CURRENT_BINARY_DIR}/java-temp)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)

# use the gradle wrapper if it exists
if(EXISTS "${JAVA_ROOT}/gradlew")
    set(GRADLE_EXECUTABLE "${JAVA_ROOT}/gradlew")
else()
    # fall back to gradle on our PATH
    find_program(GRADLE_EXECUTABLE gradle)
    if(NOT GRADLE_EXECUTABLE)
        message(SEND_ERROR "Gradle installation not found")
    endif()
endif()
message(STATUS "Using gradle: ${GRADLE_EXECUTABLE}")

# Specify the Java source files
file(GLOB_RECURSE onnxruntime_extensions4j_gradle_files "${JAVA_ROOT}/*.gradle")
file(GLOB_RECURSE onnxruntime_extensions4j_src "${JAVA_ROOT}/src/main/java/ai/onnxruntime/extensions/*.java")
set(JAVA_OUTPUT_JAR ${JAVA_OUTPUT_TEMP}/build/libs/onnxruntime_extensions.jar)
# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated and the onnxruntime_extensions4j_jni target will be rebuilt
set(GRADLE_ARGS --console=plain clean jar -p ${JAVA_ROOT} -x test )
if(WIN32)
  set(GRADLE_ARGS ${GRADLE_ARGS} -Dorg.gradle.daemon=false)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
  # For Android build, we may run gradle multiple times in same build,
  # sometimes gradle JVM will run out of memory if we keep the daemon running
  # it is better to not keep a daemon running
  set(GRADLE_ARGS ${GRADLE_ARGS} --no-daemon)
endif()

file(MAKE_DIRECTORY ${JAVA_OUTPUT_TEMP})
add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR} COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_ARGS} WORKING_DIRECTORY ${JAVA_OUTPUT_TEMP} DEPENDS ${onnxruntime_extensions4j_gradle_files} ${onnxruntime_extensions4j_src})
add_custom_target(onnxruntime_extensions4j DEPENDS ${JAVA_OUTPUT_JAR})
set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime_extensions4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the native sources
file(GLOB onnxruntime_extensions4j_native_src
    "${JAVA_ROOT}/src/main/native/*.c"
    "${JAVA_ROOT}/src/main/native/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.h"
    )
# Build the JNI library
add_library(onnxruntime_extensions4j_jni SHARED ${onnxruntime_extensions4j_native_src})

# depend on java sources. if they change, the JNI should recompile
add_dependencies(onnxruntime_extensions4j_jni onnxruntime_extensions4j)
target_include_directories(onnxruntime_extensions4j_jni PRIVATE ortcustomops)
# the JNI headers are generated in the onnxruntime_extensions4j target
target_include_directories(onnxruntime_extensions4j_jni PRIVATE ${JAVA_ROOT}/build/headers ${JNI_INCLUDE_DIRS})
target_link_libraries(onnxruntime_extensions4j_jni PRIVATE ortcustomops)
standardize_output_folder(onnxruntime_extensions4j_jni)

set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  if (OCOS_ENABLE_SPM_TOKENIZER)
    target_link_libraries(onnxruntime_extensions4j_jni PUBLIC log)
  endif()
endif()
# Set platform and arch for packaging
# Checks the names set by MLAS on non-Windows platforms first
if(APPLE)
   get_target_property(ONNXRUNTIME4J_OSX_ARCH onnxruntime_extensions4j_jni OSX_ARCHITECTURES)
   list(LENGTH ONNXRUNTIME4J_OSX_ARCH ONNXRUNTIME4J_OSX_ARCH_LEN)
   if(ONNXRUNTIME4J_OSX_ARCH)
       if(ONNXRUNTIME4J_OSX_ARCH_LEN LESS_EQUAL 1)
               list(GET ONNXRUNTIME4J_OSX_ARCH 0 JNI_ARCH)
               message("Set Java ARCH TO macOS/iOS ${JNI_ARCH}")
       else()
               message(FATAL_ERROR "Java is currently not supported for macOS universal")
       endif()
   else()
       set(JNI_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
       message("Set Java ARCH TO macOS/iOS ${JNI_ARCH}")
   endif()
   if(JNI_ARCH STREQUAL "x86_64")
       set(JNI_ARCH x64)
   elseif(JNI_ARCH STREQUAL "arm64")
       set(JNI_ARCH aarch64)
   endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(JNI_ARCH ${ANDROID_ABI})
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(JNI_ARCH x64)
else()
  # Now mirror the checks used with MSVC
  if(MSVC)
    if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
      set(JNI_ARCH aarch64)
    elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
      set(JNI_ARCH x64)
    else()
      # if everything else failed then we're on a 32-bit arch and Java isn't supported
      message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
    endif()
  else()
    # if everything else failed then we're on a 32-bit arch and Java isn't supported
    message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
  endif()
endif()

if (WIN32)
  set(JAVA_PLAT "win")
elseif (APPLE)
  set(JAVA_PLAT "osx")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(JAVA_PLAT "linux")
else()
  # We don't do distribution for Android
  # Set for completeness
  set(JAVA_PLAT "android")
endif()

# Similar to Nuget schema
set(JAVA_OS_ARCH ${JAVA_PLAT}-${JNI_ARCH})

# expose native libraries to the gradle build process
set(JAVA_PACKAGE_DIR ai/onnxruntime/extensions/native/${JAVA_OS_ARCH})
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_NATIVE_JNI_DIR ${JAVA_OUTPUT_DIR}/native-jni)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
set(JAVA_PACKAGE_JNI_DIR ${JAVA_NATIVE_JNI_DIR}/${JAVA_PACKAGE_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_JNI_DIR})

# On Windows TARGET_LINKER_FILE_NAME is the .lib, TARGET_FILE_NAME is the .dll
if (WIN32)
  #Our static analysis plugin set /p:LinkCompiled=false
  if(NOT onnxruntime_extensions_ENABLE_STATIC_ANALYSIS)
    add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_extensions4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_FILE_NAME:onnxruntime_extensions4j_jni>)
  endif()
else()
  add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_extensions4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_extensions4j_jni>)
endif()

# run the build process (this copies the results back into CMAKE_CURRENT_BINARY_DIR)
set(GRADLE_ARGS --console=plain cmakeBuild -p ${JAVA_ROOT} -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
  set(GRADLE_ARGS ${GRADLE_ARGS} -Dorg.gradle.daemon=false)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
  # For Android build, we may run gradle multiple times in same build,
  # sometimes gradle JVM will run out of memory if we keep the daemon running
  # it is better to not keep a daemon running
  set(GRADLE_ARGS ${GRADLE_ARGS} --no-daemon)
endif()

message(STATUS "GRADLE_ARGS: ${GRADLE_ARGS}")
add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_ARGS} WORKING_DIRECTORY ${JAVA_OUTPUT_TEMP})

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ANDROID_PACKAGE_JNILIBS_DIR ${JAVA_OUTPUT_DIR}/android)
  set(ANDROID_PACKAGE_ABI_DIR ${ANDROID_PACKAGE_JNILIBS_DIR}/${ANDROID_ABI})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_JNILIBS_DIR})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_ABI_DIR})

  # Copy onnxruntime_extensions4j_jni.so for building Android AAR package
  add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime_extensions4j_jni> ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_extensions4j_jni>)
  # Generate the Android AAR package
  add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} -b build-android.gradle -c settings-android.gradle build -DjniLibsDir=${ANDROID_PACKAGE_JNILIBS_DIR} -DbuildDir=${ANDROID_PACKAGE_OUTPUT_DIR} -DminSdkVer=${ANDROID_MIN_SDK} WORKING_DIRECTORY ${JAVA_ROOT})

  if (onnxruntime_extensions_BUILD_UNIT_TESTS)
    set(ANDROID_TEST_PACKAGE_ROOT ${JAVA_ROOT}/src/test/android)
    set(ANDROID_TEST_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/androidtest/android)
    #copy the androidtest project into cmake binary directory
    file(MAKE_DIRECTORY ${JAVA_OUTPUT_DIR}/androidtest)
    file(COPY ${ANDROID_TEST_PACKAGE_ROOT} DESTINATION ${JAVA_OUTPUT_DIR}/androidtest)
    set(ANDROID_TEST_PACKAGE_LIB_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/libs)
    file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_LIB_DIR})
    # Copy the built Android AAR package to libs folder of our test app
    add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ANDROID_PACKAGE_OUTPUT_DIR}/outputs/aar/onnxruntime_extensions-debug.aar ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime_extensions-mobile.aar)
    # Build Android test apk for java package
    add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} clean WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
    add_custom_command(TARGET onnxruntime_extensions4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} assembleDebug assembleDebugAndroidTest -DminSdkVer=${ANDROID_MIN_SDK} WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
  endif()
endif()
