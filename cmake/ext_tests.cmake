if (OCOS_ENABLE_SELECTED_OPLIST)
  # currently the tests don't handle operator exclusion cleanly.
  message(FATAL_ERROR "Due to usage of OCOS_ENABLE_SELECTED_OPLIST excluding operators the tests are unable to be built and run")
endif()

enable_testing()

# Enable CTest
message(STATUS "Fetch CTest")
include(CTest)

set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test)
message(STATUS "Fetch googletest")
include(googletest)

if(IOS)
  find_package(XCTest REQUIRED)

  enable_language(OBJCXX)

  # enable ARC on Objective-C++ source files
  set_property(
      SOURCE
        "${TEST_SRC_DIR}/ios/dummy_testee_main.mm"
        "${TEST_SRC_DIR}/ios/xcgtest.mm"
      APPEND
      PROPERTY COMPILE_OPTIONS "-fobjc-arc")
endif()

function(add_test_target)
  set(optional_args)
  set(single_value_args
      # test target name
      TARGET)
  set(multi_value_args
      # test source files
      TEST_SOURCES
      # libraries to link to test target
      LIBRARIES
      # test data directories to copy to target directory
      TEST_DATA_DIRECTORIES)
  cmake_parse_arguments(ARG "${optional_args}" "${single_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT IOS)
    # add a test executable

    add_executable(${ARG_TARGET})

    standardize_output_folder(${ARG_TARGET})

    add_test(NAME ${ARG_TARGET}
             COMMAND ${ARG_TARGET})

    target_sources(${ARG_TARGET} PRIVATE
                   ${ARG_TEST_SOURCES}
                   "${TEST_SRC_DIR}/unittest_main/test_main.cc")

    target_link_libraries(${ARG_TARGET} PRIVATE
                          ${ARG_LIBRARIES}
                          gtest gmock)

    if(OCOS_USE_CUDA)
      target_link_directories(${ARG_TARGET} PRIVATE $ENV{CUDA_PATH}/lib64)
    endif()

    set(test_data_destination_root_directory ${onnxruntime_extensions_BINARY_DIR})

  else()
    # add a test that can run on iOS (simulator)
    # it requires a dummy testee target

    # create a dummy app to use as the testee target in xctest_add_bundle()
    set(dummy_testee_target ${ARG_TARGET}_dummy_testee)
    add_executable(${dummy_testee_target} "${TEST_SRC_DIR}/ios/dummy_testee_main.mm")
    set(_dummy_testee_version 1)
    set_target_properties(${dummy_testee_target} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME dummy_testee
        MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.unittest.dummy_testee
        MACOSX_BUNDLE_LONG_VERSION_STRING ${_dummy_testee_version}
        MACOSX_BUNDLE_BUNDLE_VERSION ${_dummy_testee_version}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${_dummy_testee_version}
        XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
        XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")
    target_link_libraries(${dummy_testee_target} PRIVATE "-framework UIKit")

    xctest_add_bundle(${ARG_TARGET} ${dummy_testee_target})

    xctest_add_test("${ARG_TARGET}_xctest" ${ARG_TARGET})

    target_sources(${ARG_TARGET} PRIVATE
                   ${ARG_TEST_SOURCES}
                   "${TEST_SRC_DIR}/ios/xcgtest.mm")

    target_link_libraries(${ARG_TARGET} PRIVATE
                          ${ARG_LIBRARIES}
                          gtest gmock)

    set(test_data_destination_root_directory $<TARGET_FILE_DIR:${dummy_testee_target}>)

  endif()

  # copy any test data directories to the target directory
  foreach(test_data_directory ${ARG_TEST_DATA_DIRECTORIES})
    cmake_path(GET test_data_directory FILENAME test_data_directory_filename)
    if(NOT test_data_directory_filename)
      # There's no filename if the path has a trailing directory separator.
      # In that case, try to get the filename of the parent path.
      cmake_path(GET test_data_directory PARENT_PATH test_data_directory_parent)
      cmake_path(GET test_data_directory_parent FILENAME test_data_directory_filename)
    endif()
    if(NOT test_data_directory_filename)
      message(FATAL_ERROR "Failed to get filename for test data directory: ${test_data_directory}")
    endif()

    set(destination_directory ${test_data_destination_root_directory}/${test_data_directory_filename})

    add_custom_command(
      TARGET ${ARG_TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${test_data_directory} ${destination_directory})
  endforeach()

endfunction(add_test_target)

# -- static test --
file(GLOB static_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")

add_test_target(TARGET ocos_test
                TEST_SOURCES ${static_TEST_SRC}
                LIBRARIES ortcustomops ${ocos_libraries})

# -- shared test (needs onnxruntime) --
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")

if("${ONNXRUNTIME}" STREQUAL "ONNXRUNTIME-NOTFOUND")
  message(WARNING "The prebuilt onnxruntime library was not found, extensions_test will be skipped.")
else()
  block()
    if(NOT IOS)
      set(use_extensions_shared_library 1)
    endif()

    set(extensions_target $<IF:$<BOOL:${use_extensions_shared_library}>,extensions_shared,ortcustomops>)

    file(GLOB shared_TEST_SRC
        "${TEST_SRC_DIR}/shared_test/*.cc"
        "${TEST_SRC_DIR}/shared_test/*.hpp")

    set(extensions_test_libraries ${extensions_target} ${ONNXRUNTIME})

    if(use_extensions_shared_library)
      list(APPEND extensions_test_libraries ${ocos_libraries})
    endif()

    # needs to link with stdc++fs in Linux
    if(LINUX)
      list(APPEND extensions_test_libraries stdc++fs -pthread)
    endif()

    add_test_target(TARGET extensions_test
                    TEST_SOURCES ${shared_TEST_SRC}
                    LIBRARIES ${extensions_test_libraries}
                    TEST_DATA_DIRECTORIES ${TEST_SRC_DIR}/data)

    target_include_directories(extensions_test PRIVATE ${spm_INCLUDE_DIRS})

    target_compile_definitions(extensions_test PUBLIC ${OCOS_COMPILE_DEFINITIONS})
    if(use_extensions_shared_library)
      target_compile_definitions(extensions_test PUBLIC ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)
    endif()

    # FUTURE: This is required to use the ORT C++ API with delayed init which must be done conditionally using
    # ifdef OCOS_BUILD_SHARED in RegisterCustomOps and where onnxruntime_cxx_api.h is included .
    # ---
    # We have to remove the OCOS_BUILD_SHARED when building the test code. It is used to delay population of the
    # ORT api pointer until RegisterCustomOps is called, but the test code needs to create an ORT env which requires
    # the pointer to exist.
    # set(test_compile_definitions ${OCOS_COMPILE_DEFINITIONS})
    # remove(test_compile_definitions "OCOS_SHARED_LIBRARY")
    # target_compile_definitions(extensions_test PUBLIC ${test_compile_definitions})

    # Copy onnxruntime DLL files into the same directory as the test binary.
    if(WIN32)
      file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
      file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
      add_custom_command(
        TARGET extensions_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:extensions_test>)
    endif()

    # Copy onnxruntime shared library to known location for easy access, e.g., for adb push to emulator or device.
    if(ANDROID)
      add_custom_command(
        TARGET extensions_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME} ${CMAKE_BINARY_DIR}/lib
      )
    endif()
  endblock()
endif()
