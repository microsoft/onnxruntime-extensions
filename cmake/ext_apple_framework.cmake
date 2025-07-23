# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin|iOS")
  message(FATAL_ERROR "Building an Apple framework can only be enabled for MacOS or iOS.")
endif()

# build a static framework

get_target_property(_ortcustomops_type ortcustomops TYPE)
if(NOT _ortcustomops_type STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR "ortcustomops must be a static library in order to build the Apple static framework.")
endif()

set(APPLE_FRAMEWORK_NAME "onnxruntime_extensions")
set(APPLE_FRAMEWORK_IDENTIFIER "com.microsoft.onnxruntime-extensions")
set(APPLE_FRAMEWORK_VERSION "${VERSION}")

# public header files
set(APPLE_FRAMEWORK_HEADERS
    "${PROJECT_SOURCE_DIR}/include/onnxruntime_extensions.h")

# generated framework directory
set(APPLE_FRAMEWORK_DIRECTORY
    "${PROJECT_BINARY_DIR}/static_framework/${APPLE_FRAMEWORK_NAME}.framework")

# generate Info.plist and framework_info.json
configure_file(${PROJECT_SOURCE_DIR}/cmake/apple/Info.plist.in ${PROJECT_BINARY_DIR}/Info.plist)
configure_file(${PROJECT_SOURCE_DIR}/cmake/apple/framework_info.json.in ${PROJECT_BINARY_DIR}/framework_info.json)

# gets transitive static library dependencies of targets in ${target_names}
# ${target_names} - root target names
# ${transitive_static_lib_deps_var} - output variable name
function(get_transitive_static_lib_deps target_names transitive_static_lib_deps_var)
  set(targets_to_process ${target_names})
  set(processed_targets)

  while(true)
    # exit loop when ${targets_to_process} is empty
    list(LENGTH targets_to_process targets_to_process_len)
    if(${targets_to_process_len} EQUAL 0)
      break()
    endif()

    list(GET targets_to_process 0 target_to_process)
    list(POP_FRONT targets_to_process)

    if(${target_to_process} IN_LIST processed_targets)
      continue()
    endif()

    message(DEBUG "processing target: ${target_to_process}")

    get_target_property(target_to_process_deps ${target_to_process} LINK_LIBRARIES)
    if(target_to_process_deps)
      foreach(dep IN LISTS target_to_process_deps)
        if(NOT TARGET ${dep})
          message(DEBUG "not a target")
          continue()
        endif()

        get_target_property(dep_type ${dep} TYPE)
        if(NOT "${dep_type}" STREQUAL "STATIC_LIBRARY")
          message(DEBUG "not a static library")
          continue()
        endif()

        message(DEBUG "adding dep to process: ${dep}")
        list(APPEND targets_to_process ${dep})
      endforeach()
    endif()

    list(APPEND processed_targets ${target_to_process})
  endwhile()

  # output ${processed_targets} without the initial values in ${target_names}
  list(LENGTH target_names target_names_len)
  list(SUBLIST processed_targets ${target_names_len} -1 transitive_static_lib_deps)
  set(${transitive_static_lib_deps_var} ${transitive_static_lib_deps} PARENT_SCOPE)
endfunction()

# get list of static library dependencies to combine
get_transitive_static_lib_deps(ortcustomops _ortcustomops_static_lib_dep_targets)
set(_ortcustomops_all_static_lib_dep_targets
    ortcustomops ${_ortcustomops_static_lib_dep_targets})
list(TRANSFORM _ortcustomops_all_static_lib_dep_targets
     REPLACE "(.+)" "$<TARGET_FILE:\\1>"
     OUTPUT_VARIABLE _ortcustomops_all_static_lib_deps)

add_custom_command(
  TARGET ortcustomops
  POST_BUILD
  COMMAND  # remove any existing framework files
    ${CMAKE_COMMAND} -E rm -rf -- ${APPLE_FRAMEWORK_DIRECTORY}
  COMMAND  # create output directories
    ${CMAKE_COMMAND} -E make_directory ${APPLE_FRAMEWORK_DIRECTORY}/Headers
  COMMAND  # copy header files
    ${CMAKE_COMMAND} -E copy_if_different ${APPLE_FRAMEWORK_HEADERS} ${APPLE_FRAMEWORK_DIRECTORY}/Headers
  COMMAND  # copy Info.plist
    ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_BINARY_DIR}/Info.plist ${APPLE_FRAMEWORK_DIRECTORY}/Info.plist
  COMMAND  # combine static libraries
    libtool -static -o ${APPLE_FRAMEWORK_DIRECTORY}/onnxruntime_extensions ${_ortcustomops_all_static_lib_deps}
)
