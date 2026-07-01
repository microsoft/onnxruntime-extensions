# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(Python_FIND_REGISTRY NEVER)
# if we don't set this to NEVER (or possibly LAST) the builds of the wheel for different python versions will fail
# as it will find the system python version first and not the correct python version for the wheel.
set(Python_FIND_FRAMEWORK NEVER)
if(CMAKE_VERSION VERSION_LESS 3.18)
  set(_PYTHON_DEV_COMPONENT Development)
else()
  set(_PYTHON_DEV_COMPONENT Development.Module)
endif()
find_package(Python COMPONENTS Interpreter ${_PYTHON_DEV_COMPONENT} OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT Python_FOUND)
  message(FATAL_ERROR "Python not found!")
endif()

file(GLOB TARGET_SRC_PYOPS "pyop/pyfunc.cc" "pyop/*.h" "shared/*.cc")
if (OCOS_ENABLE_C_API)
  list(APPEND TARGET_SRC_PYOPS "pyop/py_c_api.cc")
endif()
if (WIN32)
  list(APPEND TARGET_SRC_PYOPS "pyop/extensions_pydll.def")
endif()

message(STATUS "Fetch nanobind")
include(nanobind)

nanobind_add_module(extensions_pydll NB_STATIC STABLE_ABI ${TARGET_SRC_PYOPS} ${shared_TARGET_LIB_SRC})
standardize_output_folder(extensions_pydll)
list(APPEND OCOS_COMPILE_DEFINITIONS PYTHON_OP_SUPPORT)
target_compile_definitions(extensions_pydll PRIVATE ${OCOS_COMPILE_DEFINITIONS})
target_include_directories(extensions_pydll PRIVATE
  $<TARGET_PROPERTY:ocos_operators,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(extensions_pydll PRIVATE ocos_operators)

if(OCOS_PYTHON_MODULE_PATH)
  add_custom_command(TARGET extensions_pydll POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:extensions_pydll> ${OCOS_PYTHON_MODULE_PATH}
    COMMENT "Copying $<TARGET_FILE:extensions_pydll> to ${OCOS_PYTHON_MODULE_PATH}")
endif()
