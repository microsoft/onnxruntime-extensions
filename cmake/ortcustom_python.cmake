# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
  message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
endif()

find_package(Python3 COMPONENTS Interpreter Development)

include(pybind11)

FIND_PACKAGE(NumPy)

if("${PYTHON_EXECUTABLE}" STREQUAL "")
  set(_python_exe "python")
else()
  set(_python_exe "${PYTHON_EXECUTABLE}")
endif()

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${_python_exe}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed? ${PYTHON_NOT_FOUND} ${PYTHON_INCLUDE_DIR} ${_python_exe}")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${_python_exe}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)


# ---[ Python + Numpy
set(ortcustom_pybind_srcs_pattern
    "${REPO_ROOT}/ocos/*.cc"
    "${REPO_ROOT}/ocos/*.h*"
    "${REPO_ROOT}/ocos/ortcustomops.def"
    "${REPO_ROOT}/ocos/kernels/*.cc"
    "${REPO_ROOT}/ocos/kernels/*.h*"
    "${REPO_ROOT}/ocos/pyfunc/*.cc"
    "${REPO_ROOT}/ocos/pyfunc/*.h*"
)

file(GLOB ortcustom_pybind_srcs CONFIGURE_DEPENDS
  ${ortcustom_pybind_srcs_pattern}
  )

add_library(_ortcustomops MODULE ${ortcustom_pybind_srcs})
target_compile_definitions(_ortcustomops PRIVATE PYTHON_OP_SUPPORT)
target_include_directories(_ortcustomops PUBLIC
   ./includes
   ./includes/onnxruntime
   ${RE2_INCLUDE_DIR}
)

if(MSVC)
  target_compile_options(_ortcustomops PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
endif()
if(HAS_CAST_FUNCTION_TYPE)
  target_compile_options(_ortcustomops PRIVATE "-Wno-cast-function-type")
endif()

if(ortcustom_PYBIND_EXPORT_OPSCHEMA)
  target_compile_definitions(_ortcustomops PRIVATE ortcustom_PYBIND_EXPORT_OPSCHEMA)
endif()

if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings
    target_compile_options(_ortcustomops PRIVATE "/wd4244")
endif()
target_include_directories(_ortcustomops PRIVATE 
    ${REPO_ROOT}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)
if(ortcustomops_USE_CUDA)
    target_include_directories(_ortcustomops PRIVATE ${ortcustomops_CUDNN_HOME}/include)
endif()

set(ortcustomops_libs ${pybind11_lib} re2::re2 Python3::Python) 
set(python_libs c:/Python372_x64/libs)
add_dependencies(_ortcustomops ${pybind11_dep} re2::re2)

if (MSVC)
  # if MSVC, pybind11 looks for release version of python lib (pybind11/detail/common.h undefs _DEBUG)
  target_link_libraries(_ortcustomops ${ortcustomops_libs}
          ${PYTHON_LIBRARY_RELEASE} ${ortcustomops_EXTERNAL_LIBRARIES})
elseif (APPLE)
  set_target_properties(_ortcustomops PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  target_link_libraries(_ortcustomops ${ortcustomops_libs} ${ortcustomops_EXTERNAL_LIBRARIES})
  set_target_properties(_ortcustomops PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  target_link_libraries(_ortcustomops PRIVATE ${ortcustomops_libs} ${ortcustomops_EXTERNAL_LIBRARIES})
  set_property(TARGET _ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\$ORIGIN")
endif()

if (MSVC)
  set_target_properties(_ortcustomops PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(_ortcustomops PROPERTIES SUFFIX ".so")
endif()

file(GLOB ortcustom_python_srcs CONFIGURE_DEPENDS
  "${REPO_ROOT}/ocos/pyfunc/*.py"
)

file(GLOB ortcustom_python_pkg CONFIGURE_DEPENDS
  "${REPO_ROOT}/setup.py"
  "${REPO_ROOT}/README.md"
  "${REPO_ROOT}/requirements.txt"
)

add_custom_command(
  TARGET _ortcustomops POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:_ortcustomops>/ortcustomops
  COMMAND ${CMAKE_COMMAND} -E copy ${ortcustom_python_srcs} $<TARGET_FILE_DIR:_ortcustomops>/ortcustomops/
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_ortcustomops> $<TARGET_FILE_DIR:_ortcustomops>/ortcustomops/
  COMMAND ${CMAKE_COMMAND} -E copy ${ortcustom_python_pkg} $<TARGET_FILE_DIR:_ortcustomops>/
)
