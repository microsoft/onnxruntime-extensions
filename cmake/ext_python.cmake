set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
  message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 or NumPy not found!")
endif()
if (WIN32)
  list(APPEND shared_TARGET_SRC "${PROJECT_SOURCE_DIR}/pyop/extensions_pydll.def")
endif()

file(GLOB TARGET_SRC_PYOPS "pyop/*.cc" "pyop/*.h" "shared/*.cc")
add_library(extensions_pydll SHARED ${TARGET_SRC_PYOPS} ${shared_TARGET_LIB_SRC})
standardize_output_folder(extensions_pydll)
list(APPEND OCOS_COMPILE_DEFINITIONS PYTHON_OP_SUPPORT)
target_compile_definitions(extensions_pydll PRIVATE ${OCOS_COMPILE_DEFINITIONS})

message(STATUS "Fetch pybind11")
include(pybind11)
target_include_directories(extensions_pydll PRIVATE
  ${pybind11_INCLUDE_DIRS}
  $<TARGET_PROPERTY:Python3::Module,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:ocos_operators,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(extensions_pydll PRIVATE
  $<TARGET_PROPERTY:Python3::Module,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(extensions_pydll PRIVATE Python3::Module ocos_operators)

if(NOT "${OCOS_EXTENTION_NAME}" STREQUAL "")
  if(NOT WIN32)
    set_target_properties(extensions_pydll PROPERTIES
      LIBRARY_OUTPUT_NAME ${OCOS_EXTENTION_NAME}
      PREFIX ""
      SUFFIX "")
  endif()
endif()
