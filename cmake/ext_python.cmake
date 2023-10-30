block(PROPAGATE Python3_FOUND)
  set(Python3_FIND_REGISTRY NEVER)
  # if we don't set this to NEVER (or possibly LAST) the builds of the wheel for different python versions will fail
  # as it will find the system python version first and not the correct python version for the wheel.
  set(Python3_FIND_FRAMEWORK NEVER)
  find_package(Python3 COMPONENTS Interpreter Development.Module NumPy)
endblock()

if (NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 or NumPy not found!")
endif()

file(GLOB TARGET_SRC_PYOPS "pyop/*.cc" "pyop/*.h" "shared/*.cc")
if (WIN32)
  list(APPEND TARGET_SRC_PYOPS "pyop/extensions_pydll.def")
endif()

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

if(OCOS_PYTHON_MODULE_PATH)
  get_filename_component(OCOS_PYTHON_MODULE_NAME ${OCOS_PYTHON_MODULE_PATH} NAME)
  if(NOT WIN32)
    set_target_properties(extensions_pydll PROPERTIES
      LIBRARY_OUTPUT_NAME ${OCOS_PYTHON_MODULE_NAME}
      PREFIX ""
      SUFFIX "")
  endif()

  add_custom_command(TARGET extensions_pydll POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:extensions_pydll> ${OCOS_PYTHON_MODULE_PATH}
    COMMENT "Copying $<TARGET_FILE:extensions_pydll> to ${OCOS_PYTHON_MODULE_PATH}")
endif()
