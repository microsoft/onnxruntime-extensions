if(onnxruntime_PREFER_SYSTEM_LIB)
  find_package(re2)
endif()

if(NOT TARGET re2::re2)
  include(ExternalProject)

  set(RE2_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/re2/src/re2)
  set(RE2_URL https://github.com/google/re2.git)
  set(RE2_TAG 2020-11-01)

  ExternalProject_Add(re2
        PREFIX re2
        GIT_REPOSITORY ${RE2_URL}
        GIT_TAG ${RE2_TAG}
        BUILD_IN_SOURCE 1
  )
  set(re2_dep re2)
  # set(re2_lib re2::re2)
  # set_target_properties(re2 PROPERTIES EXCLUDE_FROM_ALL ON)
else()
  set(re2_dep re2)
  # set(re2_lib re2::re2)
endif()

#if ( WIN32 )   
#  set(RE2_STATIC_LIBRARIES ${RE2_BUILD}/${CMAKE_BUILD_TYPE}/re2.lib) 
#else ()   
#  set(RE2_STATIC_LIBRARIES ${RE2_BUILD}/libre2.a) 
#end()

# add_library(re2)
# set_target_properties(re2 PROPERTIES FOLDER "${PROJECT_SOURCE_DIR}/cmake/externals/re2")
# set_target_properties(re2_lib PROPERTIES IMPORTED_LOCATION ${RE2_STATIC_LIBRARIES})
