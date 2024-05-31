if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    FetchContent_Declare(
        GSL
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
        URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
        FIND_PACKAGE_ARGS 4.0 NAMES Microsoft.GSL
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
else()
    FetchContent_Declare(
        GSL
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
        URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
    )
endif()

FetchContent_GetProperties(GSL)
string(TOLOWER "GSL" lcName)
if(NOT ${lcName}_POPULATED)
  FetchContent_Populate(GSL)
#  add_subdirectory(${GSL_SOURCE_DIR} ${GSL_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(GSL_INCLUDE_DIR ${gsl_SOURCE_DIR}/include)

#get_target_property(GSL_INCLUDE_DIR Microsoft.GSL::GSL INTERFACE_INCLUDE_DIRECTORIES)
