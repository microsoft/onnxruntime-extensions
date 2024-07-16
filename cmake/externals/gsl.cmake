if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    FetchContent_Declare(
        GSL
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
        URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
        FIND_PACKAGE_ARGS 4.0 NAMES Microsoft.GSL
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR not_set
    )
else()
    FetchContent_Declare(
        GSL
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
        URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
    )
endif()

FetchContent_MakeAvailable(GSL)
set(GSL_INCLUDE_DIR ${gsl_SOURCE_DIR}/include)
