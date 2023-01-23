FetchContent_Declare(
    GSL
    URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
    URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
    FIND_PACKAGE_ARGS 4.0 NAMES Microsoft.GSL
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(GSL)
get_target_property(GSL_INCLUDE_DIR Microsoft.GSL::GSL INTERFACE_INCLUDE_DIRECTORIES)
