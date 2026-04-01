FetchContent_Declare(dr_libs
    URL         https://github.com/mackron/dr_libs/archive/660795b2834aebb2217c9849d668b6e4bd4ef810.zip
    URL_HASH    SHA1=af554b21dcd1ab3c7c8d946638682a2cbccf3e16
    SOURCE_SUBDIR not_set
    PATCH_COMMAND  git apply --whitespace=fix --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/cmake/externals/dr_libs_aix.patch
)

FetchContent_MakeAvailable(dr_libs)
