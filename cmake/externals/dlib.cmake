fetchcontent_declare(
    dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.24.7.zip
    URL_HASH SHA1=6c63ea576e2b525751b0dead27c6c1139c5100ae
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR  not_set
    PATCH_COMMAND git apply --whitespace=fix --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/cmake/externals/dlib_aix.patch
)

fetchcontent_makeavailable(dlib)
