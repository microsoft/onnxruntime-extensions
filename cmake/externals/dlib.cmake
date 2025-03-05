FetchContent_Declare(
    dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.24.7.zip
    URL_HASH SHA1=6c63ea576e2b525751b0dead27c6c1139c5100ae
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR  not_set
)

FetchContent_MakeAvailable(dlib)
