FetchContent_Declare(
    dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.24.6.zip
    URL_HASH SHA1=59b1fb4e9909697c646e4f74e94871dacf49f0bf
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR  not_set
)

FetchContent_MakeAvailable(dlib)
