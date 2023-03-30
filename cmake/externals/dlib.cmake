FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    # there is non an official tag which supports STFT,
    # choose a relatively stable commit id for that.
    GIT_TAG        a12824d42584e292ecb3bad05c4b32c2015a7b89
)

FetchContent_GetProperties(dlib)
if(NOT dlib_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(dlib)
endif()
