FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.22
)

FetchContent_GetProperties(dlib)
if(NOT dlib_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(dlib)
endif()
