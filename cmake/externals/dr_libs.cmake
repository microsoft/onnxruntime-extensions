FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        dd762b861ecadf5ddd5fb03e9ca1db6707b54fbb
)

FetchContent_GetProperties(dr_libs)
if(NOT dr_libs_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(dr_libs)
endif()
