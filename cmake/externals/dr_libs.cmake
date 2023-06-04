FetchContent_Declare(dr_libs
    URL         https://github.com/mackron/dr_libs/archive/dbbd08d81fd2b084c5ae931531871d0c5fd83b87.zip
    URL_HASH    SHA1=84a2a31ef890b6204223b12f71d6e701c0edcd92
)

FetchContent_GetProperties(dr_libs)
if(NOT dr_libs_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(dr_libs)
endif()
