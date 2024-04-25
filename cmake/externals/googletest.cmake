FetchContent_Declare(
  googletest
  URL         https://github.com/google/googletest/archive/9406a60c7839052e4944ea4dbc8344762a89f9bd.zip
  URL_HASH    SHA1=06096d3900c356e468ba060a609642c635131106
)

FetchContent_MakeAvailable(googletest)
set_target_properties(gmock PROPERTIES FOLDER "externals/gtest")
set_target_properties(gmock_main PROPERTIES FOLDER "externals/gtest")
set_target_properties(gtest PROPERTIES FOLDER "externals/gtest")
set_target_properties(gtest_main PROPERTIES FOLDER "externals/gtest")
