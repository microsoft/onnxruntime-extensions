FetchContent_Declare(
  Blingfire
  GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
  GIT_TAG 0831265c1aca95ca02eca5bf1155e4251e545328
  EXCLUDE_FROM_ALL
  PATCH_COMMAND git checkout . && git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/externals/blingfire_cmake.patch)

FetchContent_MakeAvailable(Blingfire)
set_target_properties(bingfirtinydll_static PROPERTIES FOLDER
                                                       externals/bingfire)
set_target_properties(fsaClientTiny PROPERTIES FOLDER externals/bingfire)
