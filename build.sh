mkdir -p out/Linux
cd out/Linux

cmake $* ../.. && cmake --build . --config RelWithDebInfo
build_error=$?
cd ../..
exit $build_error
