mkdir -p out/Linux
cd out/Linux

cmake $* ../..
cmake --build . --config RelWithDebInfo
cd ../..
