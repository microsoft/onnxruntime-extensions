mkdir -p out/Linux
cd out/Linux

cmake -A x64 %* ../..
cmake --build . --config RelWithDebInfo
cd ../..
