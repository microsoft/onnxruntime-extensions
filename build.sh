OSNAME=$(uname -s)
mkdir -p out/$OSNAME
cd out/$OSNAME

cmake "$@" ../.. && cmake --build . --config RelWithDebInfo --parallel
build_error=$?
cd ../..
exit $build_error
