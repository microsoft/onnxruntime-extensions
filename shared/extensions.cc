// this is a stub C++ file for DLL/dlib/so generation
#include "onnxruntime_extensions.h"

// need a reference to a function from the static library for ld in Linux
auto exported_func_1 = &RegisterCustomOps;
