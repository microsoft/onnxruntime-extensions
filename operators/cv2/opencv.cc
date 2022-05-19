#include "ocos.h"
#include "gaussian_blur.hpp"
#include "imread.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_OpenCV =
    LoadCustomOpClasses<CustomOpClassBegin,
                        CustomOpGaussianBlur,
                        CustomOpImageReader
    >;
