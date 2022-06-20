#include "ocos.h"
#include "gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODEC
#include "imread.hpp"
#endif // ENABLE_OPENCV_CODEC


FxLoadCustomOpFactory LoadCustomOpClasses_OpenCV =
    LoadCustomOpClasses<CustomOpClassBegin
                        , CustomOpGaussianBlur
#ifdef ENABLE_OPENCV_CODEC
                        , CustomOpImageReader
#endif // ENABLE_OPENCV_CODEC
    >;
