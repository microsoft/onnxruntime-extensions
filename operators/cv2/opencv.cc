#include "ocos.h"
#include "imgproc/gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODECS
#include "imgcodecs/imread.hpp"
#include "imgcodecs/imdecode.hpp"
#endif // ENABLE_OPENCV_CODECS


FxLoadCustomOpFactory LoadCustomOpClasses_CV2 =
    LoadCustomOpClasses<CustomOpClassBegin
                        , CustomOpGaussianBlur
#ifdef ENABLE_OPENCV_CODECS
                        , CustomOpImageReader
                        , CustomOpImageDecoder
#endif // ENABLE_OPENCV_CODECS
    >;
