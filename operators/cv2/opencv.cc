#include "ocos.h"
#include "gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODEC
#include "imread.hpp"
#include "imdecode.hpp"
#include "super_resolution_preprocess.hpp"
#include "super_resolution_postprocess.hpp"
#endif // ENABLE_OPENCV_CODEC


FxLoadCustomOpFactory LoadCustomOpClasses_CV2 =
    LoadCustomOpClasses<CustomOpClassBegin
                        , CustomOpGaussianBlur
#ifdef ENABLE_OPENCV_CODEC
                        , CustomOpImageReader
                        , CustomOpImageDecoder
                        , CustomOpSuperResolutionPreProcess
                        , CustomOpSuperResolutionPostProcess
#endif // ENABLE_OPENCV_CODEC
    >;
