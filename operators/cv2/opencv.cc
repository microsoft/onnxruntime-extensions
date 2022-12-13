#include "ocos.h"
#include "imgproc/gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODECS
#include "imgcodecs/imread.hpp"
#include "imgcodecs/imdecode.hpp"
#include "imgcodecs/super_resolution_preprocess.hpp"
#include "imgcodecs/super_resolution_postprocess.hpp"
#endif // ENABLE_OPENCV_CODECS


FxLoadCustomOpFactory LoadCustomOpClasses_CV2 =
    LoadCustomOpClasses<CustomOpClassBegin
                        , CustomOpGaussianBlur
#ifdef ENABLE_OPENCV_CODECS
                        , CustomOpImageReader
                        , CustomOpImageDecoder
                        , CustomOpSuperResolutionPreProcess
                        , CustomOpSuperResolutionPostProcess
#endif // ENABLE_OPENCV_CODECS
    >;
