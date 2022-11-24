#include "ocos.h"
#include "no_codecs/gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODECS
#include "codecs/imread.hpp"
#include "codecs/imdecode.hpp"
#include "codecs/super_resolution_preprocess.hpp"
#include "codecs/super_resolution_postprocess.hpp"
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
