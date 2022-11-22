#if defined(ENABLE_CV2)
#include "ocos.h"
#include "gaussian_blur.hpp"
#include "imread.hpp"
#include "imdecode.hpp"
#include "super_resolution_preprocess.hpp"
#include "super_resolution_postprocess.hpp"

FxLoadCustomOpFactory LoadCustomOpClasses_CV2 =
    LoadCustomOpClasses<CustomOpClassBegin,
                        CustomOpGaussianBlur,
                        CustomOpImageReader,
                        CustomOpImageDecoder,
                        CustomOpSuperResolutionPreProcess,
                        CustomOpSuperResolutionPostProcess>;

#endif
