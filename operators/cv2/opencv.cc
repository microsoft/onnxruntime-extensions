#include "ocos.h"
#include "imgproc/gaussian_blur.hpp"
#ifdef ENABLE_OPENCV_CODECS
#include "imgcodecs/imread.hpp"
#include "imgcodecs/imdecode.hpp"
#endif  // ENABLE_OPENCV_CODECS

const std::vector<const OrtCustomOp*>& Cv2Loader() {
  static OrtOpLoader op_loader(CustomCpuFunc("GaussianBlur", gaussian_blur)
#ifdef ENABLE_OPENCV_CODECS
                                   ,
                               CustomCpuFunc("ImageDecoder", image_decoder),
                               CustomCpuFunc("ImageReader", image_reader)
#endif  // ENABLE_OPENCV_CODECS
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_CV2 = Cv2Loader;
