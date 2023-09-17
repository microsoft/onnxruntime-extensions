#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#include "dlib/stft_norm.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"
#include "bias_gelu_cuda.hpp"

FxLoadCustomOpFactory LoadCustomOpClasses_Math = []() -> CustomOpArray& {

  static OrtOpLoader op_loader(
/*
                               CustomCpuFuncV2("NegPos", neg_pos),
#ifdef ENABLE_DLIB
                               CustomCpuFuncV2("Inverse", inverse),
                               CustomCpuStructV2("StftNorm", StftNormal),
#endif
                               CustomCpuFuncV2("SegmentExtraction", segment_extraction),
                               CustomCpuFuncV2("SegmentSum", segment_sum),
*/
                               CustomCudaFuncV2("BiasGelu", bias_gelu_cuda, bias_gelu_cuda_shape_infer));

  /*
  std::shared_ptr<ortc::OrtLiteCustomOp> bias_gelu_op(ortc::CreateLiteCustomOp("BiasGelu", "CUDAExecutionProvider", bias_gelu_cuda));
  bias_gelu_op->SetShapeInferenceFn([](const Ort::Custom::TensorShapeVec& input_shapes, Ort::Custom::TensorShapeVec& output_shape) {
    bias_gelu_cuda_shape_infer(input_shapes, output_shape);
  });
  static OrtOpLoader op_loader(bias_gelu_op);
  */
  return op_loader.GetCustomOps();
};
