#include "ocos.h"

#include "dr_flac.h"
#define DR_MP3_IMPLEMENTATION 1
#define DR_MP3_FLOAT_OUTPUT 1
#include "dr_mp3.h"
#include "dr_wav.h"

#include "string_utils.h"

struct KernelAudioDecoder : public BaseKernel {
 public:
  KernelAudioDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const int64_t* p_data = ort_.GetTensorData<int64_t>(input);
    OrtTensorDimensions input_dim(ort_, input);
    if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
      ORTX_CXX_API_THROW("[AudioDecoder]: Expect input dimension [n] or [1,n].", ORT_INVALID_ARGUMENT);
    }

    drmp3 mp3_obj;
    std::list<std::vector<float>> lst_frames;
    int64_t total_size = 0;
    drmp3_init_memory(&mp3_obj, p_data, input_dim.Size(), nullptr);
    std::vector<float> buf;
    const size_t default_chunk_size = 4096;
    buf.resize(default_chunk_size * mp3_obj.channels);

    for (;;) {
      auto n_frames = drmp3_read_pcm_frames_f32(&mp3_obj, default_chunk_size, buf.data());
      if (n_frames <= 0) {
        break;
      }
      auto n_samples = n_frames * mp3_obj.channels;
      total_size += n_samples;
      buf.resize(n_samples);
      lst_frames.emplace_back(buf);
    }

    std::vector<int64_t> dim_out = {1, total_size};
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dim_out.data(), dim_out.size());
    float* p_output = ort_.GetTensorMutableData<float>(v);
    int64_t offset = 0;
    for (auto _b : lst_frames) {
      memcpy(p_output + offset, _b.data(), _b.size() * sizeof(float));
      offset += _b.size() * sizeof(float);
    }
  }

 private:
};

struct CustomOpAudioDecoder : OrtW::CustomOpBase<CustomOpAudioDecoder, KernelAudioDecoder> {
  const char* GetName() const {
    return "AudioDecoder";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};
