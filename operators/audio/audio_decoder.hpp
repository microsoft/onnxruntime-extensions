#include "ocos.h"

#include "dr_flac.h"
#include "dr_mp3.h"
#include "dr_wav.h"

#include "string_utils.h"

struct KernelAudioDecoder : public BaseKernel {
 public:
  KernelAudioDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  struct LibFlac {
    drflac_int32* pPCMFrames; /* Interleaved. */
    drflac_uint64 pcmFrameCount;
    drflac_uint64 pcmFrameCap; /* The capacity of the pPCMFrames buffer in PCM frames. */
    drflac_uint32 channels;
    drflac_uint32 sampleRate;
    drflac_uint64 currentPCMFrame; /* The index of the PCM frame the decoder is currently sitting on. */
    double decodeTimeInSeconds;    /* The total amount of time it took to decode the file. This is used for profiling. */
    drflac_uint8* pFileData;
    size_t fileSizeInBytes;
    size_t fileReadPos;
  } libFlac;


  static drflac_uint64 libflac_read_pcm_frames_s32(LibFlac* pDecoder, drflac_uint64 framesToRead, drflac_int32* pBufferOut) {
    drflac_uint64 pcmFramesRemaining;

    if (pDecoder == NULL) {
      return 0;
    }

    pcmFramesRemaining = pDecoder->pcmFrameCount - pDecoder->currentPCMFrame;
    if (framesToRead > pcmFramesRemaining) {
      framesToRead = pcmFramesRemaining;
    }

    if (framesToRead == 0) {
      return 0;
    }

    memcpy(pBufferOut, pDecoder->pPCMFrames + (pDecoder->currentPCMFrame * pDecoder->channels), (size_t)(framesToRead * pDecoder->channels * sizeof(drflac_int32)));
    pDecoder->currentPCMFrame += framesToRead;

    return framesToRead;
  }


  void Compute(OrtKernelContext* context) {
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const int64_t* p_data = ort_.GetTensorData<int64_t>(input);
    OrtTensorDimensions input_dim(ort_, input);
    if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
        ORTX_CXX_API_THROW("[AudioDecoder]: Expect input dimension [n] or [1,n].", ORT_INVALID_ARGUMENT);
    }

    LibFlac* pLibFlac = nullptr;
    drflac* pFlac = nullptr;
    drflac_int32* pPCMFrames_libflac = nullptr;
    drflac_int32* pPCMFrames_drflac = nullptr;
    drflac_uint64 pcmFrameCount = 0;
    drflac_uint64 pcmFrameCount_libflac = 0;
    drflac_uint64 pcmFrameCount_drflac = 0;
    drflac_uint64 iPCMFrame = 0;

    /* To test decoding we just read a number of PCM frames from each decoder and compare. */
    pcmFrameCount_libflac = libflac_read_pcm_frames_s32(pLibFlac, pcmFrameCount, pPCMFrames_libflac);
    pcmFrameCount_drflac = drflac_read_pcm_frames_s32(pFlac, pcmFrameCount, pPCMFrames_drflac);
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
