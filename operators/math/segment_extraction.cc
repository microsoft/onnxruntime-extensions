// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "segment_extraction.hpp"
#include "dlib/stft_norm.hpp"

OrtStatusPtr segment_extraction(const ortc::Tensor<int64_t>& input, ortc::Tensor<int64_t>& output0,
                                ortc::Tensor<int64_t>& output1) {
  auto& input_dim = input.Shape();
  if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
    return OrtW::CreateStatus("[SegmentExtraction]: Expect input dimension [n] or [1,n].", ORT_INVALID_GRAPH);
  }
  const int64_t* p_data = input.Data();
  std::vector<std::int64_t> segment_value;
  std::vector<std::int64_t> segment_position;
  for (std::int64_t i = 0; i < input.NumberOfElement(); i++) {
    if (!p_data[i]) {
      continue;
    }

    // push start position and value
    if (i == 0 || p_data[i - 1] != p_data[i]) {
      segment_value.push_back(p_data[i]);
      segment_position.push_back(i);
    }

    // push end position
    if (i == (input.NumberOfElement() - 1) || p_data[i + 1] != p_data[i]) {
      segment_position.push_back(i + 1);
    }
  }

  std::vector<int64_t> segment_value_dim({static_cast<int64_t>(segment_value.size())});
  std::vector<int64_t> segment_position_dim({static_cast<int64_t>(segment_value.size()), 2});

  int64_t* out0_data = output0.Allocate(segment_position_dim);
  std::copy(segment_position.begin(), segment_position.end(), out0_data);

  int64_t* out1_data = output1.Allocate(segment_value_dim);
  std::copy(segment_value.begin(), segment_value.end(), out1_data);
  return nullptr;
}

class CppAllocator : public ortc::IAllocator {
 public:
  void* Alloc(size_t size) override { return std::make_unique<char[]>(size).release(); }

  void Free(void* p) override {
    std::unique_ptr<char[]> ptr(static_cast<char*>(p));
    ptr.reset();
  }

  static CppAllocator& Instance() {
    static CppAllocator allocator;
    return allocator;
  }
};

  static std::vector<float> hann_window(int N) {
    std::vector<float> window(N);

    for (int n = 0; n < N; ++n) {
      // Original formula introduces more rounding errors than the current implementation
      // window[n] = static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * n / (N - 1))));
      double n_sin = std::sin(M_PI * n / N);
      window[n] = static_cast<float>(n_sin * n_sin);
    }

    return window;
  }

OrtStatusPtr segment_extraction2(const ortc::Tensor<float>& audio,
 const ortc::Tensor<int64_t>& sr_tensor,
    const ortc::Tensor<int64_t>& frame_ms_tensor,
    const ortc::Tensor<int64_t>& hop_ms_tensor,
    const ortc::Tensor<float>& energy_threshold_db_tensor,
                                 ortc::Tensor<int64_t>& output0) {
    // ✅ 1. Validate audio shape
    const auto& audio_shape = audio.Shape();
    if (audio_shape.size() != 2 || audio_shape[0] != 1) {
        return OrtW::CreateStatus("Audio must have shape [1, num_samples]", ORT_INVALID_ARGUMENT);
    }

    int sr = static_cast<int>(sr_tensor.Data()[0]);
int frame_ms = static_cast<int>(frame_ms_tensor.Data()[0]);
int hop_ms = static_cast<int>(hop_ms_tensor.Data()[0]);
float energy_threshold_db = energy_threshold_db_tensor.Data()[0];

    const int64_t n_fft = (frame_ms * sr) / 1000;
    const int64_t hop_length = (hop_ms * sr) / 1000;
    const int64_t frame_length = n_fft;

    const float* pcm_data = audio.Data();
    const int64_t num_samples = audio_shape[1];

    std::cout << "[DEBUG] Received audio with " << num_samples << " samples.\n";
    std::cout << "[DEBUG] sr=" << sr 
              << " frame_ms=" << frame_ms 
              << " hop_ms=" << hop_ms 
              << " threshold=" << energy_threshold_db << " dB\n";

    // ✅ 2. Hann window
    std::vector<float> window2 = hann_window(static_cast<int>(n_fft));

    // ✅ 3. PCM tensor
    ortc::Tensor<float> pcm_tensor({1, num_samples}, const_cast<float*>(pcm_data));

    // ✅ 4. STFT
    ortc::Tensor<float> stft_out(&CppAllocator::Instance());
    StftNormal stft;
    auto status = stft.Compute(pcm_tensor, n_fft, hop_length,
                               {window2.data(), window2.size()},
                               frame_length, stft_out);
    if (!status.IsOk()) {
        return OrtW::CreateStatus("STFT Compute failed", ORT_FAIL);
    }

    const auto& stft_shape = stft_out.Shape();
    const int64_t n_freq = stft_shape[1];
    const int64_t n_frames = stft_shape[2];
    const float* spec_ptr = stft_out.Data();

    // ================================
    // 4. Frame energy calculation
    // ================================
    std::vector<float> energy(n_frames, 0.0f);
    for (int64_t t = 0; t < n_frames; ++t) {
        float sum = 0.0f;
        // spec_ptr is [freq][time], contiguous by freq first
        for (int64_t f = 0; f < n_freq; ++f) {
            float val = spec_ptr[f * n_frames + t];
            sum += val;
        }
        energy[t] = sum;
    }

    // ================================
    // 5. Convert to dB
    // ================================
    std::vector<float> energy_db(n_frames);
    for (int64_t t = 0; t < n_frames; ++t) {
        energy_db[t] = 10.0f * std::log10(energy[t] + 1e-10f);
    }

    // ================================
    // 6. Adaptive threshold
    // ================================
    float max_val = energy_db[0];
    std::vector<float> tmp = energy_db;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    float median_val = tmp[tmp.size() / 2];
    for (float v : energy_db) max_val = std::max(max_val, v);
    float threshold = std::max(max_val + energy_threshold_db, median_val);

    // ================================
    // 7. Mask
    // ================================
    std::vector<bool> mask(n_frames);
    for (int64_t t = 0; t < n_frames; ++t) {
        mask[t] = energy_db[t] > threshold;
    }

    // ================================
    // 8. Contiguous segments
    // ================================
    std::vector<std::pair<float, float>> segments;
    bool active = false;
    int64_t start_idx = 0;

    for (int64_t t = 0; t < n_frames; ++t) {
        if (!active && mask[t]) {
            active = true;
            start_idx = t;
        } else if (active && !mask[t]) {
            active = false;
            float start_s = static_cast<float>(start_idx * hop_length) / sr;
            float end_s   = static_cast<float>(t * hop_length) / sr;
            segments.emplace_back(start_s, end_s);
        }
    }
    if (active) {
        float start_s = static_cast<float>(start_idx * hop_length) / sr;
        float end_s   = static_cast<float>(n_frames * hop_length) / sr;
        segments.emplace_back(start_s, end_s);
    }

    // ================================
    // 9. Write segments to output
    // ================================
    const int64_t num_segments = static_cast<int64_t>(segments.size());
    std::vector<int64_t> out_shape = { num_segments, 2 };
    int64_t* out_data = output0.Allocate(out_shape);

    for (int64_t i = 0; i < num_segments; ++i) {
        out_data[i * 2 + 0] = static_cast<int64_t>(segments[i].first * 1000.0f);  // start ms
        out_data[i * 2 + 1] = static_cast<int64_t>(segments[i].second * 1000.0f); // end ms
    }

    std::cout << "[DEBUG] Segments detected: " << num_segments << std::endl;

    return nullptr;
}