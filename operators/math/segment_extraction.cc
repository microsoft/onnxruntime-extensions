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

OrtStatusPtr segment_extraction2(const ortc::Tensor<float>& audio, const ortc::Tensor<int64_t>& sr_tensor,
                                 const ortc::Tensor<int64_t>& frame_ms_tensor,
                                 const ortc::Tensor<int64_t>& hop_ms_tensor,
                                 const ortc::Tensor<float>& energy_threshold_db_tensor,
                                 ortc::Tensor<int64_t>& output0) {
  const auto& audio_shape = audio.Shape();
  if (audio_shape.size() != 2 || audio_shape[0] != 1) {
    return OrtW::CreateStatus("Audio must have shape [1, num_samples]", ORT_INVALID_ARGUMENT);
  }

  const int sr = static_cast<int>(sr_tensor.Data()[0]);
  const int frame_ms = static_cast<int>(frame_ms_tensor.Data()[0]);
  const int hop_ms = static_cast<int>(hop_ms_tensor.Data()[0]);
  const float energy_threshold_db = energy_threshold_db_tensor.Data()[0];

  const int64_t n_fft = (frame_ms * sr) / 1000;
  const int64_t hop_length = (hop_ms * sr) / 1000;

  const float* pcm_data = audio.Data();
  const int64_t num_samples = audio_shape[1];

  std::cout << "[DEBUG] Received audio with " << num_samples << " samples.\n";
  std::cout << "[DEBUG] sr=" << sr << " frame_ms=" << frame_ms << " hop_ms=" << hop_ms
            << " threshold=" << energy_threshold_db << " dB\n";

  std::vector<float> hann = hann_window(static_cast<int>(n_fft));

  ortc::Tensor<float> stft_out(&CppAllocator::Instance());
  StftNormal stft;
  auto status = stft.Compute(audio, n_fft, hop_length, {hann.data(), hann.size()}, n_fft, stft_out);
  if (!status.IsOk()) {
    return OrtW::CreateStatus("STFT Compute failed", ORT_FAIL);
  }

  const auto& stft_shape = stft_out.Shape();
  const int64_t n_freq = stft_shape[1];
  const int64_t n_frames = stft_shape[2];
  const float* spec_ptr = stft_out.Data();

  std::cout << "Freq " << n_freq << std::endl;
  std::cout << "Frames " << n_frames << std::endl;

  std::vector<float> energy(n_frames, 0.0f);
  for (int64_t t = 0; t < n_frames; ++t) {
    float sum = 0.0f;
    for (int64_t f = 0; f < n_freq; ++f) {
      float val = spec_ptr[f * n_frames + t];
      sum += val;
    }
    energy[t] = sum;
  }

  std::vector<float> energy_db(n_frames);
  for (int64_t t = 0; t < n_frames; ++t) {
    energy_db[t] = 10.0f * std::log10(energy[t] + 1e-10f);
  }

  float max_val = energy_db[0];
  std::vector<float> tmp = energy_db;
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  float median_val = tmp[tmp.size() / 2];
  for (float v : energy_db) max_val = std::max(max_val, v);
  float threshold = std::max(max_val + energy_threshold_db, median_val);

  std::vector<bool> mask(n_frames);
  for (int64_t t = 0; t < n_frames; ++t) {
    mask[t] = energy_db[t] > threshold;
  }

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
      float end_s = static_cast<float>(t * hop_length) / sr;
      segments.emplace_back(start_s, end_s);
    }
  }
  if (active) {
    float start_s = static_cast<float>(start_idx * hop_length) / sr;
    float end_s = static_cast<float>(n_frames * hop_length) / sr;
    segments.emplace_back(start_s, end_s);
  }

  const int64_t num_segments = static_cast<int64_t>(segments.size());
  std::vector<int64_t> out_shape = {num_segments, 2};
  int64_t* out_data = output0.Allocate(out_shape);

  for (int64_t i = 0; i < num_segments; ++i) {
    out_data[i * 2 + 0] = static_cast<int64_t>(segments[i].first * 1000.0f);   // start ms
    out_data[i * 2 + 1] = static_cast<int64_t>(segments[i].second * 1000.0f);  // end ms
  }

  std::cout << "[DEBUG] Segments detected: " << num_segments << std::endl;

  return nullptr;
}

OrtStatusPtr merge_and_filter_segments(const Ort::Custom::Tensor<int64_t>& segments_tensor,
                                       const Ort::Custom::Tensor<int64_t>& merge_gap_in_milliseconds_tensor,
                                       Ort::Custom::Tensor<int64_t>& output0) {
  const int64_t* seg_data = segments_tensor.Data();
  const auto& seg_shape = segments_tensor.Shape();

  if (seg_shape.size() != 2 || seg_shape[1] != 2) {
    return OrtW::CreateStatus("segments must have shape [N, 2]", ORT_INVALID_ARGUMENT);
  }

  const int64_t num_segments = seg_shape[0];

  if (num_segments == 0) {
    output0.Allocate({0, 2});
    return nullptr;
  }

  const int64_t merge_gap_in_milliseconds = merge_gap_in_milliseconds_tensor.Data()[0];

  std::vector<std::pair<int64_t, int64_t>> segments;
  segments.reserve(num_segments);
  for (int64_t i = 0; i < num_segments; ++i) {
    int64_t start = seg_data[i * 2 + 0];
    int64_t end = seg_data[i * 2 + 1];
    segments.emplace_back(start, end);
  }

  // Sort by start time.
  std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  // Merge overlapping or close segments.
  std::vector<std::pair<int64_t, int64_t>> merged;
  int64_t cur_start = segments[0].first;
  int64_t cur_end = segments[0].second;

  for (size_t i = 1; i < segments.size(); ++i) {
    const int64_t s = segments[i].first;
    const int64_t e = segments[i].second;

    if (static_cast<double>(s - cur_end) <= merge_gap_in_milliseconds) {
      std::cout << "prolonging segments.." << std::endl;
      cur_end = std::max(cur_end, e);
    } else {
      std::cout << "merging segments.." << cur_end << std::endl;
      merged.emplace_back(cur_start, cur_end);
      cur_start = s;
      cur_end = e;
    }
  }
  merged.emplace_back(cur_start, cur_end);

  const int64_t m = static_cast<int64_t>(merged.size());
  int64_t* out = output0.Allocate({m, 2});

  for (int64_t i = 0; i < m; ++i) {
    out[i * 2 + 0] = merged[i].first;
    out[i * 2 + 1] = merged[i].second;
  }

  return nullptr;
}