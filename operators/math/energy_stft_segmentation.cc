// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "energy_stft_segmentation.hpp"
#include "dlib/stft_norm.hpp"
#include "../../shared/api/c_api_utils.hpp"

OrtStatusPtr detect_energy_segments(const ortc::Tensor<float>& audio,
                                    const ortc::Tensor<int64_t>& sr_tensor,
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

  std::vector<float> hann = hann_window(static_cast<int>(n_fft));

  ortc::Tensor<float> stft_out(&ort_extensions::CppAllocator::Instance());
  StftNormal stft;
  auto status = stft.Compute(audio, n_fft, hop_length, {hann.data(), hann.size()}, n_fft, stft_out);
  if (!status.IsOk()) {
    return OrtW::CreateStatus("STFT Compute failed", ORT_FAIL);
  }

  const auto& stft_shape = stft_out.Shape();
  const int64_t n_freq = stft_shape[1];
  const int64_t n_frames = stft_shape[2];
  const float* spec_ptr = stft_out.Data();

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

  std::vector<std::pair<float, float>> segments;
  bool active = false;
  int64_t start_idx = 0;

  for (int64_t t = 0; t < n_frames; ++t) {
    bool above_threshold = energy_db[t] > threshold;
    if (!active && above_threshold) {
      active = true;
      start_idx = t;
    } else if (active && !above_threshold) {
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
    out_data[i * 2] = static_cast<int64_t>(segments[i].first * 1000.0f);
    out_data[i * 2 + 1] = static_cast<int64_t>(segments[i].second * 1000.0f);
  }

  return nullptr;
}

OrtStatusPtr merge_and_filter_segments(const Ort::Custom::Tensor<int64_t>& segments_tensor,
                                       const Ort::Custom::Tensor<int64_t>& merge_gap_ms_tensor,
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

  const int64_t merge_gap_ms = merge_gap_ms_tensor.Data()[0];

  // Merge overlapping or close segments directly
  std::vector<std::pair<int64_t, int64_t>> merged;
  merged.reserve(num_segments);

  int64_t cur_start = seg_data[0];
  int64_t cur_end = seg_data[1];

  for (int64_t i = 1; i < num_segments; ++i) {
    const int64_t s = seg_data[i * 2];
    const int64_t e = seg_data[i * 2 + 1];

    if (s - cur_end <= merge_gap_ms) {
      cur_end = std::max(cur_end, e);
    } else {
      merged.emplace_back(cur_start, cur_end);
      cur_start = s;
      cur_end = e;
    }
  }
  merged.emplace_back(cur_start, cur_end);

  const int64_t m = static_cast<int64_t>(merged.size());
  int64_t* out = output0.Allocate({m, 2});
  for (int64_t i = 0; i < m; ++i) {
    out[i * 2] = merged[i].first;
    out[i * 2 + 1] = merged[i].second;
  }

  return nullptr;
}
