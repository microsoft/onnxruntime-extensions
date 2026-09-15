// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

#include <gsl/span>
#include <string>
#include <vector>

namespace ort_extensions {

struct AsrDiarizationMergeConfig {
  int64_t max_num_speakers{4};
  int64_t left_frame_shift{-1};
  int64_t right_frame_shift{0};
  float min_sigmoid_value{0.01f};
};

struct AsrDiarizationMergeResult {
  std::vector<int64_t> word_speaker_ids;
  std::vector<float> word_speaker_scores;
  int64_t score_columns{};
  std::vector<std::string> turn_texts;
  std::vector<int64_t> turn_speaker_ids;
  std::vector<float> turn_start_times;
  std::vector<float> turn_end_times;
  std::vector<std::string> plain_lines;
  std::vector<std::string> seglst_records;
  std::vector<std::string> stm_lines;
  std::vector<std::string> ctm_lines;
};

OrtxStatus MergeAsrWithRawDiarization(const std::vector<std::string>& words, gsl::span<const int64_t> word_start_frames,
                                      gsl::span<const int64_t> word_end_frames, gsl::span<const float> word_start_times,
                                      gsl::span<const float> word_end_times, const float* diarization_scores,
                                      int64_t diarization_frames, int64_t diarization_speakers,
                                      std::string_view session_id, const AsrDiarizationMergeConfig& config,
                                      AsrDiarizationMergeResult& result);

OrtxStatus MergeAsrWithDiarizationSegments(
    const std::vector<std::string>& words, gsl::span<const int64_t> word_start_frames,
    gsl::span<const int64_t> word_end_frames, gsl::span<const float> word_start_times,
    gsl::span<const float> word_end_times, gsl::span<const float> segment_start_times,
    gsl::span<const float> segment_end_times, gsl::span<const int64_t> segment_speaker_ids, std::string_view session_id,
    const AsrDiarizationMergeConfig& config, AsrDiarizationMergeResult& result);

struct AsrDiarizationMergeRaw {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);
  OrtxStatus Compute(const ortc::Tensor<std::string>& words, const ortc::Tensor<int64_t>& word_start_frames,
                     const ortc::Tensor<int64_t>& word_end_frames, const ortc::Tensor<float>& word_start_times,
                     const ortc::Tensor<float>& word_end_times, const ortc::Tensor<float>& diarization_scores,
                     const ortc::Tensor<std::string>& session_id, ortc::Tensor<int64_t>& word_speaker_ids,
                     ortc::Tensor<float>& word_speaker_scores, ortc::Tensor<std::string>& turn_texts,
                     ortc::Tensor<int64_t>& turn_speaker_ids, ortc::Tensor<float>& turn_start_times,
                     ortc::Tensor<float>& turn_end_times, ortc::Tensor<std::string>& plain_lines,
                     ortc::Tensor<std::string>& seglst_records, ortc::Tensor<std::string>& stm_lines,
                     ortc::Tensor<std::string>& ctm_lines) const;

 private:
  AsrDiarizationMergeConfig config_;
};

struct AsrDiarizationMergeSegments {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);
  OrtxStatus Compute(const ortc::Tensor<std::string>& words, const ortc::Tensor<int64_t>& word_start_frames,
                     const ortc::Tensor<int64_t>& word_end_frames, const ortc::Tensor<float>& word_start_times,
                     const ortc::Tensor<float>& word_end_times, const ortc::Tensor<float>& segment_start_times,
                     const ortc::Tensor<float>& segment_end_times, const ortc::Tensor<int64_t>& segment_speaker_ids,
                     const ortc::Tensor<std::string>& session_id, ortc::Tensor<int64_t>& word_speaker_ids,
                     ortc::Tensor<float>& word_speaker_scores, ortc::Tensor<std::string>& turn_texts,
                     ortc::Tensor<int64_t>& turn_speaker_ids, ortc::Tensor<float>& turn_start_times,
                     ortc::Tensor<float>& turn_end_times, ortc::Tensor<std::string>& plain_lines,
                     ortc::Tensor<std::string>& seglst_records, ortc::Tensor<std::string>& stm_lines,
                     ortc::Tensor<std::string>& ctm_lines) const;

 private:
  AsrDiarizationMergeConfig config_;
};

}  // namespace ort_extensions