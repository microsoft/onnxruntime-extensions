// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "asr_diarization_merge.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace ort_extensions {
namespace {

OrtxStatus InvalidArgument(std::string message) { return {kOrtxErrorInvalidArgument, std::move(message)}; }

OrtxStatus ValidateWords(const std::vector<std::string>& words, gsl::span<const int64_t> start_frames,
                         gsl::span<const int64_t> end_frames, gsl::span<const float> start_times,
                         gsl::span<const float> end_times, const AsrDiarizationMergeConfig& config) {
  const size_t count = words.size();
  if (start_frames.size() != count || end_frames.size() != count || start_times.size() != count ||
      end_times.size() != count) {
    return InvalidArgument("ASR word, frame, and time inputs must have the same number of elements");
  }
  if (config.max_num_speakers <= 0) {
    return InvalidArgument("max_num_speakers must be greater than zero");
  }
  for (size_t index = 0; index < count; ++index) {
    if (start_frames[index] < 0 || end_frames[index] < start_frames[index]) {
      return InvalidArgument("word frame intervals must satisfy 0 <= start <= end");
    }
    if (!std::isfinite(start_times[index]) || !std::isfinite(end_times[index]) ||
        end_times[index] < start_times[index]) {
      return InvalidArgument("word time intervals must be finite and satisfy start <= end");
    }
  }
  return {};
}

std::string EscapeJson(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

std::string FormatFloat(float value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << value;
  return stream.str();
}

void BuildTranscriptOutputs(const std::vector<std::string>& words, gsl::span<const float> start_times,
                            gsl::span<const float> end_times, std::string_view session_id,
                            AsrDiarizationMergeResult& result) {
  result.turn_texts.clear();
  result.turn_speaker_ids.clear();
  result.turn_start_times.clear();
  result.turn_end_times.clear();

  for (size_t index = 0; index < words.size(); ++index) {
    const int64_t speaker = result.word_speaker_ids[index];
    if (result.turn_speaker_ids.empty() || result.turn_speaker_ids.back() != speaker) {
      result.turn_texts.push_back(words[index]);
      result.turn_speaker_ids.push_back(speaker);
      result.turn_start_times.push_back(start_times[index]);
      result.turn_end_times.push_back(end_times[index]);
    } else {
      if (!result.turn_texts.back().empty() && !words[index].empty()) result.turn_texts.back() += ' ';
      result.turn_texts.back() += words[index];
      result.turn_end_times.back() = end_times[index];
    }
  }

  result.plain_lines.clear();
  result.seglst_records.clear();
  result.stm_lines.clear();
  for (size_t index = 0; index < result.turn_texts.size(); ++index) {
    const std::string speaker = "speaker_" + std::to_string(result.turn_speaker_ids[index]);
    const std::string start = FormatFloat(result.turn_start_times[index]);
    const std::string end = FormatFloat(result.turn_end_times[index]);
    result.plain_lines.push_back(speaker + ": " + result.turn_texts[index]);
    result.seglst_records.push_back("{\"speaker\":\"" + speaker + "\",\"start_time\":" + start +
                                    ",\"end_time\":" + end + ",\"words\":\"" + EscapeJson(result.turn_texts[index]) +
                                    "\",\"session_id\":\"" + EscapeJson(session_id) + "\"}");
    result.stm_lines.push_back(std::string(session_id) + " 1 " + speaker + " " + start + " " + end + " " +
                               result.turn_texts[index]);
  }

  result.ctm_lines.clear();
  for (size_t index = 0; index < words.size(); ++index) {
    const float duration = std::max(0.0f, end_times[index] - start_times[index]);
    result.ctm_lines.push_back(std::string(session_id) + " 1 " + FormatFloat(start_times[index]) + " " +
                               FormatFloat(duration) + " " + words[index]);
  }
}

void SetOutputs(const AsrDiarizationMergeResult& result, ortc::Tensor<int64_t>& word_speaker_ids,
                ortc::Tensor<float>& word_speaker_scores, ortc::Tensor<std::string>& turn_texts,
                ortc::Tensor<int64_t>& turn_speaker_ids, ortc::Tensor<float>& turn_start_times,
                ortc::Tensor<float>& turn_end_times, ortc::Tensor<std::string>& plain_lines,
                ortc::Tensor<std::string>& seglst_records, ortc::Tensor<std::string>& stm_lines,
                ortc::Tensor<std::string>& ctm_lines) {
  auto* speaker_output = word_speaker_ids.Allocate({static_cast<int64_t>(result.word_speaker_ids.size())});
  std::copy(result.word_speaker_ids.begin(), result.word_speaker_ids.end(), speaker_output);

  auto* score_output =
      word_speaker_scores.Allocate({static_cast<int64_t>(result.word_speaker_ids.size()), result.score_columns});
  std::copy(result.word_speaker_scores.begin(), result.word_speaker_scores.end(), score_output);

  turn_texts.SetStringOutput(result.turn_texts, {static_cast<int64_t>(result.turn_texts.size())});
  auto* turn_speaker_output = turn_speaker_ids.Allocate({static_cast<int64_t>(result.turn_speaker_ids.size())});
  std::copy(result.turn_speaker_ids.begin(), result.turn_speaker_ids.end(), turn_speaker_output);
  auto* turn_start_output = turn_start_times.Allocate({static_cast<int64_t>(result.turn_start_times.size())});
  std::copy(result.turn_start_times.begin(), result.turn_start_times.end(), turn_start_output);
  auto* turn_end_output = turn_end_times.Allocate({static_cast<int64_t>(result.turn_end_times.size())});
  std::copy(result.turn_end_times.begin(), result.turn_end_times.end(), turn_end_output);
  plain_lines.SetStringOutput(result.plain_lines, {static_cast<int64_t>(result.plain_lines.size())});
  seglst_records.SetStringOutput(result.seglst_records, {static_cast<int64_t>(result.seglst_records.size())});
  stm_lines.SetStringOutput(result.stm_lines, {static_cast<int64_t>(result.stm_lines.size())});
  ctm_lines.SetStringOutput(result.ctm_lines, {static_cast<int64_t>(result.ctm_lines.size())});
}

OrtStatusPtr ReadRawConfig(const OrtKernelInfo& info, AsrDiarizationMergeConfig& config) {
  ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "max_num_speakers", config.max_num_speakers));
  ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "left_frame_shift", config.left_frame_shift));
  ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "right_frame_shift", config.right_frame_shift));
  return OrtW::GetOpAttribute(info, "min_sigmoid_value", config.min_sigmoid_value);
}

}  // namespace

OrtxStatus MergeAsrWithRawDiarization(const std::vector<std::string>& words, gsl::span<const int64_t> word_start_frames,
                                      gsl::span<const int64_t> word_end_frames, gsl::span<const float> word_start_times,
                                      gsl::span<const float> word_end_times, const float* diarization_scores,
                                      int64_t diarization_frames, int64_t diarization_speakers,
                                      std::string_view session_id, const AsrDiarizationMergeConfig& config,
                                      AsrDiarizationMergeResult& result) {
  auto status = ValidateWords(words, word_start_frames, word_end_frames, word_start_times, word_end_times, config);
  if (!status.IsOk()) return status;
  if (diarization_scores == nullptr || diarization_frames <= 0 || diarization_speakers <= 0) {
    return InvalidArgument("raw diarization scores must have shape [frames, speakers] with nonzero dimensions");
  }
  if (config.max_num_speakers > diarization_speakers) {
    return InvalidArgument("max_num_speakers cannot exceed the diarization score columns");
  }
  if (config.min_sigmoid_value < 0.0f || config.min_sigmoid_value > 1.0f) {
    return InvalidArgument("min_sigmoid_value must be in [0, 1]");
  }

  result = {};
  result.score_columns = diarization_speakers;
  result.word_speaker_ids.reserve(words.size());
  result.word_speaker_scores.reserve(words.size() * static_cast<size_t>(diarization_speakers));
  std::vector<float> scores(static_cast<size_t>(diarization_speakers));

  for (size_t word_index = 0; word_index < words.size(); ++word_index) {
    int64_t start_frame = word_start_frames[word_index];
    int64_t end_frame = word_end_frames[word_index];
    if (start_frame == end_frame) {
      start_frame = std::min(start_frame, diarization_frames - 1);
      end_frame = start_frame + 1;
    }

    int64_t average_start = std::clamp(start_frame + config.left_frame_shift, int64_t{0}, diarization_frames - 1);
    int64_t average_end = std::clamp(end_frame + config.right_frame_shift, average_start + 1, diarization_frames);
    const float frame_count = static_cast<float>(average_end - average_start);
    std::fill(scores.begin(), scores.end(), 0.0f);
    for (int64_t frame = average_start; frame < average_end; ++frame) {
      for (int64_t speaker = 0; speaker < diarization_speakers; ++speaker) {
        scores[static_cast<size_t>(speaker)] +=
            diarization_scores[frame * diarization_speakers + speaker] / frame_count;
      }
    }

    for (float& score : scores) score = std::clamp(score, config.min_sigmoid_value, 1.0f);
    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    for (float& score : scores) score = sum > 0.0f ? score / sum : 1.0f / diarization_speakers;
    std::fill(scores.begin() + config.max_num_speakers, scores.end(), 0.0f);
    const auto speaker = static_cast<int64_t>(
        std::distance(scores.begin(), std::max_element(scores.begin(), scores.begin() + config.max_num_speakers)));
    result.word_speaker_ids.push_back(speaker);
    result.word_speaker_scores.insert(result.word_speaker_scores.end(), scores.begin(), scores.end());
  }

  BuildTranscriptOutputs(words, word_start_times, word_end_times, session_id, result);
  return {};
}

OrtxStatus MergeAsrWithDiarizationSegments(
    const std::vector<std::string>& words, gsl::span<const int64_t> word_start_frames,
    gsl::span<const int64_t> word_end_frames, gsl::span<const float> word_start_times,
    gsl::span<const float> word_end_times, gsl::span<const float> segment_start_times,
    gsl::span<const float> segment_end_times, gsl::span<const int64_t> segment_speaker_ids, std::string_view session_id,
    const AsrDiarizationMergeConfig& config, AsrDiarizationMergeResult& result) {
  auto status = ValidateWords(words, word_start_frames, word_end_frames, word_start_times, word_end_times, config);
  if (!status.IsOk()) return status;
  if (segment_start_times.size() != segment_end_times.size() ||
      segment_start_times.size() != segment_speaker_ids.size()) {
    return InvalidArgument("diarization segment inputs must have the same number of elements");
  }

  result = {};
  result.score_columns = config.max_num_speakers;
  result.word_speaker_ids.reserve(words.size());
  result.word_speaker_scores.assign(words.size() * static_cast<size_t>(config.max_num_speakers), 0.0f);
  for (size_t segment = 0; segment < segment_start_times.size(); ++segment) {
    if (!std::isfinite(segment_start_times[segment]) || !std::isfinite(segment_end_times[segment]) ||
        segment_end_times[segment] < segment_start_times[segment] || segment_speaker_ids[segment] < 0) {
      return InvalidArgument("diarization segments must have finite ordered times and nonnegative speaker IDs");
    }
  }

  for (size_t word_index = 0; word_index < words.size(); ++word_index) {
    int64_t best_speaker = 0;
    float best_overlap = 0.0f;
    for (size_t segment = 0; segment < segment_start_times.size(); ++segment) {
      const int64_t speaker = segment_speaker_ids[segment];
      if (speaker >= config.max_num_speakers) continue;
      const float overlap = std::min(word_end_times[word_index], segment_end_times[segment]) -
                            std::max(word_start_times[word_index], segment_start_times[segment]);
      if (overlap > best_overlap) {
        best_overlap = overlap;
        best_speaker = speaker;
      }
    }

    if (best_overlap <= 0.0f && !segment_start_times.empty()) {
      const float center = 0.5f * (word_start_times[word_index] + word_end_times[word_index]);
      float best_distance = std::numeric_limits<float>::infinity();
      for (size_t segment = 0; segment < segment_start_times.size(); ++segment) {
        const int64_t speaker = segment_speaker_ids[segment];
        if (speaker >= config.max_num_speakers) continue;
        const float distance = center >= segment_start_times[segment] && center <= segment_end_times[segment]
                                   ? 0.0f
                                   : std::min(std::abs(center - segment_start_times[segment]),
                                              std::abs(center - segment_end_times[segment]));
        if (distance < best_distance) {
          best_distance = distance;
          best_speaker = speaker;
        }
      }
    }

    result.word_speaker_ids.push_back(best_speaker);
    result.word_speaker_scores[word_index * static_cast<size_t>(config.max_num_speakers) +
                               static_cast<size_t>(best_speaker)] = 1.0f;
  }

  BuildTranscriptOutputs(words, word_start_times, word_end_times, session_id, result);
  return {};
}

OrtStatusPtr AsrDiarizationMergeRaw::OnModelAttach(const OrtApi&, const OrtKernelInfo& info) {
  return ReadRawConfig(info, config_);
}

OrtxStatus AsrDiarizationMergeRaw::Compute(
    const ortc::Tensor<std::string>& words, const ortc::Tensor<int64_t>& word_start_frames,
    const ortc::Tensor<int64_t>& word_end_frames, const ortc::Tensor<float>& word_start_times,
    const ortc::Tensor<float>& word_end_times, const ortc::Tensor<float>& diarization_scores,
    const ortc::Tensor<std::string>& session_id, ortc::Tensor<int64_t>& word_speaker_ids,
    ortc::Tensor<float>& word_speaker_scores, ortc::Tensor<std::string>& turn_texts,
    ortc::Tensor<int64_t>& turn_speaker_ids, ortc::Tensor<float>& turn_start_times, ortc::Tensor<float>& turn_end_times,
    ortc::Tensor<std::string>& plain_lines, ortc::Tensor<std::string>& seglst_records,
    ortc::Tensor<std::string>& stm_lines, ortc::Tensor<std::string>& ctm_lines) const {
  if (diarization_scores.Shape().size() != 2 || session_id.Data().size() != 1) {
    return InvalidArgument("raw diarization scores must be rank 2 and session_id must contain one string");
  }
  AsrDiarizationMergeResult result;
  const auto word_count = words.Data().size();
  if (word_start_frames.NumberOfElement() != word_count || word_end_frames.NumberOfElement() != word_count ||
      word_start_times.NumberOfElement() != word_count || word_end_times.NumberOfElement() != word_count) {
    return InvalidArgument("ASR word, frame, and time inputs must have the same number of elements");
  }
  auto status = MergeAsrWithRawDiarization(
      words.Data(), {word_start_frames.Data(), word_count}, {word_end_frames.Data(), word_count},
      {word_start_times.Data(), word_count}, {word_end_times.Data(), word_count}, diarization_scores.Data(),
      diarization_scores.Shape()[0], diarization_scores.Shape()[1], session_id.Data()[0], config_, result);
  if (!status.IsOk()) return status;
  SetOutputs(result, word_speaker_ids, word_speaker_scores, turn_texts, turn_speaker_ids, turn_start_times,
             turn_end_times, plain_lines, seglst_records, stm_lines, ctm_lines);
  return {};
}

OrtStatusPtr AsrDiarizationMergeSegments::OnModelAttach(const OrtApi&, const OrtKernelInfo& info) {
  return OrtW::GetOpAttribute(info, "max_num_speakers", config_.max_num_speakers);
}

OrtxStatus AsrDiarizationMergeSegments::Compute(
    const ortc::Tensor<std::string>& words, const ortc::Tensor<int64_t>& word_start_frames,
    const ortc::Tensor<int64_t>& word_end_frames, const ortc::Tensor<float>& word_start_times,
    const ortc::Tensor<float>& word_end_times, const ortc::Tensor<float>& segment_start_times,
    const ortc::Tensor<float>& segment_end_times, const ortc::Tensor<int64_t>& segment_speaker_ids,
    const ortc::Tensor<std::string>& session_id, ortc::Tensor<int64_t>& word_speaker_ids,
    ortc::Tensor<float>& word_speaker_scores, ortc::Tensor<std::string>& turn_texts,
    ortc::Tensor<int64_t>& turn_speaker_ids, ortc::Tensor<float>& turn_start_times, ortc::Tensor<float>& turn_end_times,
    ortc::Tensor<std::string>& plain_lines, ortc::Tensor<std::string>& seglst_records,
    ortc::Tensor<std::string>& stm_lines, ortc::Tensor<std::string>& ctm_lines) const {
  if (session_id.Data().size() != 1) return InvalidArgument("session_id must contain one string");
  AsrDiarizationMergeResult result;
  const auto word_count = words.Data().size();
  if (word_start_frames.NumberOfElement() != word_count || word_end_frames.NumberOfElement() != word_count ||
      word_start_times.NumberOfElement() != word_count || word_end_times.NumberOfElement() != word_count) {
    return InvalidArgument("ASR word, frame, and time inputs must have the same number of elements");
  }
  const size_t segment_count = static_cast<size_t>(segment_start_times.NumberOfElement());
  if (segment_end_times.NumberOfElement() != segment_count || segment_speaker_ids.NumberOfElement() != segment_count) {
    return InvalidArgument("diarization segment inputs must have the same number of elements");
  }
  auto status = MergeAsrWithDiarizationSegments(
      words.Data(), {word_start_frames.Data(), word_count}, {word_end_frames.Data(), word_count},
      {word_start_times.Data(), word_count}, {word_end_times.Data(), word_count},
      {segment_start_times.Data(), segment_count}, {segment_end_times.Data(), segment_count},
      {segment_speaker_ids.Data(), segment_count}, session_id.Data()[0], config_, result);
  if (!status.IsOk()) return status;
  SetOutputs(result, word_speaker_ids, word_speaker_scores, turn_texts, turn_speaker_ids, turn_start_times,
             turn_end_times, plain_lines, seglst_records, stm_lines, ctm_lines);
  return {};
}

}  // namespace ort_extensions