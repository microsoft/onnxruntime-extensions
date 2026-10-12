// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ort_extensions {

inline int AggregateWordSpeaker(const double* start_sums, const double* stop_sums, int64_t start_count,
                                int64_t stop_count) {
  if (!start_sums || !stop_sums)
    throw std::runtime_error("Speaker probability sums must not be null");
  if (start_count < 0 || stop_count <= start_count)
    throw std::runtime_error("Speaker evidence requires a positive evaluated-frame count");
  double scores[8];
  double total = 0;
  for (size_t speaker = 0; speaker < 8; ++speaker) {
    const double sum = stop_sums[speaker] - start_sums[speaker];
    const double count = static_cast<double>(stop_count - start_count);
    if (!std::isfinite(start_sums[speaker]) || !std::isfinite(stop_sums[speaker]) || sum < -1e-9 || sum > count + 1e-9)
      throw std::runtime_error("Speaker evidence contains invalid probability sums");
    scores[speaker] = std::clamp(sum / count, 0.01, 1.0);
    total += scores[speaker];
  }
  for (double& score : scores) score /= total;
  return static_cast<int>(std::max_element(scores, scores + 8) - scores);
}

inline bool SpeechSegmentStartsNew(bool has_pending, bool diarization, int pending_speaker, int speaker, int64_t start,
                                   int64_t previous_stop, int64_t gap) {
  if (!has_pending) return false;
  if (diarization) return pending_speaker != speaker;
  return gap >= 0 && (gap == 0 || start - previous_stop >= gap);
}

inline bool SpeechSegmentEnds(std::string_view text, const char* const* separators, size_t count) {
  if (count && !separators) throw std::runtime_error("Segment separators must not be null");
  while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\n' || text.back() == '\r' ||
                           text.back() == '\f' || text.back() == '\v'))
    text.remove_suffix(1);
  for (size_t i = 0; i < count; ++i) {
    if (!separators[i]) throw std::runtime_error("Segment separator must not be null");
    const std::string_view separator{separators[i]};
    if (!separator.empty() && text.size() >= separator.size() &&
        text.substr(text.size() - separator.size()) == separator)
      return true;
  }
  return false;
}

}  // namespace ort_extensions
