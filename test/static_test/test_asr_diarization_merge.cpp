// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "audio/asr_diarization_merge.h"

namespace {

TEST(AsrDiarizationMergeTest, RawModeUsesFrameScoresAndBuildsFormats) {
  const std::vector<std::string> words{"hello", "there", "again"};
  const std::vector<int64_t> start_frames{0, 2, 3};
  const std::vector<int64_t> end_frames{1, 3, 3};
  const std::vector<float> start_times{0.0f, 0.16f, 0.24f};
  const std::vector<float> end_times{0.08f, 0.24f, 0.32f};
  const std::vector<float> scores{
      0.9f, 0.1f, 0.8f, 0.2f, 0.2f, 0.8f, 0.1f, 0.9f,
  };
  ort_extensions::AsrDiarizationMergeConfig config;
  config.max_num_speakers = 2;
  config.left_frame_shift = 0;
  config.right_frame_shift = 0;
  config.min_sigmoid_value = 0.0f;
  ort_extensions::AsrDiarizationMergeResult result;

  const auto status = ort_extensions::MergeAsrWithRawDiarization(
      words, start_frames, end_frames, start_times, end_times, scores.data(), 4, 2, "sample", config, result);

  ASSERT_TRUE(status.IsOk()) << status.Message();
  EXPECT_EQ(result.word_speaker_ids, (std::vector<int64_t>{0, 1, 1}));
  ASSERT_EQ(result.turn_texts.size(), 2U);
  EXPECT_EQ(result.turn_texts[0], "hello");
  EXPECT_EQ(result.turn_texts[1], "there again");
  EXPECT_EQ(result.plain_lines[1], "speaker_1: there again");
  EXPECT_EQ(result.stm_lines[0], "sample 1 speaker_0 0.000 0.080 hello");
  EXPECT_EQ(result.ctm_lines[2], "sample 1 0.240 0.080 again");
  EXPECT_EQ(result.seglst_records[1],
            "{\"speaker\":\"speaker_1\",\"start_time\":0.160,\"end_time\":0.320,"
            "\"words\":\"there again\",\"session_id\":\"sample\"}");
}

TEST(AsrDiarizationMergeTest, RawModeAppliesNeMoFrameWindow) {
  const std::vector<std::string> words{"shifted"};
  const std::vector<int64_t> frames{2};
  const std::vector<int64_t> end_frames{3};
  const std::vector<float> start_times{0.16f};
  const std::vector<float> end_times{0.24f};
  const std::vector<float> scores{0.9f, 0.1f, 0.95f, 0.05f, 0.1f, 0.9f};
  ort_extensions::AsrDiarizationMergeConfig config;
  config.max_num_speakers = 2;
  config.left_frame_shift = -1;
  config.right_frame_shift = 0;
  config.min_sigmoid_value = 0.0f;
  ort_extensions::AsrDiarizationMergeResult result;

  const auto status = ort_extensions::MergeAsrWithRawDiarization(words, frames, end_frames, start_times, end_times,
                                                                 scores.data(), 3, 2, "sample", config, result);

  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_EQ(result.word_speaker_ids.size(), 1U);
  EXPECT_EQ(result.word_speaker_ids[0], 0);
}

TEST(AsrDiarizationMergeTest, SegmentModeUsesOverlapThenNearestSegment) {
  const std::vector<std::string> words{"one", "two", "three"};
  const std::vector<int64_t> start_frames{0, 5, 15};
  const std::vector<int64_t> end_frames{2, 7, 17};
  const std::vector<float> start_times{0.0f, 0.4f, 1.2f};
  const std::vector<float> end_times{0.2f, 0.6f, 1.4f};
  const std::vector<float> segment_starts{0.0f, 0.7f};
  const std::vector<float> segment_ends{0.5f, 1.0f};
  const std::vector<int64_t> segment_speakers{0, 1};
  ort_extensions::AsrDiarizationMergeConfig config;
  config.max_num_speakers = 2;
  ort_extensions::AsrDiarizationMergeResult result;

  const auto status = ort_extensions::MergeAsrWithDiarizationSegments(words, start_frames, end_frames, start_times,
                                                                      end_times, segment_starts, segment_ends,
                                                                      segment_speakers, "sample", config, result);

  ASSERT_TRUE(status.IsOk()) << status.Message();
  EXPECT_EQ(result.word_speaker_ids, (std::vector<int64_t>{0, 0, 1}));
  ASSERT_EQ(result.turn_texts.size(), 2U);
  EXPECT_EQ(result.turn_texts[0], "one two");
  EXPECT_EQ(result.turn_texts[1], "three");
  EXPECT_FLOAT_EQ(result.word_speaker_scores[0], 1.0f);
  EXPECT_FLOAT_EQ(result.word_speaker_scores[5], 1.0f);
}

}  // namespace