// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include "../../shared/api/tokenizer_speech_metadata.h"

TEST(SpeechMetadata, AggregatesEachWordWindowRatherThanTokens) {
  const double start[8]{0, 0, 0, 0, 0, 0, 0, 0};
  const double first[8]{0.9, 0.1, 0, 0, 0, 0, 0, 0};
  const double stop[8]{1.1, 1.9, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(ort_extensions::AggregateWordSpeaker(start, first, 0, 1), 0);
  EXPECT_EQ(ort_extensions::AggregateWordSpeaker(first, stop, 1, 3), 1);
  EXPECT_EQ(ort_extensions::AggregateWordSpeaker(start, stop, 0, 3), 1);
  EXPECT_THROW(ort_extensions::AggregateWordSpeaker(start, stop, 3, 3), std::runtime_error);
}

TEST(SpeechMetadata, TurnsIgnoreGapAndPunctuation) {
  EXPECT_FALSE(ort_extensions::SpeechSegmentStartsNew(true, true, 2, 2, 1000, 3, 0));
  EXPECT_TRUE(ort_extensions::SpeechSegmentStartsNew(true, true, 2, 3, 3, 3, -1));
  EXPECT_TRUE(ort_extensions::SpeechSegmentStartsNew(true, false, -1, -1, 10, 3, 5));
  const char* separators[]{"。", "."};
  EXPECT_TRUE(ort_extensions::SpeechSegmentEnds("Hello. \n", separators, 2));
  EXPECT_TRUE(ort_extensions::SpeechSegmentEnds("word。", separators, 2));
  EXPECT_FALSE(ort_extensions::SpeechSegmentEnds("Hello", separators, 2));
}
