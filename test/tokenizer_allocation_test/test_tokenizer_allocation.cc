// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory_resource>
#include <new>
#include <string>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#include "gtest/gtest.h"
#include "shared/api/tokenizer_impl.h"

namespace {
thread_local bool audit_allocations = false;
thread_local size_t allocation_count = 0;

struct ScopedAllocationAudit {
  ScopedAllocationAudit() {
    allocation_count = 0;
    audit_allocations = true;
  }
  ~ScopedAllocationAudit() { audit_allocations = false; }
};

void RecordAllocation() {
  if (audit_allocations) ++allocation_count;
}
}  // namespace

// These replacements are linked only into this dedicated test executable.
void* operator new(size_t bytes) {
  RecordAllocation();
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc{};
}

void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, size_t) noexcept { std::free(pointer); }

void* operator new(size_t bytes, std::align_val_t alignment) {
  RecordAllocation();
#ifdef _WIN32
  if (void* pointer = _aligned_malloc(bytes ? bytes : 1, static_cast<size_t>(alignment))) return pointer;
#else
  void* pointer{};
  if (posix_memalign(&pointer, static_cast<size_t>(alignment), bytes ? bytes : 1) == 0) return pointer;
#endif
  throw std::bad_alloc{};
}

void* operator new[](size_t bytes, std::align_val_t alignment) { return ::operator new(bytes, alignment); }
void operator delete(void* pointer, std::align_val_t) noexcept {
#ifdef _WIN32
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}
void operator delete[](void* pointer, std::align_val_t alignment) noexcept { ::operator delete(pointer, alignment); }
void operator delete(void* pointer, size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete[](void* pointer, size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

TEST(TokenizerAllocationTest, AuditDetectsOrdinaryAndAlignedAllocations) {
  size_t observed{};
  {
    ScopedAllocationAudit audit;
    void* ordinary = ::operator new(1);
    void* array = ::operator new[](1);
    void* aligned = ::operator new(1, std::align_val_t{64});
    void* aligned_array = ::operator new[](1, std::align_val_t{64});
    observed = allocation_count;
    ::operator delete(ordinary);
    ::operator delete[](array);
    ::operator delete(aligned, std::align_val_t{64});
    ::operator delete[](aligned_array, std::align_val_t{64});
  }
  EXPECT_EQ(observed, 4u);
}

TEST(TokenizerAllocationTest, BoundedEncodingUsesOnlyCallerScratch) {
  ort_extensions::TokenizerImpl tokenizer;
  ASSERT_TRUE(tokenizer.Load("data/phi-2").IsOk());
  ASSERT_TRUE(tokenizer.UpdateOptions({{"add_special_tokens", "false"}}).IsOk());
  ASSERT_TRUE(tokenizer.PrepareBoundedEncoding().IsOk());
  extTokenId_t added_token{};
  ASSERT_TRUE(tokenizer.Token2Id("<|endoftext|>", added_token).IsOk());

  const std::array<std::string, 3> inputs{"def f(x):\n    return x + 12345  # caf\xc3\xa9\n",
                                          std::string(512, ' ') + "\t\n", "before<|endoftext|>after<|endoftext|>end"};
  for (const auto& text : inputs) {
    SCOPED_TRACE(text.substr(0, 40));
    alignas(std::max_align_t) std::array<std::byte, 65536> backing{};
    std::pmr::monotonic_buffer_resource scratch{backing.data(), backing.size(), std::pmr::null_memory_resource()};
    std::vector<extTokenId_t> output(text.size() + 2);
    size_t count{};
    size_t uncharged_allocations{};
    OrtxStatus status;
    {
      ScopedAllocationAudit audit;
      status = tokenizer.EncodeIntoBuffer(text, output.data(), output.size(), count, scratch);
      uncharged_allocations = allocation_count;
    }
    ASSERT_TRUE(status.IsOk()) << status.ToString();
    EXPECT_EQ(uncharged_allocations, 0u);
    ASSERT_GT(count, 0u);

    std::vector<std::vector<extTokenId_t>> legacy;
    ASSERT_TRUE(tokenizer.Tokenize({text}, legacy, false).IsOk());
    ASSERT_EQ(legacy.size(), 1u);
    ASSERT_EQ(count, legacy[0].size());
    EXPECT_TRUE(std::equal(output.begin(), output.begin() + count, legacy[0].begin()));
    if (text.find("<|endoftext|>") != std::string::npos) {
      EXPECT_EQ(std::count(output.begin(), output.begin() + count, added_token), 2);
    }
  }
}
