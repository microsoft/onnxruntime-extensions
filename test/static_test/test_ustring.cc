#include "gtest/gtest.h"
#include <cstring>
#include "kernels/ustring.hpp"

void convert_test(const char* const_str) {
  std::string string(const_str);
  const std::string const_string(const_str);

  auto str = std::shared_ptr<char>(strdup(const_str));
  ustring char_construct(str.get());
  EXPECT_EQ(const_string, std::string(char_construct));

  ustring const_char_construct(const_str);
  EXPECT_EQ(const_string, std::string(const_char_construct));

  ustring string_construct(string);
  EXPECT_EQ(const_string, std::string(string_construct));

  ustring const_string_construct(const_string);
  EXPECT_EQ(const_string, std::string(const_string_construct));

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
  std::u32string u32_str(cvt.from_bytes(const_string));
  const std::u32string u32_const_str(cvt.from_bytes(const_string));

  ustring u32string_construct(u32_str);
  EXPECT_EQ(const_string, std::string(u32string_construct));

  ustring u32string_const_construct(u32_const_str);
  EXPECT_EQ(const_string, std::string(u32string_const_construct));

  ustring u32string_move_construct(std::move(u32_const_str));
  EXPECT_EQ(const_string, std::string(u32string_move_construct));

  ustring u32string_const_move_construct(std::move(u32_const_str));
  EXPECT_EQ(const_string, std::string(u32string_const_move_construct));
}

TEST(ustring, construct_and_convert) {
  convert_test("English Test");
  convert_test("Test de français");
  convert_test("中文测试");
  convert_test("日本語テスト");
  convert_test("🧐 Test");
}

TEST(ustring, operater) {
  ustring test("一些汉字");

  EXPECT_EQ(test.size(), 4);
  EXPECT_EQ(test[0], U'一');
  EXPECT_EQ(test[1], U'些');
  EXPECT_EQ(test[2], U'汉');
  EXPECT_EQ(test[3], U'字');

  EXPECT_EQ(test.find(U"一"), 0);

  EXPECT_EQ(test.find_first_not_of(U"一"), 1);
  EXPECT_EQ(test.find_last_not_of(U"一"), 3);

  test.append(U"用来测试");
  EXPECT_EQ(test.at(4), U'用');
}

