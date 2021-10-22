import unicodedata


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_upper_case(char):
    cp = ord(char)
    cat = unicodedata.category(char)
    if cat.startswith("Lu") and char.lower() != char:
        return True
    return False


def find_expect_char_in_range(judge_fun, start, end):
    result = []
    for c in range(start, end):
        if judge_fun(chr(c)):
            result.append(c)
    return result


def find_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def find_expect_category(category_func):
    expect_category_set = []

    # ASCII
    expect_category_set += find_expect_char_in_range(category_func, 0, 0x7F)

    # 	C1 Controls and Latin-1 Supplement
    expect_category_set += find_expect_char_in_range(category_func, 0x80, 0xFF)

    # Latin Extended-A
    expect_category_set += find_expect_char_in_range(category_func, 0x100, 0x17F)

    # Latin Extended-B
    expect_category_set += find_expect_char_in_range(category_func, 0x180, 0x24F)

    # IPA Extensions
    expect_category_set += find_expect_char_in_range(category_func, 0x250, 0x2AF)

    # Spacing Modifier Letters
    expect_category_set += find_expect_char_in_range(category_func, 0x2B0, 0x2FF)

    # Combining Diacritical Marks
    expect_category_set += find_expect_char_in_range(category_func, 0x300, 0x36F)

    # Greek/Coptic
    expect_category_set += find_expect_char_in_range(category_func, 0x370, 0x3FF)

    # Cyrillic and Cyrillic Supplement
    expect_category_set += find_expect_char_in_range(category_func, 0x400, 0x52F)

    # General Punctuation
    expect_category_set += find_expect_char_in_range(category_func, 0x2000, 0x206F)

    # CJK Radicals Supplement
    expect_category_set += find_expect_char_in_range(category_func, 0x2E80, 0x2EFF)

    # CJK Symbols and Punctuation
    expect_category_set += find_expect_char_in_range(category_func, 0x3000,	0x303F)

    # CJK
    expect_category_set += find_expect_char_in_range(category_func, 0x4E00,	0x9FFF)
    expect_category_set += find_expect_char_in_range(category_func, 0x3400,	0x4DBF)
    expect_category_set += find_expect_char_in_range(category_func, 0x20000, 0x2A6DF)
    expect_category_set += find_expect_char_in_range(category_func, 0x2A700, 0x2B73F)
    expect_category_set += find_expect_char_in_range(category_func, 0x2B740, 0x2CEAF)
    expect_category_set += find_expect_char_in_range(category_func, 0xF900, 0xFAFF)
    expect_category_set += find_expect_char_in_range(category_func, 0x2F800, 0x2FA1F)

    return find_ranges(expect_category_set)

def print_range(ranges):
    single_set = []
    pair_set = []
    for r in ranges:
        start, end = r
        if start == end:
            single_set.append(start)
        else:
            pair_set.append(r)

    output = "if ("
    for i in range(len(single_set)):
        if i != 0:
            output += "||"
        output += f"c == {single_set[i]}"
    output += ") {\n return true;\n}\n\n"

    output += "if ("
    for i in range(len(pair_set)):
        if i != 0:
            output += "||"
        start, end = pair_set[i]
        output += f"(c >= {start} && c <= {end})"
    output += ") {\n return true;\n}\n\nreturn false;\n"
    print(output)


print("\nis_whitespace:")
print_range(find_expect_category(_is_whitespace))

print("\nis_punctuation:")
print_range(find_expect_category(_is_punctuation))

print("\nis_control:")
print_range(find_expect_category(_is_control))


