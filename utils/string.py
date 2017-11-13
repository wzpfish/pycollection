# coding: utf-8


def is_chinese_word(word):
    return "\u4e00" <= word <= "\u9fff"


def chinese_word_count(text):
    return sum([1 for word in text if is_chinese_word(word)])


def has_black_words(text, black_words):
    for black_word in black_words:
        if black_word in text:
            return True
    return False


def is_punctuation_only(text):
    for word in text:
        if is_chinese_word(word) or word.isalnum():
            return False
    return True