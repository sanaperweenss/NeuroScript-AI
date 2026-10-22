"""
Unit Tests - Label Mappings and Class Conventions
Tests verified class counts, bidirectional mappings, and EMNIST 1-indexed conversion.
"""

import pytest
from data.labels import (
    LabelMapper, get_label_mapper,
    DIGIT_CLASSES, ALPHABET_CLASSES, COMBINED_CLASSES
)
from config.settings import TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED


def test_digit_label_mapping():
    mapper = get_label_mapper(TASK_DIGITS)
    assert mapper.num_classes == 10
    assert mapper.get_classes() == [str(i) for i in range(10)]
    assert mapper.index_to_label(0) == "0"
    assert mapper.index_to_label(9) == "9"
    assert mapper.label_to_index("5") == 5


def test_alphabet_label_mapping():
    mapper = get_label_mapper(TASK_ALPHABET)
    assert mapper.num_classes == 26
    assert mapper.get_classes()[0] == "A"
    assert mapper.get_classes()[25] == "Z"
    assert mapper.index_to_label(0) == "A"
    assert mapper.index_to_label(25) == "Z"
    assert mapper.label_to_index("A") == 0
    assert mapper.label_to_index("Z") == 25
    assert mapper.label_to_index("m") == 12  # Case insensitive


def test_combined_label_mapping():
    mapper = get_label_mapper(TASK_COMBINED)
    assert mapper.num_classes == 36
    # 0-9 are digits
    assert mapper.index_to_label(0) == "0"
    assert mapper.index_to_label(9) == "9"
    # 10-35 are letters
    assert mapper.index_to_label(10) == "A"
    assert mapper.index_to_label(35) == "Z"
    assert mapper.label_to_index("0") == 0
    assert mapper.label_to_index("A") == 10


def test_emnist_raw_label_normalization():
    # NIST 1-indexed raw labels: 1=A -> 0, 26=Z -> 25
    assert LabelMapper.normalize_emnist_raw_label(1, is_one_indexed=True) == 0
    assert LabelMapper.normalize_emnist_raw_label(26, is_one_indexed=True) == 25
    # If already 0-indexed:
    assert LabelMapper.normalize_emnist_raw_label(0, is_one_indexed=False) == 0
    assert LabelMapper.normalize_emnist_raw_label(25, is_one_indexed=False) == 25
    # Out of range
    with pytest.raises(ValueError):
        LabelMapper.normalize_emnist_raw_label(27, is_one_indexed=True)


def test_invalid_task():
    with pytest.raises(ValueError):
        LabelMapper("unsupported_task")


def test_invalid_label_or_index():
    mapper = get_label_mapper(TASK_DIGITS)
    with pytest.raises(KeyError):
        mapper.index_to_label(10)
    with pytest.raises(KeyError):
        mapper.label_to_index("Z")
