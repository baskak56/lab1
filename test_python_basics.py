import pytest

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello", 2),
        ("HELLO", 2),
        ("aeiou", 5),
        ("AEIOU", 5),
        ("", 0),
        ("xyz", 0),
        ("aAeEiIoOuU", 10),
    ],
)
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", True),
        ("a", True),
        ("abc", True),
        ("aa", False),
        ("abca", False),
        ("Aa", True),
    ],
)
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 1),
        (7, 3),
        (255, 8),
        (1024, 1),
    ],
)
def test_count_one_bits(number, expected):
    assert count_one_bits(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (4, 0),
        (9, 0),
        (39, 3),
        (999, 4),
        (10, 1),
        (25, 2),
    ],
)
def test_multiplicative_persistence(number, expected):
    assert multiplicative_persistence(PositiveIntegerInput(number)) == expected


def test_mse_basic():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0


def test_mse_nonzero():
    assert mse(VectorPairInput([0, 0], [1, 3])) == 5.0


def test_mse_empty():
    assert mse(VectorPairInput([], [])) == 0.0


def test_mse_length_mismatch():
    with pytest.raises(ValueError):
        mse(VectorPairInput([1, 2], [1]))


def test_mse_numpy_input():
    import numpy as np

    result = mse(VectorPairInput(np.array([1, 2]), np.array([1, 2])))
    assert result == 0.0


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, ""),
        (2, "(2)"),
        (4, "(2**2)"),
        (12, "(2**2)(3)"),
        (86240, "(2**5)(5)(7**2)(11)"),
        (97, "(97)"),
    ],
)
def test_prime_factorization(number, expected):
    assert prime_factorization(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, 1),
        (5, 2),
        (14, 3),
        (2, "It is impossible"),
        (4, "It is impossible"),
        (30, 4),
    ],
)
def test_pyramid(number, expected):
    assert pyramid(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, True),
        (12, True),
        (123, False),
        (1234, False),
        (1234006, True),
        (123456, False),
        (111, True),
    ],
)
def test_is_balanced_number(number, expected):
    assert is_balanced_number(PositiveIntegerInput(number)) is expected