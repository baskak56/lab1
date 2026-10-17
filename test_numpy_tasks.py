import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)


def test_sum_prod():
    result = sum_prod(MatrixVectorBatchInput(
        [[[1, 0], [0, 2]], [[2, 0], [0, 2]]],
        [[1, 3], [1, 1]],
    ))
    np.testing.assert_allclose(result, np.array([[3], [8]]))


def test_sum_prod_empty():
    assert sum_prod(MatrixVectorBatchInput([], [])).shape == (0, 1)


def test_binarize():
    result = binarize(BinarizeInput(np.array([[0.1, 0.6], [0.5, 0.9]]), threshold=0.5))
    np.testing.assert_array_equal(result, [[0, 1], [0, 1]])


def test_binarize_no_mutation():
    m = np.array([[1, 2], [3, 4]])
    binarize(BinarizeInput(m, threshold=2))
    np.testing.assert_array_equal(m, [[1, 2], [3, 4]])


def test_unique_rows_columns():
    m = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_rows(MatrixInput(m)) == [[1, 2], [3]]
    assert unique_columns(MatrixInput(m)) == [[1, 3], [1, 3], [2, 3]]


def test_matrix_statistics():
    s = matrix_statistics(RandomMatrixInput(rows=3, columns=4, seed=1))
    assert s.matrix.shape == (3, 4)
    assert s.row_means.shape == (3,)
    assert s.column_means.shape == (4,)
    assert s.row_variances.shape == (3,)
    assert s.column_variances.shape == (4,)


def test_chess():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    np.testing.assert_array_equal(result, [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])


def test_draw_rectangle():
    img = draw_rectangle(RectangleInput(
        width=2, height=2,
        image_height=4, image_width=4,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    ))
    assert img.shape == (4, 4, 3)
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_draw_ellipse():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2,
        image_height=5, image_width=5,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0),
    ))
    assert img.shape == (5, 5, 3)
    assert tuple(img[2, 2]) == (255, 255, 255)
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_draw_ellipse_zero_axis():
    img = draw_ellipse(EllipseInput(
        semi_axis_x=0, semi_axis_y=2,
        image_height=4, image_width=4,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0),
    ))
    assert np.all(img == 0)


def test_analyze_time_series():
    s = analyze_time_series(TimeSeriesInput([1, 3, 2, 5, 4], window=2))
    assert s.mean == pytest.approx(3.0)
    np.testing.assert_array_equal(s.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(s.local_minima_indices, [2])
    np.testing.assert_allclose(s.moving_average, [2.0, 2.5, 3.5, 4.5])


def test_one_hot():
    result = one_hot(OneHotInput([0, 2, 3, 0]))
    np.testing.assert_array_equal(result, [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])


def test_one_hot_class_count_too_small():
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, 3], class_count=2))