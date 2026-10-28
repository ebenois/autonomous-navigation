import numpy as np
import pytest

from autonomous_navigation.grid import OccupancyGrid


def test_grid_dimensions() -> None:
    grid = OccupancyGrid(np.zeros((3, 5), dtype=bool))

    assert grid.height == 3
    assert grid.width == 5


def test_grid_rejects_non_2d_array() -> None:
    with pytest.raises(ValueError, match="2D"):
        OccupancyGrid(np.zeros(5, dtype=bool))


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((0, 3), dtype=bool),
        np.zeros((3, 0), dtype=bool),
    ],
)
def test_grid_rejects_empty_array(data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="vide"):
        OccupancyGrid(data)


def test_grid_converts_data_to_boolean() -> None:
    grid = OccupancyGrid(np.array([[0, 1], [1, 0]]))

    assert grid.data.dtype == bool
    assert grid.data.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 0, True),
        (4, 2, True),
        (5, 2, False),
        (0, 3, False),
        (-1, 0, False),
        (0, -1, False),
    ],
)
def test_in_bounds(
    x: int,
    y: int,
    expected: bool,
) -> None:
    grid = OccupancyGrid(np.zeros((3, 5), dtype=bool))

    assert grid.in_bounds(x, y) is expected


def test_is_free_distinguishes_free_and_occupied_cells() -> None:
    grid = OccupancyGrid(
        np.array(
            [
                [False, True],
                [False, False],
            ]
        )
    )

    assert grid.is_free(0, 0)
    assert not grid.is_free(1, 0)
    assert not grid.is_free(2, 0)
    assert not grid.is_free(-1, 0)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_random_rejects_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="obstacle_probability"):
        OccupancyGrid.random(obstacle_probability=probability)


@pytest.mark.parametrize(
    "size",
    [(0, 5), (5, 0), (-1, 5)],
)
def test_random_rejects_invalid_dimensions(
    size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="size"):
        OccupancyGrid.random(size=size)


def test_random_rejects_size_with_three_dimensions() -> None:
    with pytest.raises(ValueError, match="size"):
        OccupancyGrid.random(size=(2, 3, 4))


def test_random_is_reproducible_with_seed() -> None:
    grid_a = OccupancyGrid.random(size=(10, 12), seed=42)
    grid_b = OccupancyGrid.random(size=(10, 12), seed=42)

    np.testing.assert_array_equal(grid_a.data, grid_b.data)


def test_random_respects_obstacle_probability_extremes() -> None:
    empty_grid = OccupancyGrid.random(
        size=(5, 5),
        obstacle_probability=0.0,
        seed=42,
    )
    full_grid = OccupancyGrid.random(
        size=(5, 5),
        obstacle_probability=1.0,
        seed=42,
    )

    assert not empty_grid.data.any()
    assert full_grid.data.all()


def test_neighbors_without_diagonals() -> None:
    grid = OccupancyGrid(np.zeros((3, 3), dtype=bool))

    assert set(grid.neighbors(1, 1, diagonal=False)) == {
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    }


def test_neighbors_with_diagonals() -> None:
    grid = OccupancyGrid(np.zeros((3, 3), dtype=bool))

    assert len(list(grid.neighbors(1, 1, diagonal=True))) == 8


def test_neighbors_excludes_obstacles() -> None:
    grid = OccupancyGrid(
        np.array(
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        )
    )

    neighbors = set(grid.neighbors(1, 1, diagonal=False))

    assert (1, 0) not in neighbors
    assert (0, 1) in neighbors
    assert (2, 1) in neighbors


@pytest.mark.parametrize(
    ("data", "start", "forbidden"),
    [
        (
            [
                [False, True],
                [True, False],
            ],
            (0, 0),
            (1, 1),
        ),
        (
            [
                [False, False],
                [True, False],
            ],
            (0, 0),
            (1, 1),
        ),
    ],
)
def test_neighbors_does_not_cut_diagonal_corners(
    data: list[list[bool]],
    start: tuple[int, int],
    forbidden: tuple[int, int],
) -> None:
    grid = OccupancyGrid(np.array(data))

    assert forbidden not in list(grid.neighbors(*start, diagonal=True))


def test_neighbors_allows_diagonal_when_adjacent_cells_are_free() -> None:
    grid = OccupancyGrid(np.zeros((2, 2), dtype=bool))

    assert (1, 1) in set(grid.neighbors(0, 0, diagonal=True))


def test_neighbors_at_grid_edge() -> None:
    grid = OccupancyGrid(np.zeros((3, 3), dtype=bool))

    neighbors = set(grid.neighbors(0, 0))

    assert neighbors == {(1, 0), (0, 1), (1, 1)}


def test_neighbors_from_occupied_cell_is_empty() -> None:
    grid = OccupancyGrid(
        np.array(
            [
                [True, False],
                [False, False],
            ]
        )
    )

    assert list(grid.neighbors(0, 0)) == []
