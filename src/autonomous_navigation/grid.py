from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OccupancyGrid:
    """Carte d'occupation 2D : True = obstacle, False = libre."""

    data: NDArray[np.bool_]

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=bool)

        if self.data.ndim != 2:
            raise ValueError("La carte doit être une matrice 2D.")

        if self.data.size == 0:
            raise ValueError("La carte ne peut pas être vide.")

    @classmethod
    def random(
        cls,
        size: tuple[int, int] = (50, 50),
        obstacle_probability: float = 0.2,
        seed: int | None = None,
    ) -> "OccupancyGrid":
        """Génère une carte aléatoire reproductible si seed est défini."""

        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError("size doit contenir deux dimensions positives.")

        if not 0.0 <= obstacle_probability <= 1.0:
            raise ValueError("obstacle_probability doit être comprise entre 0 et 1.")

        rng = np.random.default_rng(seed)
        data = rng.random(size) < obstacle_probability

        return cls(data=data)

    @property
    def height(self) -> int:
        """Nombre de lignes de la carte."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Nombre de colonnes de la carte."""
        return self.data.shape[1]

    def in_bounds(self, x: int, y: int) -> bool:
        """Indique si la position est dans les limites de la carte."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        """Indique si une cellule est libre et dans les limites."""
        return self.in_bounds(x, y) and not self.data[y, x]

    def neighbors(
        self,
        x: int,
        y: int,
        diagonal: bool = True,
    ) -> Iterator[tuple[int, int]]:
        """Retourne les cellules libres voisines sans couper les coins."""

        if not self.is_free(x, y):
            return

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]

        if diagonal:
            directions += [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if not self.is_free(nx, ny):
                continue

            # Interdit le passage diagonal entre deux obstacles
            if dx != 0 and dy != 0:
                if not self.is_free(x + dx, y):
                    continue
                if not self.is_free(x, y + dy):
                    continue

            yield nx, ny


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid = OccupancyGrid.random(
        size=(50, 50),
        obstacle_probability=0.2,
        seed=42,
    )

    print(grid.width, grid.height)
    print(grid.is_free(10, 12))
    print(list(grid.neighbors(10, 12)))

    fig, ax = plt.subplots()
    mesh = ax.imshow(
        grid.data,
        cmap="gray_r",
        origin="upper",
        interpolation="nearest",
    )

    ax.set_title("Carte")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(mesh, ax=ax, label="Libre (0) / Obstacle (1)")
    plt.show()
