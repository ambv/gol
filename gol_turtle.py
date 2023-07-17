from dataclasses import dataclass, field

import turtle
import numpy as np
import numpy.typing as npt


@dataclass
class GameOfLife:
    rows: int
    columns: int
    size: int
    width: int = field(init=False)
    height: int = field(init=False)
    grid: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        self.grid = np.random.choice(
            [0, 1],
            size=(self.rows, self.columns),
            p=[0.5, 0.5],
        )
        self.width = self.columns * self.size
        self.height = self.rows * self.size
        self.setup_turtle()
        self.draw_board()

    def setup_turtle(self) -> None:
        turtle.setup(self.width + 50, self.height + 50)
        turtle.screensize(self.width, self.height)
        turtle.speed(0)
        turtle.tracer(0)

    def draw_board(self) -> None:
        g = self.grid
        turtle.clear()
        for i in range(self.rows):
            for j in range(self.columns):
                if g[i, j] == 1:
                    self.draw_cell(i, j)
        turtle.update()

    def draw_cell(self, row: int, column: int) -> None:
        x = column * self.size - self.width / 2
        y = self.height / 2 - row * self.size

        turtle.penup()
        turtle.goto(x, y)
        turtle.pendown()
        turtle.setheading(0)
        turtle.fillcolor("black")
        turtle.begin_fill()
        for _ in range(4):
            turtle.forward(self.size)
            turtle.right(90)
        turtle.end_fill()

    def update_board(self) -> None:
        new_grid = np.copy(self.grid)
        for i in range(self.rows):
            for j in range(self.columns):
                new_grid[i, j] = self.update_cell(i, j)
        self.grid = new_grid

    def update_cell(self, row: int, column: int) -> int:
        neighbors = (
            np.sum(
                self.grid[
                    max(row - 1, 0) : min(row + 2, self.rows),
                    max(column - 1, 0) : min(column + 2, self.columns),
                ]
            )
            - self.grid[row, column]
        )
        if self.grid[row, column] == 1 and (neighbors < 2 or neighbors > 3):
            return 0
        elif self.grid[row, column] == 0 and neighbors == 3:
            return 1
        return self.grid[row, column]

    def animate(self) -> None:
        while True:
            self.update_board()
            self.draw_board()


def main() -> None:
    gol = GameOfLife(rows=80, columns=80, size=10)
    gol.animate()


if __name__ == "__main__":
    main()
