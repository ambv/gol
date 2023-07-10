import turtle
import numpy as np


def draw_cell(x, y, size):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.setheading(0)
    turtle.fillcolor('black')
    turtle.begin_fill()
    for _ in range(4):
        turtle.forward(size)
        turtle.right(90)
    turtle.end_fill()


def animate_life(grid: np.array, size: int) -> None:
    rows, columns = grid.shape
    width = columns * size
    height = rows * size

    turtle.setup(width + 50, height + 50)
    turtle.screensize(width, height)
    turtle.speed(0)
    turtle.tracer(0)

    for i in range(rows):
        for j in range(columns):
            if grid[i, j] == 1:
                x = j * size - width / 2
                y = height / 2 - i * size
                draw_cell(x, y, size)

    while True:
        new_grid = np.copy(grid)
        for i in range(rows):
            for j in range(columns):
                neighbors = np.sum(grid[max(i - 1, 0):min(i + 2, rows), max(j - 1, 0):min(j + 2, columns)]) - grid[i, j]
                if grid[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                    new_grid[i, j] = 0
                elif grid[i, j] == 0 and neighbors == 3:
                    new_grid[i, j] = 1
        grid = new_grid

        turtle.clear()
        for i in range(rows):
            for j in range(columns):
                if grid[i, j] == 1:
                    x = j * size - width / 2
                    y = height / 2 - i * size
                    draw_cell(x, y, size)

        turtle.update()


# Set up the initial grid
rows, columns = 80, 80
grid = np.random.choice([0, 1], size=(rows, columns), p=[0.5, 0.5])

# Set up the turtle screen and animation
animate_life(grid, 10)