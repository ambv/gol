import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update(frame):
    global grid
    new_grid = np.copy(grid)
    for i in range(rows):
        for j in range(columns):
            # Count the number of live neighbors
            neighbors = (
                grid[(i - 1) % rows, (j - 1) % columns]
                + grid[(i - 1) % rows, j]
                + grid[(i - 1) % rows, (j + 1) % columns]
                + grid[i, (j - 1) % columns]
                + grid[i, (j + 1) % columns]
                + grid[(i + 1) % rows, (j - 1) % columns]
                + grid[(i + 1) % rows, j]
                + grid[(i + 1) % rows, (j + 1) % columns]
            )

            # Apply the rules of Conway's Game of Life
            if grid[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                new_grid[i, j] = 0
            elif grid[i, j] == 0 and neighbors == 3:
                new_grid[i, j] = 1

    grid = new_grid
    img.set_array(grid)
    return [img]


# Set up the initial grid
rows, columns = 50, 50
grid = np.random.choice([0, 1], size=(rows, columns), p=[0.5, 0.5])

# Set up the figure and axis
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation="nearest", cmap="binary")

# Create the animation
ani = animation.FuncAnimation(fig, update, interval=20, blit=True)

# Show the animation
plt.show()
