from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@dataclass
class GameOfLife:
    rows: int
    columns: int
    fig: Figure = field(init=False)
    ax: Axes = field(init=False)
    grid: npt.NDArray[np.int64] = field(init=False)
    image: AxesImage = field(init=False)

    def __post_init__(self):
        self.fig, self.ax = plt.subplots()
        self.grid = np.random.choice(
            [0, 1],
            size=(self.rows, self.columns),
            p=[0.5, 0.5],
        )
        self.image = self.ax.imshow(
            self.grid,
            interpolation="nearest",
            cmap="binary",
        )

    def update(self, frame: int) -> tuple[AxesImage]:
        old = self.grid
        new = np.copy(old)
        r = self.rows
        c = self.columns
        for i in range(r):
            for j in range(c):
                neighbors = (
                    old[(i - 1) % r, (j - 1) % c]
                    + old[(i - 1) % r, j]
                    + old[(i - 1) % r, (j + 1) % c]
                    + old[i, (j - 1) % c]
                    + old[i, (j + 1) % c]
                    + old[(i + 1) % r, (j - 1) % c]
                    + old[(i + 1) % r, j]
                    + old[(i + 1) % r, (j + 1) % c]
                )

                if old[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                    new[i, j] = 0
                elif old[i, j] == 0 and neighbors == 3:
                    new[i, j] = 1

        self.grid = new
        self.image.set_data(new)
        return (self.image,)


def main():
    gol = GameOfLife(rows=50, columns=50)
    anim = animation.FuncAnimation(
        gol.fig,
        gol.update,
        frames=900,
        interval=100,
        blit=True,
    )
    anim.save("gol.mp4", fps=30, dpi=150, extra_args=["-vcodec", "libx264"])


if __name__ == "__main__":
    main()
