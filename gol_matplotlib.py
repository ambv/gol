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
        self.grid = np.zeros((self.rows, self.columns))

        g = self.grid

        # pulsar
        g[2, 4:7] = 1
        g[2, 10:13] = 1
        g[4:7, 2] = 1
        g[4:7, 7] = 1
        g[4:7, 9] = 1
        g[4:7, 14] = 1
        g[7, 4:7] = 1
        g[7, 10:13] = 1
        g[9, 4:7] = 1
        g[9, 10:13] = 1
        g[10:13, 2] = 1
        g[10:13, 7] = 1
        g[10:13, 9] = 1
        g[10:13, 14] = 1
        g[14, 4:7] = 1
        g[14, 10:13] = 1

        # pentadecathlon
        g[5, 24:27] = 1
        g[6, 24] = 1
        g[6, 26] = 1
        g[7:11, 24:27] = 1
        g[11, 24] = 1
        g[11, 26] = 1
        g[12, 24:27] = 1

        # LWSS
        g[20, 3:5] = 1
        g[21, 2:6] = 1
        g[22, 2:4] = 1
        g[22, 5:7] = 1
        g[23, 4:6] = 1

        # MWSS
        g[30, 13:16] = 1
        g[31, 12:17] = 1
        g[32, 12:15] = 1
        g[32, 16:18] = 1
        g[33, 15:17] = 1

        # HWSS
        g[40, 23:27] = 1
        g[41, 22:28] = 1
        g[42, 22:26] = 1
        g[42, 27:29] = 1
        g[43, 26:28] = 1

        # glider
        g[40, 6:8] = 1
        g[41, 7:9] = 1
        g[42, 6] = 1

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
        gol.update, # lambda x: [gol.image], # gol.update,
        frames=900,
        interval=100,
        blit=True,
    )
    anim.save("gol-figures.mp4", fps=10, dpi=150, extra_args=["-vcodec", "libx264"])
    # plt.show()


if __name__ == "__main__":
    main()
