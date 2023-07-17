import math
import turtle


WIDTH = 1280
HEIGHT = 800


def init() -> None:
    turtle.setup(WIDTH, HEIGHT)
    turtle.speed(0)
    turtle.colormode(1.0)
    turtle.pencolor(1.0, 1.0, 1.0)
    turtle.bgcolor(0.0, 0.0, 0.0)
    turtle.pensize(1)
    turtle.penup()
    turtle.hideturtle()
    turtle.tracer(0, 0)


def clifford_attractor(
    a: float, b: float, c: float, d: float, iterations: int, scale: int
) -> None:
    x = y = 0.0

    for i in range(iterations):
        part = i / iterations
        turtle.pencolor(part * part, part, part * part * part)
        size = 10 - 9 * part
        turtle.pensize(round(size))

        new_x = math.sin(a * y) + c * math.cos(a * x)
        new_y = math.sin(b * x) + d * math.cos(b * y)

        x = new_x
        y = new_y

        turtle.goto(x * scale, y * scale)
        turtle.pendown()
        turtle.forward(0)
        turtle.penup()

        if i % 500 == 0 or (i + 1) == iterations:
            turtle.update()
    turtle.done()


def main() -> None:
    init()
    clifford_attractor(
        a=1.7,
        b=1.7,
        c=0.6,
        d=1.2,
        iterations=50000,
        scale=166,
    )


if __name__ == "__main__":
    main()
