import math
import turtle


WIDTH = 1280
HEIGHT = 800


def init() -> None:
    turtle.setup(WIDTH, HEIGHT)
    # turtle.setworldcoordinates(0, 0, WIDTH, HEIGHT)
    turtle.speed(0)
    turtle.colormode(1.0)
    turtle.bgcolor("black")
    turtle.pensize(1)
    turtle.penup()
    turtle.hideturtle()
    turtle.tracer(0, 0)


def clifford_attractor(
    a: float, b: float, c: float, d: float, iterations: int, scale: int
) -> None:
    x = 0.0
    y = 0.0

    for i in range(iterations):
        part = (i + 1) / iterations
        size = 10 - 9 * part
        turtle.pensize(round(size))
        turtle.pencolor(part * part, part, part * part * part)
        new_x = math.sin(a * y) + c * math.cos(a * x)
        new_y = math.sin(b * x) + d * math.cos(b * y)

        x = new_x
        y = new_y

        scaled_x = x * scale
        scaled_y = y * scale

        turtle.goto(scaled_x, scaled_y)
        turtle.pendown()
        turtle.forward(0)
        turtle.penup()

        if i % 500 == 0:
            turtle.update()
    
    print("Done.")


def main() -> None:
    init()
    # Draw the Clifford attractor
    clifford_attractor(
        a=-1.4,
        b=1.7,
        c=1,
        d=0.7,
        iterations=50000,
        scale=220,
    )
    turtle.update()
    turtle.done()


if __name__ == "__main__":
    main()
