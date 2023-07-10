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


def clifford_attractor(a: float, b: float, c: float, d: float, iterations: int, scale: int) -> None:
    x = 0.0
    y = 0.0
    
    for i in range(iterations):
        part = i / iterations
        turtle.pencolor(part, part, part)
        new_x = math.sin(a * y) + c * math.cos(a * x)
        new_y = math.sin(b * x) + d * math.cos(b * y)
        
        x = new_x
        y = new_y
        
        scaled_x = x * scale
        scaled_y = y * scale
        
        turtle.goto(scaled_x, scaled_y)
        turtle.dot()

        if i % 500 == 0:
            turtle.update()
    
    canv = turtle.getcanvas()
    # breakpoint()
    print()


def main() -> None:
    init()
    # Draw the Clifford attractor
    clifford_attractor(
        a=1.6,
        b=1.7,
        c=1.7,
        d=0.7,
        iterations=50000,
        scale=200,
    )
    turtle.update()
    turtle.done()


if __name__ == "__main__":
    main()