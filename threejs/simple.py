import tortle
turtle = tortle

my_turtle = turtle.Turtle()
my_turtle.speed(2)
my_turtle.pendown()

for _ in range(4):
    my_turtle.forward(100)
    my_turtle.right(90)


def draw_triangle(color):
    my_turtle.pendown()
    my_turtle.fillcolor(color)
    my_turtle.begin_fill()
    for _ in range(3):
        my_turtle.forward(100)
        my_turtle.right(120)
    my_turtle.end_fill()
    my_turtle.penup()


my_turtle.penup()
my_turtle.forward(250)

for color in ("red", "blue", "purple"):
    draw_triangle(color)
    my_turtle.right(120)

turtle.done()
