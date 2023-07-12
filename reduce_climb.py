import random
import numpy as np
from PIL import Image, ImageDraw


def calculate_difference(image1: Image.Image, image2: Image.Image):
    """Calculate the difference between two images."""
    diff = np.array(image1) - np.array(image2)
    squared_diff = np.square(diff)
    return np.sum(squared_diff)


def generate_candidate_triangle(max_x: int, max_y: int) -> list[tuple[int, int]]:
    """Generate a random candidate triangle."""
    x = [random.randint(0, max_x) for _ in range(3)]
    y = [random.randint(0, max_y) for _ in range(3)]
    return list(zip(x, y))


def calculate_candidate_color(input_image, candidate_triangle):
    # Calculate the bounding box of the candidate triangle
    min_x = min([x for x, _ in candidate_triangle])
    min_y = min([y for _, y in candidate_triangle])
    max_x = max([x for x, _ in candidate_triangle])
    max_y = max([y for _, y in candidate_triangle])

    # Crop the input image to the bounding box
    cropped_region = input_image.crop((min_x, min_y, max_x, max_y))

    # Calculate the average color of the cropped region
    return tuple(np.array(cropped_region).mean(axis=(0, 1)).astype(int))


def calculate_triangle_area(triangle):
    """Calculate the area of a triangle."""
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)


def choose_next_triangle(
    input_image: Image.Image, output_image: Image.Image, attempts: int = 1000
) -> None:
    width, height = input_image.size

    # Calculate the difference between the input and the output image
    smallest_difference = calculate_difference(input_image, output_image)

    iterations = 0
    while smallest_difference > 0 and iterations < attempts:
        iterations += 1

        # Generate a candidate triangle
        candidate_triangle = generate_candidate_triangle(width, height)
        candidate_color = calculate_candidate_color(input_image, candidate_triangle)

        # Create a copy of the output image
        new_output_image = output_image.copy()

        # Draw the candidate triangle on the copied image with the average color
        draw = ImageDraw.Draw(new_output_image)
        draw.polygon(candidate_triangle, fill=candidate_color)

        # Calculate the difference between the input and the new output image
        new_difference = calculate_difference(input_image, new_output_image)

        # If the new image is an improvement, replace the output image
        if new_difference < smallest_difference:
            smallest_difference = new_difference
            best_triangle = candidate_triangle
            best_color = candidate_color
            print(
                f"  Difference improved to {smallest_difference}"
                f" at iteration {iterations}"
            )
            return new_output_image

    raise LookupError(f"Couldn't improve the score in {attempts} iterations")


def reduce_image_to_output(input_path, output_path, num_triangles=500):
    # Load the input image
    input_image = Image.open(input_path)
    width, height = input_image.size

    # Create the output image with the average color of the input as the background
    average_color = input_image.resize((1, 1), resample=Image.LANCZOS).getpixel((0, 0))
    output_image = Image.new("RGB", (width, height), average_color)

    candidate_images = [output_image]
    # Hill-climbing algorithm to improve the output image
    while len(candidate_images) < num_triangles:
        print(f"{len(candidate_images)} triangles so far...")
        try:
            candidate_images.append(
                choose_next_triangle(input_image, candidate_images[-1])
            )
        except LookupError:
            if len(candidate_images) > 1:
                candidate_images.pop()

    # Save the final output image
    output_image = candidate_images[-1]
    output_image.save(output_path)
    print(f"Output image saved to {output_path}")


if __name__ == "__main__":
    reduce_image_to_output("input.jpg", "output_image.jpg")
