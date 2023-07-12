import random
import numpy as np
from PIL import Image, ImageDraw

def reduce_image_to_collage(image_path, num_triangles=500):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Quantize the image to a limited number of colors
    quantized_image = image.quantize(256)

    # Get the representative colors and their frequencies
    colors = quantized_image.getpalette()
    color_frequencies = quantized_image.histogram()

    # Normalize the color frequencies
    total_frequency = sum(color_frequencies)
    normalized_frequencies = [freq / total_frequency for freq in color_frequencies]

    # Create a blank canvas
    canvas = Image.new('RGB', (width, height), (255, 255, 255))

    # Draw triangles on the canvas with representative colors
    draw = ImageDraw.Draw(canvas)
    for i in range(num_triangles):
        # Choose a color based on its normalized frequency
        color_index = np.random.choice(range(256), p=normalized_frequencies)
        color = tuple(colors[color_index * 3:color_index * 3 + 3])

        # Choose triangle vertices based on color region
        region_indices = np.where(quantized_image == color_index)
        x = [random.choice(region_indices[0]) for _ in range(3)]
        y = [random.choice(region_indices[1]) for _ in range(3)]
        triangle = list(zip(x, y))

        # Draw the triangle with the chosen color
        draw.polygon(triangle, fill=color)

    # Save the final image
    output_path = 'output_collage.jpg'
    canvas.save(output_path)
    print(f"Reduced image collage saved to {output_path}")

# Example usage
reduce_image_to_collage('input.jpg')
