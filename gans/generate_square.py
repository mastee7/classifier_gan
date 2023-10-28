import os
import random
from PIL import Image, ImageDraw

def generate_image(image_size, square_size, color, rotation):
    image = Image.new('RGB', image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = (image_size[0] - square_size[0]) // 2
    y = (image_size[1] - square_size[1]) // 2
    draw.rectangle([x, y, x + square_size[0], y + square_size[1]], fill=color)
    return image.rotate(rotation)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def main():
    dataset_size = 5000
    image_size = (64, 64)
    output_dir = '/home/wchung25/eee515/HW2/EEE515_HW2/gan/squares/1'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(dataset_size):
        side_length = random.randint(10, 50)
        square_size = (side_length, side_length)  # Ensuring both dimensions are the same
        color = random_color()
        rotation = random.randint(0, 360)
        image = generate_image(image_size, square_size, color, rotation)
        image.save(f"{output_dir}/image_{i:04d}.png")


if __name__ == "__main__":
    main()
