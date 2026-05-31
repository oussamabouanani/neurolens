from PIL import Image

from neurolens.vlm import image_grid


def test_image_grid_places_images_in_row_major_order():
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]
    images = [Image.new("RGB", (4, 4), color) for color in colors]

    grid = image_grid(images=images, size=4, rows=2)

    assert grid.size == (4, 4)
    assert grid.getpixel((0, 0)) == colors[0]
    assert grid.getpixel((2, 0)) == colors[1]
    assert grid.getpixel((0, 2)) == colors[2]
    assert grid.getpixel((2, 2)) == colors[3]


def test_image_grid_handles_partial_final_row():
    images = [
        Image.new("RGB", (4, 4), (255, 0, 0)),
        Image.new("RGB", (4, 4), (0, 255, 0)),
        Image.new("RGB", (4, 4), (0, 0, 255)),
    ]

    grid = image_grid(images=images, size=4, rows=2)

    assert grid.size == (4, 4)
    assert grid.getpixel((0, 0)) == (255, 0, 0)
    assert grid.getpixel((2, 0)) == (0, 255, 0)
    assert grid.getpixel((0, 2)) == (0, 0, 255)
