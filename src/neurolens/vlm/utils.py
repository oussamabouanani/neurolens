import math

from PIL import Image


def image_grid(
    *,
    images: list[Image.Image],
    size: int,
    rows: int,
) -> Image.Image:

    count = len(images)

    columns = math.ceil(count / rows)

    cell_w = size // columns
    cell_h = size // rows

    grid = Image.new("RGB", (size, size))

    for i, img in enumerate(images):
        row, col = divmod(i, columns)

        x = col * cell_w
        y = row * cell_h

        img = img.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        grid.paste(img, (x, y))

    return grid
