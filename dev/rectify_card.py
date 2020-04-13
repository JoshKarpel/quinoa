import logging

logging.basicConfig()

from pathlib import Path
from pprint import pprint
import itertools

from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2 as cv

import quinoa as q

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent.absolute()
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem


def process(path):
    img_bgr = q.read_image(path)

    corners = q.find_card_corners(img_bgr)

    side_lengths = [np.linalg.norm(s - e) for s, e in q.window(corners + [corners[0]])]
    print(side_lengths)
    target_side_length = max(side_lengths)
    print(target_side_length)

    tl_corner_x, tl_corner_y = corners[0]

    new_corners = np.array(
        [
            corners[0],
            [tl_corner_x + target_side_length, tl_corner_y],
            [tl_corner_x + target_side_length, tl_corner_y + target_side_length],
            [tl_corner_x, tl_corner_y + target_side_length],
        ]
    )

    show_original = img_bgr.copy()

    for idx, (x, y) in enumerate(corners):
        show_original = q.draw_circle(
            show_original, (x, y), radius=10, thickness=-1, color=q.MAGENTA,
        )
        show_original = q.draw_text(
            show_original,
            (x - 20, y - 20),
            str(idx),
            color=q.MAGENTA,
            size=5,
            thickness=3,
        )

    for idx, (x, y) in enumerate(new_corners):
        show_original = q.draw_circle(
            show_original, (x, y), radius=10, thickness=-1, color=q.GREEN,
        )
        show_original = q.draw_text(
            show_original,
            (x - 20, y - 20),
            str(idx),
            color=q.GREEN,
            size=5,
            thickness=3,
        )

    q.write_image(show_original, OUT / f"{path.stem}_0.jpg")

    transform_matrix = cv.getPerspectiveTransform(
        corners.astype(np.float32), new_corners.astype(np.float32)
    )
    transformed = cv.warpPerspective(
        img_bgr, transform_matrix, (img_bgr.shape[1], img_bgr.shape[0]),
    )

    show_transformed = transformed.copy()

    for idx, (x, y) in enumerate(corners):
        show_transformed = q.draw_circle(
            show_transformed, (x, y), radius=10, thickness=-1, color=q.MAGENTA,
        )
        show_transformed = q.draw_text(
            show_transformed,
            (x - 20, y - 20),
            str(idx),
            color=q.MAGENTA,
            size=5,
            thickness=3,
        )

    for idx, (x, y) in enumerate(new_corners):
        show_transformed = q.draw_circle(
            show_transformed, (x, y), radius=10, thickness=-1, color=q.GREEN,
        )
        show_transformed = q.draw_text(
            show_transformed,
            (x - 20, y - 20),
            str(idx),
            color=q.GREEN,
            size=5,
            thickness=3,
        )

    q.write_image(show_transformed, OUT / f"{path.stem}_1.jpg")

    min_x = int(np.floor(np.min(new_corners[:, 0])))
    max_x = int(np.ceil(np.max(new_corners[:, 0])))
    min_y = int(np.floor(np.min(new_corners[:, 1])))
    max_y = int(np.ceil(np.max(new_corners[:, 1])))

    cropped = transformed[min_y : max_y + 1, min_x : max_x + 1]

    q.write_image(cropped, OUT / f"{path.stem}_2.jpg")


if __name__ == "__main__":
    image_paths = [path for path in DATA.iterdir() if path.suffix.lower() == ".jpg"]

    for path in image_paths:
        process(path)
