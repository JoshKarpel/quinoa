import logging

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

    corners = q.find_card_corners(image_bgr=img_bgr)

    new_corners = q.determine_new_corners(corners)

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

    crop_slice = q.corners_to_slice(new_corners)

    cropped = transformed[crop_slice]

    q.write_image(cropped, OUT / f"{path.stem}_2.jpg")


if __name__ == "__main__":
    image_paths = [
        path for path in (DATA / "aus").iterdir() if path.suffix.lower() == ".jpg"
    ]
    # image_paths = [DATA / 'aus' / '475.JPG']

    for path in tqdm(image_paths):
        process(path)
