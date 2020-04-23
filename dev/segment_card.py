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
    # logger.debug(f"Processing image at {path}")
    #
    # img_bgr = q.read_image(path)
    # img_lab = q.convert_colorspace(img_bgr, cv.COLOR_BGR2LAB)
    # img_b = img_lab[..., 2]  # just the blue-yellow channel
    #
    # thresh, thresholded = cv.threshold(
    #     cv.bitwise_not(img_b), 0, maxval=255, type=cv.THRESH_OTSU
    # )
    #
    # closed = cv.morphologyEx(
    #     thresholded,
    #     cv.MORPH_CLOSE,
    #     kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)),
    # )
    #
    # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
    #     closed, connectivity=8
    # )
    #
    # largest_label = max(
    #     range(1, num_labels), key=lambda label: stats[label, cv.CC_STAT_AREA]
    # )
    #
    # just_largest_component = np.where(labels == largest_label, closed, 0)
    #
    # edges = cv.Canny(
    #     cv.GaussianBlur(just_largest_component, (21, 21), 3),
    #     10,
    #     100,
    #     apertureSize=7,
    #     L2gradient=True,
    # )
    #
    # lines = cv.HoughLinesP(
    #     edges,
    #     rho=4,
    #     theta=np.deg2rad(1),
    #     threshold=100,
    #     minLineLength=200,
    #     maxLineGap=500,
    # ).squeeze()
    # lines = [q.Line((xs, ys), (xe, ye)) for (xs, ys, xe, ye) in lines[:4]]
    #
    # show_lines = img_bgr.copy()
    #
    # top, bottom, left, right = ordered = sorted(lines, key=lambda line: abs(line.slope))
    # horizontal = sorted(ordered[:2], key=lambda line: line.start_y)
    # vertical = sorted(ordered[2:], key=lambda line: line.start_x)
    #
    # for line in ordered:
    #     print(line)
    #
    # labels = [
    #     "T",
    #     "B",
    #     "L",
    #     "R",
    # ]
    # colors = [q.RED, q.MAGENTA, q.GREEN, q.WHITE]
    #
    # # for idx, (line, label, color) in enumerate(zip(itertools.chain(horizontal, vertical), labels, colors)):
    # #     show_lines = q.draw_line(
    # #         show_lines,
    # #         line.start,
    # #         line.end,
    # #         color = color,
    # #         thickness = 3,
    # #     )
    # #     show_lines = q.draw_text(
    # #         show_lines,
    # #         (line.start_x - 20, line.start_y - 20),
    # #         label,
    # #         color = color,
    # #         size = 5,
    # #         thickness = 3,
    # #     )
    #
    # intersections = []
    # for vline, hline in itertools.product(vertical, horizontal):
    #     print(vline.slope)
    #     if np.isinf(vline.slope):
    #         x = vline.start_x
    #     else:
    #         x = (vline.intercept - hline.intercept) / (hline.slope - vline.slope)
    #     y = hline(x)
    #     print(vline, hline)
    #     print(x, y)
    #     intersections.append((x, y))
    #
    # intersections = np.array(intersections)
    # print(intersections.shape)
    # center = np.mean(intersections, axis=0)
    # print(center)
    #
    # # show_lines = q.draw_circle(
    # #     show_lines,
    # #     center,
    # #     radius = 10,
    # #     thickness = -1,
    # #     color = q.GREEN,
    # # )
    #
    # rel_center = intersections - center
    # angles = np.arctan2(rel_center[:, 1], rel_center[:, 0])
    # print(angles)
    # intersections = [
    #     i
    #     for idx, i in sorted(
    #         enumerate(intersections), key=lambda idx_i: angles[idx_i[0]]
    #     )
    # ]

    img_bgr = q.read_image(path)
    intersections = q.find_card_corners(img_bgr)
    show_lines = img_bgr.copy()
    for idx, (x, y) in enumerate(intersections):
        show_lines = q.draw_circle(
            show_lines, (x, y), radius=10, thickness=-1, color=q.MAGENTA,
        )
        show_lines = q.draw_text(
            show_lines,
            (x - 20, y - 20),
            str(idx),
            color=q.MAGENTA,
            size=5,
            thickness=3,
        )

    q.write_image(show_lines, OUT / f"{path.stem}_card.jpg")


if __name__ == "__main__":
    image_paths = [
        path for path in (DATA / "aus").iterdir() if path.suffix.lower() == ".jpg"
    ]

    for path in image_paths:
        process(path)
