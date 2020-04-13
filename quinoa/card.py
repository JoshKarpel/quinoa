import itertools

import cv2 as cv
import numpy as np

from .colors import convert_colorspace


def find_card_corners(image_bgr):
    # we'll work in LAB colorspace because the card is very blue, and can be
    # isolated in the B (blue-yellow) channel
    img_lab = convert_colorspace(image_bgr, cv.COLOR_BGR2LAB)
    img_b = img_lab[..., 2]  # just the blue-yellow channel

    # threshold and close the blue-yellow channel to make blue blobs stand out
    thresh, thresholded = cv.threshold(
        cv.bitwise_not(img_b), 0, maxval=255, type=cv.THRESH_OTSU
    )
    closed = cv.morphologyEx(
        thresholded,
        cv.MORPH_CLOSE,
        kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)),
    )

    # select just the largest connected object in the closed image, which
    # should be the card
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        closed, connectivity=8
    )
    largest_label = max(
        range(1, num_labels), key=lambda label: stats[label, cv.CC_STAT_AREA]
    )
    just_largest_component = np.where(labels == largest_label, closed, 0)

    # blur the closed image, then find edges on it
    edges = cv.Canny(
        cv.GaussianBlur(just_largest_component, (21, 21), 3),
        10,
        100,
        apertureSize=7,
        L2gradient=True,
    )

    # find lines in the edge image
    # choose the 4 most-voted lines as the edges of the card
    lines = cv.HoughLinesP(
        edges,
        rho=4,
        theta=np.deg2rad(1),
        threshold=100,
        minLineLength=200,
        maxLineGap=500,
    ).squeeze()
    lines = [Line((xs, ys), (xe, ye)) for (xs, ys, xe, ye) in lines[:4]]

    # order the lines by slope to find the horizontal and vertical lines
    ordered = sorted(lines, key=lambda line: abs(line.slope))
    horizontal = sorted(ordered[:2], key=lambda line: line.start_y)
    vertical = sorted(ordered[2:], key=lambda line: line.start_x)

    # find intersections between the horizontal and vertical lines
    intersections = []
    for vline, hline in itertools.product(vertical, horizontal):
        if np.isinf(vline.slope):
            x = vline.start_x
        else:
            x = (vline.intercept - hline.intercept) / (hline.slope - vline.slope)
        y = hline(x)
        intersections.append((x, y))

    intersections = np.array(intersections)
    center = np.mean(intersections, axis=0)
    rel_center = intersections - center
    angles = np.arctan2(rel_center[:, 1], rel_center[:, 0])

    intersections = [
        i
        for idx, i in sorted(
            enumerate(intersections), key=lambda idx_i: angles[idx_i[0]]
        )
    ]

    return intersections


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

        self.slope = (self.end[-1] - self.start[-1]) / (self.end[0] - self.start[0])
        self.intercept = self.start[1] - (self.slope * self.start[0])

    @property
    def start_x(self):
        return self.start[0]

    @property
    def start_y(self):
        return self.start[1]

    @property
    def end_x(self):
        return self.end[0]

    @property
    def end_y(self):
        return self.end[1]

    def __repr__(self):
        return f"{self.__class__.__name__}(start = {self.start}, end = {self.end}, slope = {self.slope}, intercept = {self.intercept})"

    def __call__(self, x):
        return (self.slope * x) + self.intercept
