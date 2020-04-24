import itertools

import cv2 as cv
import numpy as np
import sklearn.mixture

from . import colors, utils


def find_card_corners(image_bgr=None, image_lab=None, image_b=None):
    # We'll work in LAB colorspace because the card is very blue, and can be
    # more easily isolated in the B (blue-yellow) channel.
    if image_b is None:
        if image_lab is None:
            if image_bgr is None:
                raise ValueError("Must pass one of image_bgr, image_lab, or image_b")
            image_lab = colors.convert_colorspace(image_bgr, cv.COLOR_BGR2LAB)
        image_b = image_lab[..., 2]  # just the blue-yellow channel

    # Invert, threshold, and close the blue-yellow channel to make blue blobs stand out.
    img_inverted = cv.bitwise_not(image_b)
    _thresh, img_thresholded = cv.threshold(
        img_inverted, 0, maxval=255, type=cv.THRESH_OTSU
    )
    img_closed = cv.morphologyEx(
        img_thresholded,
        cv.MORPH_CLOSE,
        kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)),
    )

    # Select just the largest connected object in the closed image, which should be the card.
    num_labels, img_labels, stats, centroids = cv.connectedComponentsWithStats(
        img_closed, connectivity=8
    )
    largest_label = max(
        range(1, num_labels), key=lambda label: stats[label, cv.CC_STAT_AREA]
    )
    largest_label_slice = img_labels == largest_label
    img_just_largest_component = np.where(largest_label_slice, img_closed, 0)

    # Now we do an error-checking step: in the ideal case, the thresholding above
    # should have isolated the card, and the largest connected component of the
    # thresholded image should therefore be the card. But in some images, especially
    # when there is a shadow covering part of the card and a surrounding white
    # surface, the threshold will pick up the shadow as well. In every example I've seen,
    # the shadow makes the largest component extend to the edge of the image,
    # so in that case we do some extra work: find the brightest component of
    # the inverted B channel (i.e., the most blue thing in the thresholded area),
    # and use that as the thing to find contours on instead of the raw thresholded image.
    edge_width = 5
    edges = np.ones_like(img_just_largest_component)
    edges[edge_width:-edge_width, edge_width:-edge_width] = 0

    if np.any(cv.bitwise_and(img_just_largest_component, edges)):
        mix = sklearn.mixture.GaussianMixture(n_components=3)
        clustered = mix.fit_predict(
            X=img_inverted[largest_label_slice].ravel().reshape(-1, 1)
        )
        brightest_cluster = np.argmax(mix.means_)

        show_clusters = np.zeros_like(image_b) - 1
        show_clusters[largest_label_slice] = clustered

        show_brightest = np.zeros_like(image_b)
        show_brightest[show_clusters == brightest_cluster] = 255

        img_for_contours = show_brightest
    else:
        img_for_contours = img_just_largest_component

    # Find the largest, outermost contour (i.e., the card)
    contours, hierarchy = cv.findContours(
        img_for_contours, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=lambda c: cv.contourArea(c))

    img_contour = np.zeros_like(img_for_contours)
    img_contour = cv.drawContours(img_contour, [contour], -1, (255,), thickness=3)

    # Find lines in the contour image.
    lines = cv.HoughLines(
        img_contour, rho=10, theta=np.deg2rad(1), threshold=100,
    ).squeeze()
    lines = [HoughLine(rho, theta) for rho, theta in lines]

    # Find good pairs of horizontal and vertical lines.
    # We won't let each pair be too close, to prevent getting multiple lines
    # on the same side of the card. The limit is quite conservative, since the card
    # usually occupies a large fraction of the image.
    horizontal = []
    vertical = []
    height, width = image_bgr.shape[:2]
    frac = 10
    for line in lines:
        if np.pi / 4 < line.theta < 3 * np.pi / 4:
            if len(horizontal) == 0:
                horizontal.append(line)
            if np.abs(horizontal[0].rho - line.rho) > width / frac:
                horizontal.append(line)
        else:
            if len(vertical) == 0:
                vertical.append(line)
            if np.abs(vertical[0].rho - line.rho) > height / frac:
                vertical.append(line)

        if len(horizontal) >= 2 and len(vertical) >= 2:
            break

    horizontal = horizontal[:2]
    vertical = vertical[:2]

    # Find intersections between the horizontal and vertical lines.
    intersections = []
    for vline, hline in itertools.product(vertical, horizontal):
        if np.isinf(vline.slope):
            x = vline.rho
        else:
            x = (vline.intercept - hline.intercept) / (hline.slope - vline.slope)
        y = hline(x)
        intersections.append((x, y))
    intersections = np.array(intersections)
    # Put the intersections in order by angle relative to their mean position.
    # The order doesn't really matter, as long as it is consistent.
    center = np.mean(intersections, axis=0)
    rel_center = intersections - center
    angles = np.arctan2(rel_center[:, 1], rel_center[:, 0])

    intersections = [
        i
        for idx, i in sorted(
            enumerate(intersections), key=lambda idx_i: angles[idx_i[0]]
        )
    ]

    # TODO: panic if the intersections don't make sense

    return np.array(intersections)


class HoughLine:
    def __init__(self, rho, theta):
        self.rho = np.abs(rho)
        self.theta = theta if rho > 0 else np.pi - theta

        sin = np.sin(theta)
        if sin != 0:
            self.slope = -np.cos(theta) / sin
            self.intercept = rho / sin
        else:
            self.slope = self.intercept = np.inf

    def __repr__(self):
        return f"{self.__class__.__name__}(rho = {self.rho}, theta = {self.theta}, slope = {self.slope}, intercept = {self.intercept})"

    def __call__(self, x):
        return (self.slope * x) + self.intercept


def determine_new_corners(card_corners, target_side_length=None):
    if target_side_length is None:
        target_side_length = max(
            np.linalg.norm(s - e)
            for s, e in utils.window(card_corners + [card_corners[0]])
        )
    tl_corner_x, tl_corner_y = card_corners[0]
    new_corners = np.array(
        [
            card_corners[0],
            [tl_corner_x + target_side_length, tl_corner_y],
            [tl_corner_x + target_side_length, tl_corner_y + target_side_length],
            [tl_corner_x, tl_corner_y + target_side_length],
        ]
    )

    return new_corners


def get_rectifier(old_corners, new_corners):
    transform_matrix = cv.getPerspectiveTransform(
        old_corners.astype(np.float32), new_corners.astype(np.float32)
    )

    def rectify(image):
        return cv.warpPerspective(
            image, transform_matrix, (image.shape[1], image.shape[0]),  # y-x indexing
        )

    return rectify


def corners_to_slice(corners):
    min_x = int(np.floor(np.min(corners[:, 0])))
    max_x = int(np.ceil(np.max(corners[:, 0])))
    min_y = int(np.floor(np.min(corners[:, 1])))
    max_y = int(np.ceil(np.max(corners[:, 1])))

    return slice(min_y, max_y + 1), slice(min_x, max_x + 1)
