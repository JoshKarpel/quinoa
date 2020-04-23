import itertools
import dataclasses

import numpy as np
import cv2 as cv


def find_seed_mask(
    img_lab_cropped,
    area_cutoff=40,
    cut_edges=True,
    edge_cut_fraction=0.01,
    cut_bottom=True,
    bottom_cut_fraction=0.1,
):
    img_b_cropped = img_lab_cropped[..., 2]

    thresh, img_b_thresholded = cv.threshold(
        img_b_cropped, 0, maxval=255, type=cv.THRESH_OTSU
    )
    not_shadows = find_not_shadows(img_lab_cropped)

    cut_shadows = cv.bitwise_and(img_b_thresholded, not_shadows)

    num_labels, img_b_labels, stats, centroids = cv.connectedComponentsWithStats(
        cut_shadows, connectivity=8
    )

    blob_labels = np.zeros_like(img_b_cropped)

    # use this to cut everything near the edges of the image
    boundary_width = int(np.ceil(edge_cut_fraction * np.min(img_b_cropped.shape)))
    edges = np.zeros_like(img_b_thresholded)
    edges[0:boundary_width, :] = 1
    edges[:, 0:boundary_width] = 1
    edges[-boundary_width:-1, :] = 1
    edges[:, -boundary_width:-1] = 1

    # use this to cut out the numbers at the bottom of the card
    bottom_cut_index = int(np.ceil(bottom_cut_fraction * img_b_cropped.shape[0]))
    cut_numbers = np.zeros_like(img_b_thresholded)
    cut_numbers[-bottom_cut_index:-1, :] = 1

    blobs = []
    for label in range(1, num_labels):
        area = stats[label, cv.CC_STAT_AREA]
        if area < area_cutoff:
            continue

        just_this_label = np.where(label == img_b_labels, 1, 0).astype(np.uint8)

        if cut_edges and np.any(cv.bitwise_and(edges, just_this_label)):
            continue

        if cut_bottom and np.any(cv.bitwise_and(cut_numbers, just_this_label)):
            continue

        contours, hierarchy = cv.findContours(
            just_this_label, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
        )
        contour = contours[0]

        slice = label == img_b_labels
        blob_labels[slice] = label
        blobs.append(Blob(label=label, slice = slice, area=area, contour=contour))

    return blobs, blob_labels


def find_not_shadows(img_lab_cropped, fudge_factor=2 / 3):
    img_l_cropped = img_lab_cropped[..., 0]
    l_thresh, _ = cv.threshold(img_l_cropped, 0, maxval=255, type=cv.THRESH_OTSU)
    _, not_shadows = cv.threshold(
        img_l_cropped, l_thresh * fudge_factor, maxval=255, type=cv.THRESH_BINARY
    )

    return not_shadows


@dataclasses.dataclass(frozen=True)
class Blob:
    label: int
    area: float
    slice: np.array
    contour: np.array

    def __repr__(self):
        return f'Blob(label={self.label})'

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    @property
    def perimeter(self):
        return cv.arcLength(self.contour, closed=True)
