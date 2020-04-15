import itertools

import numpy as np
import cv2 as cv


def find_seed_mask(img_lab_cropped, area_cutoff = 40):
    img_b_cropped = img_lab_cropped[..., 2]

    thresh, img_b_thresholded = cv.threshold(
        img_b_cropped, 0, maxval = 255, type = cv.THRESH_OTSU
    )
    not_shadows = find_not_shadows(img_lab_cropped)

    cut_shadows = cv.bitwise_and(img_b_thresholded, not_shadows)

    num_labels, img_b_labels, stats, centroids = cv.connectedComponentsWithStats(
        cut_shadows, connectivity = 8
    )

    seed_labels = np.zeros_like(img_b_cropped)

    # use this to cut everything near the edges of the image
    boundary_width = 3
    edges = np.zeros_like(img_b_thresholded)
    edges[0:boundary_width, :] = 1
    edges[:, 0:boundary_width] = 1
    edges[-boundary_width:-1, :] = 1
    edges[:, -boundary_width:-1] = 1

    labels = []
    for label in range(1, num_labels):
        area = stats[label, cv.CC_STAT_AREA]
        if area < area_cutoff:
            continue

        in_label = np.where(label == img_b_labels, 1, 0).astype(np.uint8)

        if np.any(cv.bitwise_and(edges, in_label)):
            continue

        seed_labels[label == img_b_labels] = label
        labels.append(label)

    return np.where(seed_labels != 0, 255, 0), labels, seed_labels


def find_not_shadows(img_lab_cropped, fudge_factor = 2 / 3):
    img_l_cropped = img_lab_cropped[..., 0]
    l_thresh, _ = cv.threshold(img_l_cropped, 0, maxval = 255, type = cv.THRESH_OTSU)
    _, not_shadows = cv.threshold(
        img_l_cropped, l_thresh * fudge_factor, maxval = 255, type = cv.THRESH_BINARY
    )

    return not_shadows
