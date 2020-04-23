import logging

from pathlib import Path
from pprint import pprint
import itertools

from tqdm import tqdm, trange

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import sklearn.mixture as mixtures
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
    new_corners = q.determine_new_corners(corners, target_side_length=None)
    crop_slice = q.corners_to_slice(new_corners)
    rectifier = q.get_rectifier(corners, new_corners)

    img_bgr_cropped = rectifier(img_bgr)[crop_slice]
    img_lab_cropped = q.convert_colorspace(img_bgr_cropped, cv.COLOR_BGR2LAB)

    seed_blobs, img_seed_labels = q.find_seed_mask(img_lab_cropped)

    print(len(seed_blobs))

    rough_singleton_blobs_by_area = find_seed_blobs_with_one_seed_rough(
        seed_blobs, "area"
    )
    rough_singleton_blobs_by_perim = find_seed_blobs_with_one_seed_rough(
        seed_blobs, "perimeter"
    )
    print(len(rough_singleton_blobs_by_area))
    print(len(rough_singleton_blobs_by_perim))
    # single_by_both = set

    rough_seed_area = np.mean([b.area for b in rough_singleton_blobs_by_area])
    rough_seed_perim = np.mean([b.perimeter for b in rough_singleton_blobs_by_perim])
    print("rough area", rough_seed_area)
    print("rough perim", rough_seed_perim)

    # expand to include not just "exact matches" but also blobs with nearly the right area or perimeter
    singleton_blobs_by_area = [
        blob for blob in seed_blobs if int(np.rint(blob.area / rough_seed_area)) == 1
    ]
    singleton_blobs_by_perim = [
        blob
        for blob in seed_blobs
        if int(np.round(blob.perimeter / rough_seed_perim)) == 1
    ]

    # this is our final estimate of the seed area
    seed_area = np.mean([b.area for b in singleton_blobs_by_area])
    seed_perim = np.mean([b.perimeter for b in singleton_blobs_by_perim])
    print("final area", seed_area)
    print("final perim", seed_perim)

    # display
    show_counts = np.zeros_like(img_bgr_cropped)

    for blob in seed_blobs:
        area_ratio = np.round(blob.area / seed_area)
        perim_ratio = np.round(blob.perimeter / seed_perim)

        area_ratio = int(area_ratio)
        perim_ratio = int(perim_ratio)

        if area_ratio == perim_ratio and area_ratio < len(q.BGR_COLORS_8):
            color = q.BGR_COLORS_8[area_ratio]
        else:
            color = q.MAGENTA

        show_counts[blob.label == img_seed_labels] = color

    q.show_image(q.overlay_image(img_bgr_cropped, show_counts))

    for idx, img in enumerate(
        [img_bgr_cropped, q.overlay_image(img_bgr_cropped, show_counts), show_counts]
    ):
        q.write_image(img, OUT / f"{path.stem}_{idx}.jpg")


def find_seed_blobs_with_one_seed_rough(seed_blobs, measure):
    getter = lambda blob: getattr(blob, measure)

    all_measures = np.array([getter(blob) for blob in seed_blobs])

    # for each measure, count up the number of blobs that have (roughly) that measure
    num_matching_measure_ratios = {}
    for blob in seed_blobs:
        ratios = all_measures / blob.area
        rounded = np.rint(ratios)
        num_matching_measure_ratios[blob] = np.count_nonzero(rounded == 1)

    # assuming that most seeds are alone, the most common sum is the one where
    # the measure we divided by was roughly the measure of a single seed
    most_common_sum = stats.mode(list(num_matching_measure_ratios.values()))
    mode = most_common_sum.mode[0]

    return [blob for blob, sum in num_matching_measure_ratios.items() if sum == mode]


if __name__ == "__main__":
    # image_paths = [path for path in (DATA / 'aus').iterdir() if path.suffix.lower() == ".jpg"]
    image_paths = [DATA / "aus" / "104.JPG"]

    for path in tqdm(image_paths):
        process(path)
