import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import stats

import quinoa as q


def process(path, out_dir = None):
    if out_dir is None:
        out_dir = Path.cwd()

    img_bgr = q.read_image(path)

    corners = q.find_card_corners(image_bgr = img_bgr)
    new_corners = q.determine_new_corners(corners)
    crop_slice = q.corners_to_slice(new_corners)
    rectifier = q.get_rectifier(corners, new_corners)

    img_bgr_cropped = rectifier(img_bgr)[crop_slice]
    img_lab_cropped = q.convert_colorspace(img_bgr_cropped, cv.COLOR_BGR2LAB)

    seed_blobs, img_seed_labels = q.find_seed_mask(img_lab_cropped)

    rough_singleton_blobs_by_area = find_seed_blobs_with_one_seed_rough(
        seed_blobs, "area"
    )
    rough_singleton_blobs_by_perim = find_seed_blobs_with_one_seed_rough(
        seed_blobs, "perimeter"
    )

    rough_seed_area = np.mean([b.area for b in rough_singleton_blobs_by_area])
    rough_seed_perim = np.mean([b.perimeter for b in rough_singleton_blobs_by_perim])

    # expand to include not just "exact matches" but also blobs with nearly the right area or perimeter
    singleton_blobs_by_area = [
        blob for blob in seed_blobs if np.rint(blob.area / rough_seed_area) == 1
    ]
    singleton_blobs_by_perim = [
        blob
        for blob in seed_blobs
        if np.rint(blob.perimeter / rough_seed_perim) == 1
    ]

    # this is our final estimate of the seed area
    seed_area = np.mean([b.area for b in singleton_blobs_by_area])
    seed_perim = np.mean([b.perimeter for b in singleton_blobs_by_perim])

    # display
    show_counts = np.zeros_like(img_bgr_cropped)

    isolated_seeds = []
    for blob in seed_blobs:
        area_ratio = int(np.rint(blob.area / seed_area))
        perim_ratio = int(np.rint(blob.perimeter / seed_perim))

        if area_ratio == perim_ratio and area_ratio < len(q.BGR_COLORS_8):
            color = q.BGR_COLORS_8[area_ratio]
            isolated_seeds.append(blob)
        else:
            color = q.MAGENTA

        show_counts[blob.label == img_seed_labels] = color

    for idx, img in enumerate(
        [img_bgr_cropped, q.overlay_image(img_bgr_cropped, show_counts), show_counts]
    ):
        q.write_image(img, out_dir / f"{path.stem}_{idx}.jpg")

    write_per_seed_csv(out_dir / f"{path.stem}_per_isolated_seed.csv", isolated_seeds, img_bgr_cropped)
    write_avg_seed_csv(out_dir / f"{path.stem}_avg_isolated_seed.csv", seed_area, seed_perim)
    write_all_seed_rgb_csv(out_dir / f"{path.stem}_all_seeds_rgb.csv", img_seed_labels, img_bgr_cropped)


def write_per_seed_csv(path, isolated_seeds, image_bgr_cropped):
    with path.open(mode = 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',')

        for seed in isolated_seeds:
            b, g, r = np.mean(image_bgr_cropped[seed.slice], axis = 0)
            writer.writerow([seed.area, seed.perimeter, r, g, b])


def write_avg_seed_csv(path, seed_area, seed_perimeter):
    with path.open(mode = 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',')

        writer.writerow([seed_area, seed_perimeter])


def write_all_seed_rgb_csv(path, img_seed_labels, image_bgr_cropped):
    with path.open(mode = 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',')

        for b, g, r in image_bgr_cropped[img_seed_labels != 0]:
            writer.writerow([r, g, b])


def find_seed_blobs_with_one_seed_rough(seed_blobs, measure):
    getter = lambda blob: getattr(blob, measure)

    all_measures = np.array([getter(blob) for blob in seed_blobs])

    # for each measure, count up the number of blobs that have (roughly) that measure
    num_matching_measure_ratios = {}
    for blob in seed_blobs:
        ratios = all_measures / getter(blob)
        rounded = np.rint(ratios)
        num_matching_measure_ratios[blob] = np.count_nonzero(rounded == 1)

    # assuming that most seeds are alone, the most common sum is the one where
    # the measure we divided by was roughly the measure of a single seed
    most_common_sum = stats.mode(list(num_matching_measure_ratios.values()))
    mode = most_common_sum.mode[0]

    return [blob for blob, sum in num_matching_measure_ratios.items() if sum == mode]
