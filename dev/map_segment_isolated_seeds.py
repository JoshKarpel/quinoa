import logging

from pathlib import Path
import sys
import subprocess
import csv

import cv2 as cv
import numpy as np

import quinoa as q

import htmap

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent.absolute()
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem


def process(path, out_dir=None):
    if out_dir is None:
        out_dir = Path.cwd()

    img_bgr = q.read_image(path)

    corners = q.find_card_corners(image_bgr=img_bgr)
    new_corners = q.determine_new_corners(corners)
    crop_slice = q.corners_to_slice(new_corners)
    rectifier = q.get_rectifier(corners, new_corners)

    img_bgr_cropped = rectifier(img_bgr)[crop_slice]
    img_lab_cropped = q.convert_colorspace(img_bgr_cropped, cv.COLOR_BGR2LAB)

    seed_blobs, img_seed_labels = q.find_seed_mask(img_lab_cropped)

    rough_singleton_blobs_by_area = q.find_seed_blobs_with_one_seed_rough(
        seed_blobs, "area"
    )
    rough_singleton_blobs_by_perim = q.find_seed_blobs_with_one_seed_rough(
        seed_blobs, "perimeter"
    )

    rough_seed_area = np.mean([b.area for b in rough_singleton_blobs_by_area])
    rough_seed_perim = np.mean([b.perimeter for b in rough_singleton_blobs_by_perim])

    # expand to include not just "exact matches" but also blobs with nearly the right area or perimeter
    singleton_blobs_by_area = [
        blob for blob in seed_blobs if np.rint(blob.area / rough_seed_area) == 1
    ]
    singleton_blobs_by_perim = [
        blob for blob in seed_blobs if np.rint(blob.perimeter / rough_seed_perim) == 1
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

    out_paths = []
    for idx, img in enumerate(
        [img_bgr_cropped, q.overlay_image(img_bgr_cropped, show_counts), show_counts]
    ):
        out_path = out_dir / f"{path.stem}_{idx}.jpg"
        q.write_image(img, out_path)

    per_seed_path = out_dir / f"{path.stem}_per_isolated_seed.csv"
    write_per_seed_csv(per_seed_path, isolated_seeds, img_bgr_cropped)

    avg_seed_path = out_dir / f"{path.stem}_avg_isolated_seed.csv"
    write_avg_seed_csv(avg_seed_path, seed_area, seed_perim)

    all_seed_path = out_dir / f"{path.stem}_all_seeds_rgb.csv"
    write_all_seed_rgb_csv(all_seed_path, img_seed_labels, img_bgr_cropped)

    htmap.transfer_output_files(*out_paths)


def write_per_seed_csv(path, isolated_seeds, image_bgr_cropped):
    with path.open(mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")

        for seed in isolated_seeds:
            b, g, r = np.mean(image_bgr_cropped[seed.slice], axis=0)
            writer.writerow([seed.area, seed.perimeter, r, g, b])


def write_avg_seed_csv(path, seed_area, seed_perimeter):
    with path.open(mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")

        writer.writerow([seed_area, seed_perimeter])


def write_all_seed_rgb_csv(path, img_seed_labels, image_bgr_cropped):
    with path.open(mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")

        for b, g, r in image_bgr_cropped[img_seed_labels != 0]:
            writer.writerow([r, g, b])


if __name__ == "__main__":
    docker_image, s3_url, s3_root, s3_bucket = sys.argv[1:]

    image_names = [
        line.split()[-1]
        for line in subprocess.run(
            ["mc", "ls", f"{s3_root}/{s3_bucket}"], capture_output=True, text=True
        ).stdout.splitlines()
    ]
    image_paths = [[f"s3://{s3_url}/{s3_bucket}/{im}"] for im in image_names]

    htmap.settings["DOCKER.IMAGE"] = docker_image

    s3_keys_root = Path.home() / ".chtc_s3"

    m = htmap.map(
        process,
        image_names,
        map_options=htmap.MapOptions(
            request_memory="2GB",
            request_disk="1GB",
            input_files=image_paths,
            aws_access_key_id_file=(s3_keys_root / "access.key").as_posix(),
            aws_secret_access_key_file=(s3_keys_root / "secret.key").as_posix(),
        ),
    )

    print(m, len(m))
