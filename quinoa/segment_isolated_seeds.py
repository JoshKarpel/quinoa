import numpy as np
from scipy import stats


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
