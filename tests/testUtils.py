import numpy as np

from evaluation.distanceData import DistanceData
from evaluation import utils

from unittest.mock import MagicMock
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

FEAT_VALUES_A = np.array([1, 2, 3, 4])
FEAT_VALUES_B = np.array([5, 6, 7, 8])

EXPECTED_INTRASET_DISTANCE_MATRIX = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
EXP_INTERSET_DISTANCE_MATRIX = np.array([[4, 5, 6, 7], [3, 4, 5, 6], [2, 3, 4, 5], [1, 2, 3, 4]])


def test_get_intraset_distance_matrix():
    actual_intraset_distance_matrix = utils.get_intraset_distance_matrix(FEAT_VALUES_A)
    assert actual_intraset_distance_matrix.all() == EXPECTED_INTRASET_DISTANCE_MATRIX.all()
    print('test_get_intraset_distance_matrix passed')


def test_get_interset_distance_matrix():
    actual_interset_distance_matrix = utils.get_interset_distance_matrix(FEAT_VALUES_A, FEAT_VALUES_B)
    assert actual_interset_distance_matrix.all() == EXP_INTERSET_DISTANCE_MATRIX.all()
    print('test_get_interset_distance_matrix passed')

def test_kl_divergence_identical_kdes():
    data_a = np.random.normal(0, 1, 1000)
    data_b = np.copy(data_a)

    print(data_a.shape)
    print(data_b.shape)

    kde_a = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_a.reshape(-1, 1))
    kde_b = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_b.reshape(-1, 1))

    points = np.array([1,2,3,4])

    actual_kl_divergence = utils.kl_divergence(kde_a, kde_b, points)
    assert actual_kl_divergence == 0
    print('test_kl_divergence_identical_kdes passed')


def test_kl_divergence_normal_distributions():
    """
    Tests KL Divergence between two normal distributions.
    The KL_divergence should be equal to the following:

    log(std_2/std_1) + (std_1^2 + ((mean_1 - mean_2)^2)/(2*std_2^2)) - 1/2
    = log(std_2/std_1) + (std_1^2/(2*std_2^2)) - 1/2
    
    Source: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """

    data_a = np.random.normal(0, 1, 10000)
    data_b = np.random.normal(0, 2, 10000)
    
    kde_a = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_a.reshape(-1, 1))
    kde_b = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_b.reshape(-1, 1))

    # min and max of points are more than 3 standard deviations away from the mean for both distributions
    points = np.linspace(-20, 20, 10000)

    expected_kl_divergence = np.log(2/1) + (1/(2 * 2**2)) - 1/2
    actual_kl_divergence = utils.kl_divergence(kde_a, kde_b, points)

    print("Expected KL Divergence: ", expected_kl_divergence)
    print("Actual KL Divergence: ", actual_kl_divergence)
    assert np.isclose(expected_kl_divergence, actual_kl_divergence, rtol=5e-1), f"Expected kl_divergence: {expected_kl_divergence}; Actual kl_divergence: {actual_kl_divergence}"
    print('test_kl_divergence_normal_distributions passed')

def test_overlapping_area():
    np.random.seed(100)

    mean_1, mean_2 = 1, 3
    std_1, std_2 = 1, 1
    intersection = (1+3)/2

    data_a = np.random.normal(mean_1, std_1, 1000)
    data_b = np.random.normal(mean_2, std_2, 1000)
    
    kde_a = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_a.reshape(-1, 1))
    kde_b = KernelDensity(kernel='gaussian', bandwidth='scott').fit(data_b.reshape(-1, 1))

    # min and max of points are more than 3 standard deviations away from the mean for both distributions
    points = np.linspace(-10, 10, 10000)

    expected_oa = norm.cdf(intersection, mean_2, std_2) + (1 - norm.cdf(intersection, mean_1, std_1))
    actual_oa = utils.overlapping_area(kde_a, kde_b, points)
    assert np.isclose(expected_oa, actual_oa, rtol=5e-1), f"Expected overlapping area: {expected_oa}; Actual overlapping area: {actual_oa}"
    print("Expected Overlapping Area: ", expected_oa)
    print("Actual Overlapping Area: ", actual_oa)
    print('test_overlapping_area passed')

def test_evaluation_points():

    dd_a = MagicMock(DistanceData)
    dd_a.flattened_distances = np.array([7,8,9,10])
    dd_b = MagicMock(DistanceData)
    dd_b.flattened_distances = np.array([0,1,2,3])
    dd_interset = MagicMock()

    num_points = 11
    padding_factor = 2

    expected_min_val = -20 # min - padding
    expected_max_val = 30 # max + padding

    expected_points = np.linspace(expected_min_val, expected_max_val, num_points)
    actual_points = utils.evaluation_points([dd_a, dd_b, dd_interset], num_points, padding_factor)

    print("actual_points: ", actual_points)
    assert np.array_equal(expected_points, actual_points)
    print('test_evaluation_points passed')


if __name__ == "__main__":
    test_get_intraset_distance_matrix()
    test_get_interset_distance_matrix()
    test_kl_divergence_identical_kdes()
    test_kl_divergence_normal_distributions()
    test_overlapping_area()
    test_evaluation_points()
