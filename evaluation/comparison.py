from evaluation.constants import *
import evaluation.featureExtractor as featExt
import evaluation.utils as utils

from typing import Dict
from tqdm import tqdm

GENERATED_INTRASET_KEY = "generated_intraset"
EVALUATION_INTRASET_KEY = "evaluation_intraset"
INTERSET_KEY = "interset"

MEAN_KEY = "mean"
STD_KEY = "std"

class ComparisonResult:
    """
    Holds the results of a relative comparison between two hvo sets. For each feature, 
    it computes the kl_divergence and overlapping_area between 
    the pdf of the generated intraset distances and the pdf of the interset distances.
    """

    def __init__(self, kl_divergence, overlapping_area, kde_dict, points):
        self.kl_divergence = kl_divergence
        self.overlapping_area = overlapping_area
        self.kde_dict = kde_dict
        self.points = points
    
    def __str__(self):
        return f"kl_divergence: {self.kl_divergence}, overlapping_area: {self.overlapping_area}, kde_dict: {self.kde_dict}, stats_dict: {self.stats_dict}, points: {self.points}"
    
class SimpleComparisonResult:
    """
    Like ComparisonResult, but stores mean + std instead of kdes.
    """
    def __init__(self, kl_divergence, overlapping_area, stats_dict, points):
        self.kl_divergence = kl_divergence
        self.overlapping_area = overlapping_area
        self.stats_dict = stats_dict
        self.points = points

def relative_comparison(generated_set, evaluation_set, features_to_extract=EVAL_FEATURES, simple=False, num_points=1000, padding_factor=0.05, use_tqdm = True) -> Dict[str, ComparisonResult]:
    """
    Runs a relative comparison between two hvo sets. For each feature, it computes the kl_divergence and overlapping_area between the pdf of the validation intraset distances and the pdf of the interset distances.
    Returns the kdes for each set and the interset, the points used to evaluate the kdes, and the kl_divergence and overlapping_area.
    
    :param generated_set: hvo set to be used as the generated set
    :param evaluation_set: hvo set to be used as the evaluation set
    :param features_to_extract: list of features to be extracted from the hvo sets
    :param simple: if True, returns a dict of SimpleComparisonResult instead of ComparisonResult
    :param num_points: number of points to be used to evaluate the kdes and metrics
    :param padding_factor: factor to be used to pad the range of the kdes

    :return: dictionary with the comparison results for each feature
    """

    if features_to_extract != EVAL_FEATURES:
        raise ValueError("Relative comparison currently only supports the default feature set.")
    
    assert len(generated_set) == len(evaluation_set), "Generated and validation sets must be the same length."
    
    generated_features =  featExt.get_features_dict(generated_set, features_to_extract)
    evaluation_features = featExt.get_features_dict(evaluation_set, features_to_extract)

    generated_intraset_dd_dict = featExt.get_intraset_dd_dict(generated_features)
    evaluation_intraset_dd_dict = featExt.get_intraset_dd_dict(evaluation_features)

    interset_dd_dict = featExt.get_interset_dd_dict(generated_features, evaluation_features)

    comparison_results_by_feat = {feature: None for feature in features_to_extract}

    iterable = tqdm(features_to_extract, desc="Computing relative comparison metrics") if use_tqdm else features_to_extract
    for feature in iterable:
        # compute kl_divergence and overlapping_area for each feature
        generation_dd = generated_intraset_dd_dict[feature]
        evaluation_dd = evaluation_intraset_dd_dict[feature]
        interset_dd = interset_dd_dict[feature]

        points = utils.evaluation_points([generation_dd, evaluation_dd, interset_dd], num_points, padding_factor)
        
        # compute comparison metrics between the validation intraset pdf and the interset pdf
        kl_d = utils.kl_divergence(evaluation_dd.kde, interset_dd.kde, points)
        oa = utils.overlapping_area(evaluation_dd.kde, interset_dd.kde, points)

        if simple:
            stats_dict = {
                GENERATED_INTRASET_KEY: {
                    MEAN_KEY: generation_dd.mean,
                    STD_KEY: generation_dd.std
                },
                EVALUATION_INTRASET_KEY: {
                    MEAN_KEY: evaluation_dd.mean,
                    STD_KEY: evaluation_dd.std
                },
                INTERSET_KEY: {
                    MEAN_KEY: interset_dd.mean,
                    STD_KEY: interset_dd.std
                }
            }
            comparison_results_by_feat[feature] = SimpleComparisonResult(kl_d, oa, stats_dict, points)
        else: 
            kde_dict = {
                GENERATED_INTRASET_KEY: generation_dd.kde,
                EVALUATION_INTRASET_KEY: evaluation_dd.kde,
                INTERSET_KEY: interset_dd.kde
            }
            comparison_results_by_feat[feature] = ComparisonResult(kl_d, oa, kde_dict, points)

    return comparison_results_by_feat
