import pandas as pd

from tests.constants import TEST_OUT_DIR, EVAL_RUN_PATH_1, EVAL_RUN_PATHS, REPORT_PATH

from evaluation import analysis

OUT_DIR = TEST_OUT_DIR / "analysis"

LOSSES_DICT = {
    "model1": 0.1,
    "model2": 0.2,
    "model3": 0.3,
    "model4": 0.4,
}

def test_get_min_loss_entries():
    n = 2
    expected = {
        "model1": 0.1,
        "model2": 0.2,
    }
    result = analysis.get_min_loss_entries(LOSSES_DICT, n)
    assert result == expected
    print(f"Test get_min_loss_entries passed")

def test_get_model_name_from_eval_run():
    eval_run_path = EVAL_RUN_PATH_1
    model_name = analysis.get_model_name_from_eval_run(eval_run_path)
    assert model_name == "smol_hopeful-gorge_1723146153", f'Got: {model_name}'
    print(f"Test get_model_name_from_eval_run passed")

def test_analysis():
    n = 2
    analysis_data = analysis.analysis(EVAL_RUN_PATHS, REPORT_PATH, n)
    analysis_df = pd.DataFrame(analysis_data)
    assert analysis_df.shape[0] == n, f"Expected {n} rows, got {analysis_df.shape[0]}"
    # write the dataframe to a csv file
    analysis_df.to_csv(OUT_DIR / "analysis.csv")
    print(f"Analysis data written to csv file at {OUT_DIR / 'analysis.csv'}")

def test_analysis_reduced_features():
    n = 2
    analysis_data = analysis.analysis(EVAL_RUN_PATHS, REPORT_PATH, n, reduced_features=True)
    analysis_df = pd.DataFrame(analysis_data)
    assert analysis_df.shape[0] == n, f"Expected {n} rows, got {analysis_df.shape[0]}"
    # write the dataframe to a csv file
    analysis_df.to_csv(OUT_DIR / "analysis_reduced_features.csv")
    print(f"Reduced Analysis data written to csv file at {OUT_DIR / 'analysis_reduced_features.csv'}")

def test_analysis_all_models():
    analysis_data = analysis.analysis(EVAL_RUN_PATHS, REPORT_PATH, reduced_features=True)
    analysis_df = pd.DataFrame(analysis_data)
    assert analysis_df.shape[0] == len(EVAL_RUN_PATHS), f"Expected {len(EVAL_RUN_PATHS)} rows, got {analysis_df.shape[0]}"
    # write the dataframe to a csv file
    analysis_df.to_csv(OUT_DIR / "analysis_all_models.csv")
    print(f"Analysis data for all models written to csv file at {OUT_DIR / 'analysis_all_models.csv'}")

if __name__ == "__main__":
    test_get_min_loss_entries()
    test_get_model_name_from_eval_run()
    test_analysis()
    test_analysis_reduced_features()
    test_analysis_all_models()
    
    
    
