from pathlib import Path

TEST_OUT_DIR = Path(__file__).parent / 'out'
TEST_DATA_DIR = Path(__file__).parent / 'data'

MODEL_PATH = TEST_DATA_DIR / 'smol_solar-shadow_1723132208.pth'
EVALUATION_SET_PATH = TEST_DATA_DIR / 'test_preprocessed_evalset'

EVAL_RUN_PATH_1 = TEST_DATA_DIR / "eval_runs" / "smol_hopeful-gorge_1723146153_evaluation_1723146433"
EVAL_RUN_PATH_2 = TEST_DATA_DIR / "eval_runs" / "smol_rosy-durian_1723146262_evaluation_1723146470"
EVAL_RUN_PATH_3 = TEST_DATA_DIR / "eval_runs" / "smol_solar-shadow_1723132208_evaluation_1723142369"

EVAL_RUN_PATHS = [EVAL_RUN_PATH_1, EVAL_RUN_PATH_2, EVAL_RUN_PATH_3]
REPORT_PATH = TEST_DATA_DIR / "test_training_runs_report.csv"
