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

DEFAULT_FILTERS = {
    "drummer": None,  # ["drummer1", ..., and/or "session9"]
    "session": None,  # ["session1", "session2", and/or "session3"]
    "loop_id": None,
    "master_id": None,
    "style_primary": None,  # [funk, latin, jazz, rock, gospel, punk, hiphop, pop, soul, neworleans, afrobeat]
    "bpm": None,  # [(range_0_lower_bound, range_0_upper_bound), ..., (range_n_lower_bound, range_n_upper_bound)]
    "beat_type": ["beat"],  # ["beat" or "fill"]
    "time_signature": ["4-4"],  # ["4-4", "3-4", "6-8"]
    "full_midi_filename": None,  # list_of full_midi_filenames
    "full_audio_filename": None  # list_of full_audio_filename
}

ROCK_FILTERS = DEFAULT_FILTERS.copy()
ROCK_FILTERS["style_primary"] = ["rock"]

AFROBEAT_FILTERS = DEFAULT_FILTERS.copy()
AFROBEAT_FILTERS["style_primary"] = ["afrobeat"]
