from pathlib import Path

HYPERS_DIR = Path(__file__).parent.parent / 'hypers'
SF_PATH = str(Path(__file__).parent.parent / "soundfonts/Standard_Drum_Kit.sf2")
RANDOM_SEED = 42

# Features

NUM_INSTRUMENTS_KEY = "num_instruments"
TOTAL_STEP_DENSITY_KEY = "total_step_density"
AVERAGE_VOICE_DENSITY_KEY = "average_voice_density"
VEL_SIMILARITY_SCORE_KEY = "vel_similarity_score"
COMBINED_SYNCOPATION_KEY = "combined_syncopation"
POLYPHONIC_SYNCOPATION_KEY = "polyphonic_syncopation"
LOW_SYNC_KEY = "low_sync"
MID_SYNC_KEY = "mid_sync"
HIGH_SYNC_KEY = "high_sync"
LOW_SYNESS_KEY = "low_syness"
MID_SYNESS_KEY = "mid_syness"
HIGH_SYNESS_KEY = "high_syness"
COMPLEXITY_KEY = "complexity"
LAIDBACKNESS_KEY = "laidbackness"
TIMING_ACCURACY_KEY = "timing_accuracy"
AUTO_CORR_SKEW_KEY = "auto_corr_skew"
AUTO_CORR_MAX_KEY = "auto_corr_max"
AUTO_CORR_CENTROID_KEY = "auto_corr_centroid"
AUTO_CORR_HARMONICITY_KEY = "auto_corr_harmonicity"

EVAL_FEATURES = [
    NUM_INSTRUMENTS_KEY,
    TOTAL_STEP_DENSITY_KEY,
    AVERAGE_VOICE_DENSITY_KEY,
    VEL_SIMILARITY_SCORE_KEY,
    COMBINED_SYNCOPATION_KEY,
    POLYPHONIC_SYNCOPATION_KEY,
    LOW_SYNC_KEY,
    MID_SYNC_KEY,
    HIGH_SYNC_KEY,
    LOW_SYNESS_KEY,
    MID_SYNESS_KEY,
    HIGH_SYNESS_KEY,
    COMPLEXITY_KEY,
    LAIDBACKNESS_KEY,
    # TIMING_ACCURACY_KEY,
    AUTO_CORR_SKEW_KEY,
    AUTO_CORR_MAX_KEY,
    AUTO_CORR_CENTROID_KEY,
    AUTO_CORR_HARMONICITY_KEY
]
