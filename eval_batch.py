import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from evaluation import evaluation as eval

EVALUATION_OUT_DIR = Path('eval_out')
EVALUATION_DATASET = Path('mock_evaluation_set')
MODELS_DIR = Path('models')
EVALUATION_ERROR_LOGS = 'eval_errors.log'

def eval_pipeline(model_paths):
    # name each run with a timestamp
    time_str = str(int(datetime.now().timestamp()))
    run_dir = Path(EVALUATION_OUT_DIR, time_str)
    Path.mkdir(run_dir, exist_ok=True)

    error_log_path = run_dir / EVALUATION_ERROR_LOGS
    with open(error_log_path, 'w') as f:
        f.write(f"Eval error log for run {time_str} \n")

    error_count = 0
    for model_path in tqdm(model_paths, desc="Evaluation pipeline"):
        try:
            # evaluate the model
            evaluation_path = eval.evaluateModel(run_dir, model_path, EVALUATION_DATASET, simple=True)
        except Exception as e:
            print("An error occured while evaluating the model.")
            with open(error_log_path, 'a') as f:
                f.write(f"Error evaluating model: {model_path}. Stack trace: {traceback.format_exc()} \n")
                error_count += 1


def get_model_paths(models_dir):
    return [model_path for model_path in models_dir.iterdir() if model_path.suffix == '.pth']

if __name__ == "__main__":
    model_paths = get_model_paths(MODELS_DIR)
    eval_pipeline(model_paths)
