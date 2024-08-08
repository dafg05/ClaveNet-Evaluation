from evaluation import evaluation as eval
from evaluation import evalDatasets
from tests.constants import MODEL_PATH, EVALUATION_SET_PATH, TEST_OUT_DIR

OUT_DIR = TEST_OUT_DIR / 'evaluation'
INDICES = [3, 5, 9]

def test_evaluate_model():
    eval_time = eval.evaluateModel(OUT_DIR, MODEL_PATH, EVALUATION_SET_PATH, num_points=500)
    print(f"Output evaluation saved to {OUT_DIR}. Evaltime: {eval_time}")

def test_simple_evaluate_model():
    eval_time = eval.evaluateModel(OUT_DIR, MODEL_PATH, EVALUATION_SET_PATH, simple=True, num_points=500)
    print(f"Output evaluation saved to {OUT_DIR}. Evaltime: {eval_time}")

def test_audio_eval():
    evaluation_set = evalDatasets.EvaluationHvoDataset(EVALUATION_SET_PATH)
    audio_out_dir = OUT_DIR / 'audio_samples'
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    eval.audioEval(audio_out_dir, MODEL_PATH, evaluation_set, INDICES)
    print(f"Audio evaluation saved to {audio_out_dir}")

if __name__ == "__main__":
    test_evaluate_model()
    test_simple_evaluate_model()
    test_audio_eval()
