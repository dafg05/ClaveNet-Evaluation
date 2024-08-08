import torch
from hvo_sequence.hvo_seq import HVO_Sequence

from evaluation.evalDatasets import EvaluationHvoDataset, MonotonicHvoDataset, GeneratedHvoDataset
from evaluation.constants import SF_PATH
from tests.constants import MODEL_PATH, EVALUATION_SET_PATH, TEST_OUT_DIR

from cnarch.grooveTransformer import GrooveTransformer

AUDIO_OUT_DIR = TEST_OUT_DIR / 'datasets'

# Load model
MODEL = GrooveTransformer(d_model = 8, nhead = 4, num_layers=11, dim_feedforward=16, dropout=0.1594, voices=9, time_steps=32, hit_sigmoid_in_forward=False)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

SUBSET_SIZE = 4

def testEvalDatasets():
    evaluation_set = EvaluationHvoDataset(EVALUATION_SET_PATH)
    print(f"Evaluation set length: {len(evaluation_set)}")
    
    # check if evaluation set is valid
    for i in range(len(evaluation_set)):
        evaluation_seq = evaluation_set[i]
        assert evaluation_seq.hvo.shape == (32, 27), f"Evaluation hvo shape is invalid: {evaluation_seq.hvo.shape}"
        filename = getAudioFilename(evaluation_seq, "evaluation")
        evaluation_seq.save_audio(filename=filename, sf_path=SF_PATH)
    print("evaluation set looking good")

    monotonic_set = MonotonicHvoDataset(evaluation_set)
    # check if monotonic set is valid
    assert len(monotonic_set) == len(evaluation_set)
    assert monotonic_set[0].master_id == evaluation_set[0].master_id
    assert monotonic_set[0].style_primary == evaluation_set[0].style_primary
    for i in range(SUBSET_SIZE):
        monotonic_seq = monotonic_set[i]
        assert monotonic_seq.hvo.shape == (32, 27), f"Monotonic hvo shape is invalid: {monotonic_seq.hvo.shape}"
        filename = getAudioFilename(monotonic_seq, "monotonic")
        monotonic_seq.save_audio(filename=filename, sf_path=SF_PATH)
    print("monotonic set looking good")

    generated_set = GeneratedHvoDataset(monotonic_set, MODEL)
    # check if generated set is valid
    assert len(generated_set) == len(monotonic_set)
    assert generated_set[0].master_id == monotonic_set[0].master_id
    assert generated_set[0].style_primary == monotonic_set[0].style_primary
    for i in range(SUBSET_SIZE):
        generated_seq = generated_set[i]
        assert generated_seq.hvo.shape == (32, 27), f"Generated hvo shape is invalid: {generated_seq.hvo.shape}"
        filename = getAudioFilename(generated_seq, "generated")
        generated_seq.save_audio(filename=filename, sf_path=SF_PATH)
    print("generated set looking good")

    print("All sets looking good")

def getAudioFilename(hvo_seq, prefix):
    master_id = hvo_seq.master_id.replace("/", "-")
    return f"{AUDIO_OUT_DIR}/{prefix}_{master_id}.wav"

if __name__ == "__main__":
    testEvalDatasets()
