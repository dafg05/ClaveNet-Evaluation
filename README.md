# ClaveNet-Evaluation

## Comparison module

## Evaluation module

The module contained in `evaluation.py` processes the evaluation set, creates a monotonic set based on the evaluation set, uses the montonic set and the provided model to create a generated set, and then uses the comparison module to compare the evaluation set and the generated set.

### Arguments

#### Model to evaluate

Provide the path to the model that will be evaluated via comparison.

#### Evaluation set

Provide the path to the evaluation set that will be compared to the model's generated set.

#### Number of points for KDE estimation

Default is 10000.

#### Simple (Return KDE or just mean and std)

If `simple=False`, the KDEs will be pickled as part of the output. Otherwise, the mean and std of these will be provided as part of the output csv.

#### Output dir

Provide the path to where the results will be written.

## Output



## Evaluation batch processing

## Usage

## Output


