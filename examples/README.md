# Input specification and synthetic example

The self-contained demo does not require study data. It creates two synthetic
BOLD time series in memory and writes them to the selected output directory.
They are artificial signals generated from a fixed random seed and contain no
participant information.

The study pipeline expects one preprocessed resting-state fMRI time series per
CSV file:

- rows: consecutive BOLD frames;
- columns: 166 AAL3 regions in the ordering used by the project;
- first row: column names;
- values: finite numeric values;
- minimum length: more than 37 frames when the first 30 frames are discarded
  and seven preceding frames are used to forecast the next frame.

The demo applies the same per-region temporal z-score normalization and
seven-frame window construction described in the manuscript. Its model is
deliberately small and is trained for only a few iterations so that the command
finishes quickly. Demo metrics verify software execution only and have no
scientific or clinical interpretation.

Run from the repository root:

```bash
python scripts/run_demo.py --output-dir demo_outputs
```

Expected files are:

```text
demo_outputs/
├── fbdm_predictions.npy
├── ibdm_predictions.npy
├── metrics.json
├── synthetic_participant_bold.csv
├── synthetic_reference_bold.csv
└── targets.npy
```
