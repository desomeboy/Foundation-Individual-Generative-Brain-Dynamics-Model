# Predicting Neuromodulation Outcome for Parkinson's Disease with a Generative Brain Dynamics Model

This repository provides the code implementation and data-processing workflow
for the Generative Brain Dynamics Model (GBDM) described in the manuscript. A
Foundation Brain Dynamics Model (FBDM) forecasts the next 166-region BOLD state
from seven preceding frames and clinical context; participant-specific
fine-tuning yields an Individualized Brain Dynamics Model (iBDM). The generated
BOLD forecasts are used to derive Counterfactual Brain Mismatch (CBM) features
for downstream neuromodulation-response prediction.

Study imaging data are not bundled with the repository. A self-contained
synthetic smoke test is provided so that the model interface can be run without
access to any study data.

## Quick start: data-independent smoke test

The following commands create the tested Python 3.9 environment and run the
synthetic example:

```bash
git clone https://github.com/desomeboy/Foundation-Individual-Generative-Virtual-Brain.git
cd Foundation-Individual-Generative-Virtual-Brain
conda env create -f environment.yml
conda activate gbdm
python scripts/run_demo.py --output-dir demo_outputs
```

Alternatively, install the same pinned Python packages in an existing Python
3.9 environment:

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py --output-dir demo_outputs
```

The pinned default uses the portable CPU build of PyTorch. For full-scale
training on a GPU, install the PyTorch build matching the local CUDA driver in
place of the CPU wheel; no change to the study scripts is required.

The command generates artificial 166-region BOLD signals, applies per-region
temporal z-scoring, constructs seven-frame forecasting windows, runs a short
context-conditioned FBDM optimization and participant-specific iBDM
fine-tuning, and saves predictions and basic fitting metrics. Expected output:

```text
demo_outputs/
├── fbdm_predictions.npy
├── ibdm_predictions.npy
├── metrics.json
├── synthetic_participant_bold.csv
├── synthetic_reference_bold.csv
└── targets.npy
```

The synthetic data contain no participant information. The deliberately small
model and short optimization are solely a software smoke test; their metrics
have no scientific or clinical interpretation. See
[`examples/README.md`](examples/README.md) for the input specification.

## Framework overview

![Overview of the proposed framework](Figure/framework.png)

![Architecture of the generative brain dynamics model](Figure/model_structure.png)

![Workflow and CBM calculation](Figure/CBM.png)

The repository retains several legacy `VTB` class and file names so that the
original scripts continue to run. `GBDMTransformer` is provided as the current
manuscript-aligned alias of the same implementation.

## Study data and access

The source neuroimaging files are not mirrored in this code repository. Users
should obtain them directly from the originating repositories and follow the
applicable access and data-use conditions:

| Dataset | Access |
|---|---|
| ADNI | [ADNI data access](https://adni.loni.usc.edu/data-samples/adni-data/) |
| PPMI | [PPMI data access](https://www.ppmi-info.org/access-data-specimens/download-data) |
| ABIDE | [NITRC/INDI ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) |
| HCP Young Adult | [HCP 1200 Subjects release](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) |
| AAL3 atlas | [AAL3](https://www.gin.cnrs.fr/en/tools/aal/) |

Clinical TI and DBS data are governed by institutional privacy and ethics
requirements and are not publicly distributed through GitHub. As described in
the manuscript, de-identified preprocessed derivatives and associated clinical
assessments may be requested from the corresponding author, subject to ethical
approval and a data-sharing agreement.

## Full study workflow

The original pretraining, individualized-model, and downstream prediction
entry points are retained. They require locally obtained study data and paths
configured for the user's environment.

### 1. Preprocessing

Python preprocessing scripts and the AAL3 resources are under `Data_process/`.
The volume-based preprocessing workflow also calls external software that must
be installed separately:

- [`dcm2niix`](https://github.com/rordenlab/dcm2niix) for DICOM-to-NIfTI conversion;
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), including `bet`, `fast`,
  `flirt`, `fnirt`, `slicetimer`, `mcflirt`, `fslmaths`, and `applywarp`.

Processed input CSV files contain consecutive BOLD frames in rows and 166 AAL3
regions in columns. Configure the public-cohort paths in `vtb/config.py` or pass
the corresponding command-line arguments. `Data_process/Data_label.csv`
contains the public-cohort disease labels used by the training loader.

### 2. FBDM pretraining

```bash
python scripts/train_FVB.py --train
```

This trains the foundation forecasting model after the dataset locations have
been configured.

### 3. Participant-specific iBDMs

Set the checkpoint, input, and output paths in `batch_iVB_final.sh`, then run:

```bash
bash batch_iVB_final.sh
```

The batch script calls `scripts/train_iVB.py` to fine-tune the FBDM for each
clinical input time series and save participant-level outputs.

### 4. Response prediction

The TI and DBS downstream classifiers/regressors are under `predict_exp/`. For
example:

```bash
python predict_exp/DBS/diff/PP_diff.py
python predict_exp/DBS/diff_regression/PP_diff_regression.py
```

These scripts require the generated participant-level features and the
corresponding controlled clinical tables; update their input paths before use.

## Reproducibility scope

The synthetic demo verifies installation, the 166-region/seven-frame input
contract, the context-conditioned forecasting pass, participant-specific
fine-tuning, and output serialization. Reproducing manuscript values requires
the study datasets acquired through the routes above and the study-specific
analysis configuration. The synthetic example was not used to derive any
reported result.

## License

The code is released under the Apache License 2.0. See [`LICENSE`](LICENSE).
