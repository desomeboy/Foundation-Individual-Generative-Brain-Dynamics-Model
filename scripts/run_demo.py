#!/usr/bin/env python3
"""Self-contained smoke test for the GBDM-to-iBDM forecasting workflow.

The generated signals are synthetic and the deliberately short optimization is
only intended to verify the public software interface. Demo metrics have no
scientific or clinical interpretation.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vtb.data import multi2one  # noqa: E402
from vtb.models import GBDMTransformer  # noqa: E402


ROI_COUNT = 166
HISTORY_FRAMES = 7
SKIP_FIRST = 30
DEMOGRAPHIC_DIM = 56


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_bold(
    timepoints: int,
    seed: int,
    participant_shift: float,
) -> np.ndarray:
    """Generate correlated AR signals with no relation to any study subject."""
    rng = np.random.default_rng(seed)
    latent_count = 10
    latent = np.zeros((timepoints, latent_count), dtype=np.float32)
    latent_ar = np.linspace(0.55, 0.88, latent_count, dtype=np.float32)
    for t in range(1, timepoints):
        latent[t] = (
            latent_ar * latent[t - 1]
            + rng.normal(0.0, 0.45, latent_count).astype(np.float32)
        )

    loadings = rng.normal(0.0, 0.35, (latent_count, ROI_COUNT)).astype(np.float32)
    if participant_shift:
        loadings[:3, :40] += participant_shift

    bold = latent @ loadings
    regional_ar = np.zeros_like(bold)
    for t in range(1, timepoints):
        regional_ar[t] = 0.45 * regional_ar[t - 1] + bold[t]
    regional_ar += rng.normal(0.0, 0.30, regional_ar.shape).astype(np.float32)
    return regional_ar


def save_bold_csv(path: Path, bold: np.ndarray) -> None:
    header = ",".join("ROI_{:03d}".format(i + 1) for i in range(ROI_COUNT))
    np.savetxt(path, bold, delimiter=",", header=header, comments="", fmt="%.7f")


def load_and_window(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    bold = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if bold.ndim != 2 or bold.shape[1] != ROI_COUNT:
        raise ValueError(
            "Expected a timepoints x {} CSV, received {}".format(ROI_COUNT, bold.shape)
        )
    if bold.shape[0] <= SKIP_FIRST + HISTORY_FRAMES:
        raise ValueError("The example time series is too short to construct windows.")
    if not np.isfinite(bold).all():
        raise ValueError("The input contains non-finite values.")

    mean = bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    standardized = (bold - mean) / std
    inputs, targets = multi2one(
        standardized,
        steps=HISTORY_FRAMES,
        skip_first=SKIP_FIRST,
    )
    return inputs.astype(np.float32), targets.astype(np.float32)


def context_arrays(
    sample_count: int,
    disease_label: int,
    demographic_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.full(sample_count, disease_label, dtype=np.int64)
    demographics = np.repeat(
        demographic_vector.reshape(1, -1), sample_count, axis=0
    ).astype(np.float32)
    return labels, demographics


def optimize(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    demographics: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Tuple[torch.nn.Module, list]:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.MSELoss()
    losses = []

    x = torch.from_numpy(inputs).to(device)
    y = torch.from_numpy(targets).to(device)
    label_tensor = torch.from_numpy(labels).to(device)
    demographic_tensor = torch.from_numpy(demographics).to(device)

    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, x.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x[index],
                label_tensor[index],
                Demographic=demographic_tensor[index],
            )
            loss = criterion(prediction, y[index])
            loss.backward()
            optimizer.step()
            count = int(index.numel())
            epoch_loss += float(loss.detach().cpu()) * count
            seen += count
        losses.append(epoch_loss / seen)
    return model, losses


def predict(
    model: torch.nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    demographics: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(labels).to(device),
            Demographic=torch.from_numpy(demographics).to(device),
        )
    return output.detach().cpu().numpy()


def fc_correlation(targets: np.ndarray, predictions: np.ndarray) -> float:
    target_fc = np.corrcoef(targets, rowvar=False)
    prediction_fc = np.corrcoef(predictions, rowvar=False)
    upper = np.triu_indices(ROI_COUNT, k=1)
    return float(np.corrcoef(target_fc[upper], prediction_fc[upper])[0, 1])


def summarize(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
        "fc_correlation": fc_correlation(targets, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a data-independent GBDM/iBDM software smoke test."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timepoints", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.timepoints <= SKIP_FIRST + HISTORY_FRAMES + 10:
        raise ValueError("Use at least 48 synthetic timepoints for this demo.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.output_dir / "synthetic_reference_bold.csv"
    participant_path = args.output_dir / "synthetic_participant_bold.csv"
    save_bold_csv(
        reference_path,
        generate_synthetic_bold(args.timepoints, args.seed, participant_shift=0.0),
    )
    save_bold_csv(
        participant_path,
        generate_synthetic_bold(args.timepoints, args.seed, participant_shift=0.12),
    )

    reference_inputs, reference_targets = load_and_window(reference_path)
    participant_inputs, participant_targets = load_and_window(participant_path)
    split = max(1, int(0.70 * participant_inputs.shape[0]))

    reference_demographics = np.zeros(DEMOGRAPHIC_DIM, dtype=np.float32)
    participant_demographics = np.linspace(
        -0.25, 0.25, DEMOGRAPHIC_DIM, dtype=np.float32
    )
    reference_labels, reference_context = context_arrays(
        reference_inputs.shape[0], 0, reference_demographics
    )
    participant_labels, participant_context = context_arrays(
        participant_inputs.shape[0], 1, participant_demographics
    )

    if args.d_model % 4 != 0:
        raise ValueError("--d-model must be divisible by four.")
    fbdm = GBDMTransformer(
        input_dim=HISTORY_FRAMES * ROI_COUNT,
        steps=HISTORY_FRAMES,
        roi_num=ROI_COUNT,
        d_model=args.d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=2 * args.d_model,
        dropout=0.0,
        use_layernorm=True,
        use_last_token=False,
        num_labels=4,
        num_cross_layers=1,
        demographic_dim=DEMOGRAPHIC_DIM,
    ).to(device)

    fbdm, fbdm_losses = optimize(
        fbdm,
        reference_inputs,
        reference_targets,
        reference_labels,
        reference_context,
        device,
        args.epochs,
        learning_rate=5e-4,
        batch_size=args.batch_size,
    )

    test_inputs = participant_inputs[split:]
    test_targets = participant_targets[split:]
    test_labels = participant_labels[split:]
    test_context = participant_context[split:]
    fbdm_predictions = predict(
        fbdm, test_inputs, test_labels, test_context, device
    )

    ibdm = copy.deepcopy(fbdm)
    ibdm, ibdm_losses = optimize(
        ibdm,
        participant_inputs[:split],
        participant_targets[:split],
        participant_labels[:split],
        participant_context[:split],
        device,
        args.fine_tune_epochs,
        learning_rate=1e-4,
        batch_size=args.batch_size,
    )
    ibdm_predictions = predict(
        ibdm, test_inputs, test_labels, test_context, device
    )

    np.save(args.output_dir / "targets.npy", test_targets)
    np.save(args.output_dir / "fbdm_predictions.npy", fbdm_predictions)
    np.save(args.output_dir / "ibdm_predictions.npy", ibdm_predictions)

    report = {
        "purpose": "Software smoke test only; no scientific interpretation.",
        "configuration": {
            "seed": args.seed,
            "device": str(device),
            "timepoints": args.timepoints,
            "history_frames": HISTORY_FRAMES,
            "roi_count": ROI_COUNT,
            "discarded_initial_frames": SKIP_FIRST,
            "d_model": args.d_model,
            "fbdm_epochs": args.epochs,
            "ibdm_fine_tune_epochs": args.fine_tune_epochs,
        },
        "output_shape": list(test_targets.shape),
        "all_outputs_finite": bool(
            np.isfinite(test_targets).all()
            and np.isfinite(fbdm_predictions).all()
            and np.isfinite(ibdm_predictions).all()
        ),
        "training_loss": {
            "fbdm": fbdm_losses,
            "ibdm": ibdm_losses,
        },
        "metrics": {
            "fbdm": summarize(test_targets, fbdm_predictions),
            "ibdm": summarize(test_targets, ibdm_predictions),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("GBDM/iBDM synthetic smoke test completed successfully.")
    print("Output shape: {}".format(tuple(test_targets.shape)))
    print("All outputs finite: {}".format(report["all_outputs_finite"]))
    print("Metrics: {}".format(metrics_path))


if __name__ == "__main__":
    main()
