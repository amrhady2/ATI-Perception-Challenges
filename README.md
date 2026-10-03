# Automotive perception experiments

Research and take-home challenge experiments by Amr Attia. **Status: notebook-based prototype.**

## Wheel-center localization

The final notebook uses a PyTorch `UNetResNet18Multi` model with heatmap and wheel-presence outputs. It includes training, evaluation, and visualization routines. Earlier approaches are retained under `Challenge 1 - Rim Center Location/Old/`.

Start with the [final heatmap notebook](Challenge%201%20-%20Rim%20Center%20Location/Final/Heatmaps_Beat_Direct_Regression.ipynb). Review its dataset paths, configuration, and split before running cells.

## Tread-depth exploration

The notebooks in [Challenge 2 - Tread Depth](Challenge%202%20-%20Tread%20Depth/) explore laser-line detection, pixel-to-distance calibration, video processing, and synthetic profiles. These are experiments; physical accuracy depends on calibration and has not been independently established.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

On Windows, activate with `.venv\Scripts\activate`. Select the final notebook and set its working directory and data paths explicitly. PyTorch hardware support depends on your platform; the dependency list is not a validated lockfile.

## Repository layout

```text
Challenge 1 - Rim Center Location/Final/    Final notebooks and evaluation artifacts
Challenge 1 - Rim Center Location/Old/      Earlier experimental approaches
Challenge 2 - Tread Depth/  Tread-depth notebooks
```

## Evaluation and limitations

Inspect the notebook’s pixel-error, wheel-presence, and overlay outputs with the associated dataset and split. Saved outputs are historical artifacts, not a fresh benchmark. Real-time performance, robustness, and physical measurement accuracy require evaluation on the intended hardware and data.

The repository contains substantial image and experiment artifacts, so a full clone may be large. Data and model redistribution rights should be verified before reuse.
