# Predictive Semantic Cooperative Perception

Risk-aware semantic scheduling for cooperative perception in V2X networks.

## Research question

Can short-horizon motion prediction and future-risk estimation identify which cooperative-perception objects should be transmitted under a limited communication budget?

## Core idea

```text
Cooperative observations
        ↓
Object tracks
        ↓
Trajectory prediction
        ↓
Future-risk estimation
        ↓
Semantic priority score
        ↓
Bandwidth-aware scheduler
        ↓
Selected V2X messages
```

The first prototype works directly with object tracks/annotations so that the research contribution remains focused on prediction, risk and communication scheduling rather than on training a full 3D detector.

## Primary dataset

The initial benchmark is **OPV2V**, an open simulated dataset for vehicle-to-vehicle cooperative perception. Dataset files are intentionally excluded from Git. See [`datasets/README.md`](datasets/README.md) and [`scripts/download_opv2v.sh`](scripts/download_opv2v.sh).

## Baselines

- `all`: transmit every visible object.
- `nearest`: transmit the nearest objects first.
- `random`: communication-budget control baseline.
- `predictive_risk`: rank objects using predicted future risk, proximity and uncertainty.

## Metrics

- communication load and reduction,
- recall of future-critical objects,
- missed-critical-object rate,
- utility retained under a fixed message budget,
- scheduler performance across different budgets.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.demo --config configs/opv2v.yaml
pytest
```

The demo uses deterministic synthetic tracks and validates the full scheduling pipeline without requiring the large dataset download.

## Repository structure

```text
configs/            Experiment configuration
datasets/           Dataset instructions; raw data is ignored
scripts/            Dataset helpers
src/                Prediction, risk, scheduling and metrics
tests/              Unit tests
results/             Generated experiment outputs
```

## Status

Early-stage research prototype. It is not yet a complete 6G network simulator or an end-to-end cooperative 3D perception system.

## Planned work

1. Parse OPV2V object annotations and agent poses.
2. Build temporal object tracks across frames.
3. Evaluate predictive-risk scheduling against communication baselines.
4. Add packet budget, delay and loss models.
5. Validate on a real-world cooperative-perception dataset.

## Citation

When using OPV2V/OpenCOOD, cite the original OPV2V paper and follow the dataset/framework licence terms.