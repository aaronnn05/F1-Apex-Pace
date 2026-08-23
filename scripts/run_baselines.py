"""CLI entry point to execute Day 3 model training workflow."""

from apex_pace.models.baseline import mean_baseline, previous_lap_baseline, train_baseline_model

if __name__ == "__main__":
    mean_baseline()
    previous_lap_baseline()
    train_baseline_model()