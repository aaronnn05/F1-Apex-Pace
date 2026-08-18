"""
CLI entry point to execute Day 4 Optuna tuning and slice analysis.
"""

from apex_pace.models.tune import run_tuning

if __name__ == "__main__":
    run_tuning(n_trials=25)