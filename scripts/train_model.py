"""CLI entry point to execute Day 3 model training workflow."""

from apex_pace.models.train import train_and_log

if __name__ == "__main__":
    train_and_log()