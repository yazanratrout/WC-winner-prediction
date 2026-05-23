"""
Feature engineering runner — runs all steps in order.

Usage:
  python -m src.features.run_phase2
"""

from src.features.team_features import run as run_21
from src.features.player_features import run as run_22
from src.features.chemistry_features import run as run_23
from src.features.tactical_features import run as run_24
from src.features.build_features import run as run_25


def main() -> None:
    print("=" * 60)
    print("PHASE 2 — FEATURE ENGINEERING")
    print("=" * 60)
    run_21()
    print()
    run_22()
    print()
    run_23()
    print()
    run_24()
    print()
    run_25()
    print()
    print("=" * 60)
    print("Feature engineering complete — features.parquet ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
