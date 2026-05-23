"""
Data pipeline runner — executes all data loading and cleaning steps in order.

Usage:
  python -m src.data.run_phase1              # full run
  python -m src.data.run_phase1 --skip-api   # skip live API sync (no key needed)
"""

import sys

from src.data.database import init_db
from src.data.load_kaggle import run as run_kaggle
from src.data.clean_unify import run as run_clean

SKIP_API = "--skip-api" in sys.argv


def main():
    print("\n" + "=" * 55)
    print("  Data Pipeline")
    print("=" * 55 + "\n")

    print("[1/3] Initialise database")
    init_db()

    print("\n[2/3] Load Kaggle historical data")
    run_kaggle()

    if not SKIP_API:
        print("\n[3a] Live API sync")
        try:
            from src.data.api_client import run_sync
            run_sync()
        except RuntimeError as e:
            print(f"  Skipping API sync: {e}")
    else:
        print("\n[3a] Live API sync — Skipped (--skip-api)")

    print("\n[3b] Clean and unify")
    run_clean()

    print("\n" + "=" * 55)
    print("  Data pipeline complete.")
    print("  Next: run src.features.run_phase2 for feature engineering.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
