"""
Match predictor wrapper — builds feature vectors for any two teams
and returns W/D/L probabilities using the saved model.

Used by the simulation engine to get match probabilities without
re-loading the model on every call.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/match_predictor.pkl")

# Competition type encoding for WC group stage and knockout
COMP_ENC_WC = 5  # world_cup


class MatchPredictor:
    def __init__(self):
        payload = joblib.load(MODEL_PATH)
        self.model       = payload["model"]
        self.scaler      = payload["scaler"]
        self.feature_cols = payload["feature_cols"]
        self.classes     = payload["classes"]  # ['D', 'L', 'W']
        # Index positions
        self._d = self.classes.index("D")
        self._l = self.classes.index("L")
        self._w = self.classes.index("W")

    def predict(
        self,
        team_feats_a: dict,
        team_feats_b: dict,
        elo_a: float,
        elo_b: float,
        neutral: bool = True,
        comp_enc: int = COMP_ENC_WC,
    ) -> dict[str, float]:
        """
        Return {"W": p_w, "D": p_d, "L": p_l} from team A's perspective.

        team_feats_a / team_feats_b: dicts of raw (non-diff) feature values.
        elo_a / elo_b: current in-tournament Elo ratings (updated after each match).
        """
        row = {}
        # Build differential features
        for col in self.feature_cols:
            if col.startswith("diff_"):
                feat = col[5:]  # strip "diff_"
                if feat == "elo":
                    row[col] = elo_a - elo_b
                elif feat == "best_rank":
                    row[col] = team_feats_a.get("best_rank", 100) - team_feats_b.get("best_rank", 100)
                else:
                    row[col] = team_feats_a.get(feat, 0.0) - team_feats_b.get(feat, 0.0)
            elif col.startswith("abs_diff_"):
                feat = col[9:]  # strip "abs_diff_"
                if feat == "elo":
                    row[col] = abs(elo_a - elo_b)
                else:
                    row[col] = abs(team_feats_a.get(feat, 0.0) - team_feats_b.get(feat, 0.0))
            elif col == "neutral":
                row[col] = int(neutral)
            elif col == "comp_type_enc":
                row[col] = comp_enc

        X = np.array([[row.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        X_scaled = self.scaler.transform(X)

        # Handle XGBoost label encoding
        le = getattr(self.model, "_le", None)
        proba = self.model.predict_proba(X_scaled)[0]

        if le is not None:
            # proba columns are in encoded integer order (D=0, L=1, W=2 after sort)
            classes_enc = le.classes_  # original string labels in encoded order
            prob_map = dict(zip(classes_enc, proba))
        else:
            prob_map = dict(zip(self.classes, proba))

        return {
            "W": float(prob_map.get("W", proba[self._w])),
            "D": float(prob_map.get("D", proba[self._d])),
            "L": float(prob_map.get("L", proba[self._l])),
        }
