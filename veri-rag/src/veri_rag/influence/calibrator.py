"""Harmfulness calibrator — logistic regression on RIAA features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from veri_rag.config.schema import InfluenceScore, RiskScore


class HarmfulnessCalibrator:
    """Predict P(harmful) from risk + LOO + pairwise features."""

    def __init__(self, model_path: str | Path | None = None):
        self._model = None
        self._fitted = False
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def feature_vector(
        self,
        chunk_id: str,
        risk: RiskScore | None,
        influence: InfluenceScore | None,
        max_pairwise: float = 0.0,
    ) -> list[float]:
        r = risk
        inf = influence
        return [
            r.risk_score if r else 0.0,
            inf.influence_score if inf else 0.0,
            max_pairwise,
            r.features.instruction_score if r else 0.0,
            r.features.sensitive_score if r else 0.0,
            r.features.retrieval_anomaly_score if r else 0.0,
            r.features.conflict_score if r else 0.0,
            r.features.source_trust_score if r else 1.0,
        ]

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=500, class_weight="balanced")
        self._model.fit(np.array(X), np.array(y))
        self._fitted = True

    def predict_proba(self, features: list[float]) -> float:
        if not self._fitted or self._model is None:
            return 0.0
        prob = self._model.predict_proba(np.array([features]))[0]
        # class 1 = harmful
        if len(prob) == 1:
            return float(prob[0])
        return float(prob[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            return
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        import pickle

        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._fitted = True

    @classmethod
    def train_from_labels(
        cls,
        samples: list[dict],
        output_path: str | Path,
    ) -> "HarmfulnessCalibrator":
        """Train from list of {features, is_attack} dicts."""
        cal = cls()
        X = [s["features"] for s in samples]
        y = [int(s["is_attack"]) for s in samples]
        if len(set(y)) < 2:
            return cal
        cal.fit(X, y)
        cal.save(output_path)
        return cal
