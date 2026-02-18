"""
test_ml_models.py — Unit tests for src/ml_models.py.

Test scenarios
--------------
1. test_cost_overrun_model_trains    — artifact keys, feature count, label classes
2. test_cost_overrun_predictions     — valid probability / prediction / features
3. test_maintenance_model_trains     — artifact keys, feature count, equipment types
4. test_maintenance_predictions      — valid probability, days estimate, labels
5. test_saved_models_exist           — both .pkl files exist after training
6. test_feature_importance           — importances sum ≈ 1.0, non-negative, right count
7. test_handles_missing_data         — graceful fallback for projects/equipment with no data

All tests rely on session-scoped fixtures from conftest.py:
  • patch_model_paths  (autouse=True) — .pkl files go to a temp directory
  • db_engine          — in-memory SQLite with minimal seed data
  • trained_artifacts  — both models trained once per test session
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable when pytest is run from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ml_models
from ml_models import predict_cost_overrun, predict_maintenance_needs


# ===========================================================================
# 1. test_cost_overrun_model_trains
# ===========================================================================


def test_cost_overrun_model_trains(trained_artifacts):
    """
    train_cost_overrun_model() produces an artifact dict with all expected keys.
    The fitted RandomForest must accept the same number of features that are
    declared in feature_names, and the label encoder must know at least one
    project type.
    """
    artifact = trained_artifacts["cost_overrun"]

    required_keys = {
        "model",
        "label_encoder",
        "feature_names",
        "train_metrics",
        "test_metrics",
        "feature_importance",
        "class_labels",
    }
    missing = required_keys - artifact.keys()
    assert not missing, f"Artifact missing keys: {missing}"

    # Feature list length must match what the RandomForest was trained on
    assert len(artifact["feature_names"]) == artifact["model"].n_features_in_, (
        "feature_names length does not match model.n_features_in_"
    )

    # Label encoder must have seen at least one project type
    assert len(artifact["class_labels"]) > 0, "class_labels is empty"

    # Metrics dicts must contain the standard keys
    for split_name in ("train_metrics", "test_metrics"):
        metrics = artifact[split_name]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics, f"{split_name} missing '{key}'"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"{split_name}['{key}'] = {metrics[key]} is out of [0, 1]"
            )


# ===========================================================================
# 2. test_cost_overrun_predictions
# ===========================================================================


def test_cost_overrun_predictions(db_engine, trained_artifacts):
    """
    predict_cost_overrun() on project 9 (ACTIVE, has cost data) returns a
    well-formed result dict:
      • overrun_probability ∈ [0, 1]
      • prediction is one of the two expected labels
      • features dict matches the trained feature_names exactly
      • recommendation is a non-empty string
    """
    result = predict_cost_overrun(db_engine, 9)

    assert result["project_id"] == 9, "project_id echo mismatch"

    prob = result["overrun_probability"]
    assert 0.0 <= prob <= 1.0, f"overrun_probability {prob} out of range"

    assert result["prediction"] in ("Will Overrun", "On Track"), (
        f"Unexpected prediction label: {result['prediction']!r}"
    )

    assert isinstance(result["recommendation"], str) and result["recommendation"], (
        "recommendation must be a non-empty string"
    )

    expected_features = set(trained_artifacts["cost_overrun"]["feature_names"])
    actual_features   = set(result["features"].keys())
    assert actual_features == expected_features, (
        f"Feature key mismatch.\n  Expected: {expected_features}\n  Got: {actual_features}"
    )


# ===========================================================================
# 3. test_maintenance_model_trains
# ===========================================================================


def test_maintenance_model_trains(trained_artifacts):
    """
    train_equipment_maintenance_model() produces an artifact dict with all
    expected keys.  Same structural checks as the cost overrun model.
    """
    artifact = trained_artifacts["maintenance"]

    required_keys = {
        "model",
        "label_encoder",
        "feature_names",
        "train_metrics",
        "test_metrics",
        "feature_importance",
        "equipment_types",
    }
    missing = required_keys - artifact.keys()
    assert not missing, f"Artifact missing keys: {missing}"

    assert len(artifact["feature_names"]) == artifact["model"].n_features_in_, (
        "feature_names length does not match model.n_features_in_"
    )

    assert len(artifact["equipment_types"]) > 0, "equipment_types is empty"

    for split_name in ("train_metrics", "test_metrics"):
        metrics = artifact[split_name]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics, f"{split_name} missing '{key}'"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"{split_name}['{key}'] = {metrics[key]} is out of [0, 1]"
            )


# ===========================================================================
# 4. test_maintenance_predictions
# ===========================================================================


def test_maintenance_predictions(db_engine, trained_artifacts):
    """
    predict_maintenance_needs() on equipment 1 (EXCAVATOR with 365 days of
    usage data) returns a well-formed result dict:
      • maintenance_probability ∈ [0, 1]
      • predicted_days_until_maintenance ≥ 1
      • prediction is one of the two expected labels
      • features dict matches the trained feature_names exactly
    """
    result = predict_maintenance_needs(db_engine, 1)

    assert result["equipment_id"] == 1, "equipment_id echo mismatch"

    prob = result["maintenance_probability"]
    assert 0.0 <= prob <= 1.0, f"maintenance_probability {prob} out of range"

    days = result["predicted_days_until_maintenance"]
    assert days >= 1, f"predicted_days_until_maintenance must be ≥ 1, got {days}"

    assert result["prediction"] in ("Maintenance Needed", "No Immediate Maintenance"), (
        f"Unexpected prediction label: {result['prediction']!r}"
    )

    assert isinstance(result["recommendation"], str) and result["recommendation"], (
        "recommendation must be a non-empty string"
    )

    expected_features = set(trained_artifacts["maintenance"]["feature_names"])
    actual_features   = set(result["features"].keys())
    assert actual_features == expected_features, (
        f"Feature key mismatch.\n  Expected: {expected_features}\n  Got: {actual_features}"
    )


# ===========================================================================
# 5. test_saved_models_exist
# ===========================================================================


def test_saved_models_exist(trained_artifacts):
    """
    After training both models, the .pkl files must exist at the (patched)
    model paths.  This confirms joblib.dump() ran successfully and that the
    predict_* functions will be able to load them.
    """
    cost_path  = ml_models.COST_OVERRUN_MODEL_PATH
    maint_path = ml_models.EQUIPMENT_MAINTENANCE_MODEL_PATH

    assert cost_path.exists(), (
        f"Cost overrun model not found at: {cost_path}"
    )
    assert maint_path.exists(), (
        f"Equipment maintenance model not found at: {maint_path}"
    )
    # Basic sanity: files must not be empty
    assert cost_path.stat().st_size > 0,  "Cost overrun .pkl file is empty"
    assert maint_path.stat().st_size > 0, "Maintenance .pkl file is empty"


# ===========================================================================
# 6. test_feature_importance
# ===========================================================================


def test_feature_importance(trained_artifacts):
    """
    For both models:
      • feature_importance DataFrame has one row per declared feature
      • importances are all non-negative
      • importances sum to approximately 1.0 (RandomForest guarantee)
      • DataFrame is sorted descending by importance
    """
    for model_key in ("cost_overrun", "maintenance"):
        artifact = trained_artifacts[model_key]
        fi_df    = artifact["feature_importance"]
        n_feats  = len(artifact["feature_names"])

        assert len(fi_df) == n_feats, (
            f"[{model_key}] Expected {n_feats} importance rows, got {len(fi_df)}"
        )

        assert (fi_df["importance"] >= 0).all(), (
            f"[{model_key}] Negative feature importance found"
        )

        total = fi_df["importance"].sum()
        assert abs(total - 1.0) < 0.01, (
            f"[{model_key}] Feature importances sum to {total:.6f}, expected ≈ 1.0"
        )

        # Verify descending order
        sorted_vals = fi_df["importance"].tolist()
        assert sorted_vals == sorted(sorted_vals, reverse=True), (
            f"[{model_key}] feature_importance is not sorted descending"
        )


# ===========================================================================
# 7. test_handles_missing_data
# ===========================================================================


def test_handles_missing_data(db_engine, trained_artifacts):
    """
    Both predict_* functions must return valid predictions when the input
    project / equipment has no cost or usage records:

    Cost overrun  — project 10 has no fact_project_costs rows.
                    The function should default early_phase_cpi to 1.0 and
                    still return a probability in [0, 1].

    Maintenance   — equipment 5 (FORKLIFT) has no fact_equipment_usage rows.
                    The function should default all features to 0 / neutral
                    values and still return a probability in [0, 1].
    """
    # ── Cost overrun: project with no cost data ────────────────────────────
    cost_result = predict_cost_overrun(db_engine, 10)

    cost_prob = cost_result["overrun_probability"]
    assert 0.0 <= cost_prob <= 1.0, (
        f"Cost probability {cost_prob} out of range for project with no data"
    )
    assert cost_result["features"]["early_phase_cpi"] == 1.0, (
        "early_phase_cpi should default to 1.0 when no cost data exists"
    )

    # ── Maintenance: equipment with no usage data ──────────────────────────
    maint_result = predict_maintenance_needs(db_engine, 5)

    maint_prob = maint_result["maintenance_probability"]
    assert 0.0 <= maint_prob <= 1.0, (
        f"Maintenance probability {maint_prob} out of range for equipment with no data"
    )
    assert maint_result["predicted_days_until_maintenance"] >= 1, (
        "predicted_days_until_maintenance must be ≥ 1 even with no usage data"
    )
    # All numeric features should have sensible defaults (≥ 0)
    for feat, val in maint_result["features"].items():
        assert val >= 0, (
            f"Feature '{feat}' has unexpected negative default value: {val}"
        )
