"""
ml_models.py — Machine learning models for the Construction Analytics Platform.

MODEL 1: Cost Overrun Prediction (RandomForestClassifier)
  Predicts whether a project will overrun its budget by >10%, using project
  metadata and early-phase cost performance as features.

MODEL 2: Equipment Maintenance Prediction (RandomForestClassifier)
  Predicts whether a piece of equipment will need maintenance within the
  next 30 days, using rolling usage and maintenance history features.

Usage:
    python src/ml_models.py
"""

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
from database import get_engine

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

MODELS_DIR                    = Path(__file__).resolve().parent.parent / "models"
COST_OVERRUN_MODEL_PATH       = MODELS_DIR / "cost_overrun_model.pkl"
EQUIPMENT_MAINTENANCE_MODEL_PATH = MODELS_DIR / "equipment_maintenance_model.pkl"
TODAY = date(2026, 2, 17)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _query(engine, sql: str, params: dict = None) -> pd.DataFrame:
    """Execute *sql* with optional bound *params* and return a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def _metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return a dict of standard binary classification metrics."""
    out = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            out["roc_auc"] = None
    return out


# ===========================================================================
# MODEL 1 — Cost Overrun Prediction
# ===========================================================================


def _build_cost_overrun_dataset(engine) -> pd.DataFrame:
    """
    Pull completed projects and engineer features for cost overrun prediction.

    Features
    --------
    project_type_enc  — label-encoded project type
    contract_value    — contract value ($)
    planned_duration  — planned duration in calendar days
    square_footage    — building size (sq ft); 0 when missing
    early_phase_cpi   — CPI during first 25% of the planned timeline

    Target
    ------
    will_overrun — 1 if final budget variance > 10%, else 0
    """
    # ── Fetch completed project metadata ───────────────────────────────────
    projects = _query(engine, """
        SELECT
            p.id,
            p.project_type,
            p.contract_value,
            COALESCE(p.square_footage, 0)                                 AS square_footage,
            CAST(
                julianday(p.planned_end_date)
                - julianday(p.planned_start_date)
            AS INTEGER)                                                    AS planned_duration,
            CAST(
                0.25 * (
                    julianday(p.planned_end_date)
                    - julianday(p.planned_start_date)
                )
            AS INTEGER)                                                    AS early_phase_days,
            p.actual_start_date
        FROM dim_projects p
        WHERE p.status            = 'COMPLETED'
          AND p.actual_start_date IS NOT NULL
          AND p.actual_end_date   IS NOT NULL
          AND p.planned_start_date IS NOT NULL
          AND p.planned_end_date   IS NOT NULL
    """)

    if projects.empty:
        raise ValueError("No completed projects found. Run data_generator.py first.")

    # ── Fetch all cost records for those projects in a single query ────────
    ids_placeholder = ", ".join([f":p{i}" for i in range(len(projects))])
    id_params = {f"p{i}": int(pid) for i, pid in enumerate(projects["id"])}

    costs = _query(engine, f"""
        SELECT project_id, date_id, amount, budgeted_amount
        FROM   fact_project_costs
        WHERE  project_id IN ({ids_placeholder})
    """, id_params)

    if costs.empty:
        raise ValueError("No cost data found for completed projects.")

    costs["date_id"] = pd.to_datetime(costs["date_id"])

    # ── Feature engineering (per project) ─────────────────────────────────
    rows = []
    for _, proj in projects.iterrows():
        pid = int(proj["id"])
        proj_costs = costs[costs["project_id"] == pid]

        total_actual   = proj_costs["amount"].sum()
        total_budgeted = proj_costs["budgeted_amount"].sum()
        if total_budgeted == 0:
            continue

        variance_pct = 100.0 * (total_actual - total_budgeted) / total_budgeted
        will_overrun = 1 if variance_pct > 10.0 else 0

        # Early-phase CPI: cost performance in first 25% of planned timeline
        cutoff = (
            pd.to_datetime(proj["actual_start_date"])
            + timedelta(days=max(int(proj["early_phase_days"]), 1))
        )
        early = proj_costs[proj_costs["date_id"] <= cutoff]
        e_actual   = early["amount"].sum()
        e_budgeted = early["budgeted_amount"].sum()
        early_cpi  = (e_budgeted / e_actual) if e_actual > 0 else 1.0

        rows.append({
            "project_id":       pid,
            "project_type":     proj["project_type"],
            "contract_value":   float(proj["contract_value"]),
            "planned_duration": max(int(proj["planned_duration"]), 1),
            "square_footage":   float(proj["square_footage"]),
            "early_phase_cpi":  round(early_cpi, 4),
            "variance_pct":     round(variance_pct, 2),
            "will_overrun":     will_overrun,
        })

    return pd.DataFrame(rows)


def train_cost_overrun_model(engine) -> dict:
    """
    Train a RandomForestClassifier to predict budget overruns (>10%) for
    construction projects, using only features available early in the project.

    Returns
    -------
    dict with keys:
        model              — fitted RandomForestClassifier
        label_encoder      — fitted LabelEncoder for project_type
        feature_names      — ordered list of feature column names
        train_metrics      — accuracy, precision, recall, F1, ROC-AUC on train set
        test_metrics       — same metrics on held-out test set
        feature_importance — DataFrame sorted by importance descending
        class_labels       — list of project type strings known to the encoder
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Building cost overrun dataset...")
    df = _build_cost_overrun_dataset(engine)
    n_over = int(df["will_overrun"].sum())
    print(
        f"  Samples: {len(df):,}  |  overruns: {n_over} ({n_over/len(df):.1%})  |  "
        f"on-track: {len(df)-n_over} ({(len(df)-n_over)/len(df):.1%})"
    )

    # ── Feature encoding ───────────────────────────────────────────────────
    le = LabelEncoder()
    df["project_type_enc"] = le.fit_transform(df["project_type"])

    FEATURES = [
        "project_type_enc",
        "contract_value",
        "planned_duration",
        "square_footage",
        "early_phase_cpi",
    ]

    X = df[FEATURES].fillna(0)
    y = df["will_overrun"]

    # Guard against tiny datasets
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # ── Train ──────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Metrics ────────────────────────────────────────────────────────────
    train_metrics = _metrics(
        y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1]
    )
    test_metrics = _metrics(
        y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]
    )

    # ── Feature importance ─────────────────────────────────────────────────
    fi_df = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["importance"] = fi_df["importance"].round(4)

    # ── Persist ────────────────────────────────────────────────────────────
    artifact = {
        "model":              model,
        "label_encoder":      le,
        "feature_names":      FEATURES,
        "train_metrics":      train_metrics,
        "test_metrics":       test_metrics,
        "feature_importance": fi_df,
        "class_labels":       le.classes_.tolist(),
    }
    joblib.dump(artifact, COST_OVERRUN_MODEL_PATH)
    print(f"  Model saved → {COST_OVERRUN_MODEL_PATH}")

    return artifact


def predict_cost_overrun(engine, project_id: int) -> dict:
    """
    Predict cost overrun probability for a single project.

    Parameters
    ----------
    engine      — SQLAlchemy engine
    project_id  — dim_projects.id

    Returns
    -------
    dict with keys:
        project_id          — echo of input
        project_name        — project display name
        overrun_probability — float 0–1 (probability of >10% overrun)
        prediction          — "Will Overrun" | "On Track"
        recommendation      — actionable guidance string
        features            — dict of feature values used for this prediction
    """
    if not COST_OVERRUN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {COST_OVERRUN_MODEL_PATH}. "
            "Run train_cost_overrun_model() first."
        )

    artifact = joblib.load(COST_OVERRUN_MODEL_PATH)
    model    = artifact["model"]
    le       = artifact["label_encoder"]

    # Fetch project
    proj_df = _query(engine, """
        SELECT
            p.id,
            p.project_name,
            p.project_type,
            p.contract_value,
            COALESCE(p.square_footage, 0) AS square_footage,
            CAST(
                julianday(COALESCE(p.planned_end_date,
                    date(p.planned_start_date, '+365 days')))
                - julianday(p.planned_start_date)
            AS INTEGER) AS planned_duration,
            CAST(
                0.25 * (
                    julianday(COALESCE(p.planned_end_date,
                        date(p.planned_start_date, '+365 days')))
                    - julianday(p.planned_start_date)
                )
            AS INTEGER) AS early_phase_days,
            p.actual_start_date
        FROM dim_projects p
        WHERE p.id = :project_id
    """, {"project_id": project_id})

    if proj_df.empty:
        raise ValueError(f"Project {project_id} not found.")

    proj = proj_df.iloc[0]

    # Early-phase CPI from actual cost data
    costs = _query(engine, """
        SELECT date_id, amount, budgeted_amount
        FROM   fact_project_costs
        WHERE  project_id = :project_id
    """, {"project_id": project_id})

    if not costs.empty and proj["actual_start_date"]:
        costs["date_id"] = pd.to_datetime(costs["date_id"])
        cutoff = (
            pd.to_datetime(proj["actual_start_date"])
            + timedelta(days=max(int(proj["early_phase_days"]), 1))
        )
        early = costs[costs["date_id"] <= cutoff]
        e_actual   = early["amount"].sum()
        e_budgeted = early["budgeted_amount"].sum()
        early_cpi  = (e_budgeted / e_actual) if e_actual > 0 else 1.0
    else:
        early_cpi = 1.0

    # Encode project type (gracefully handle unseen labels)
    try:
        project_type_enc = int(le.transform([proj["project_type"]])[0])
    except ValueError:
        project_type_enc = 0

    features = {
        "project_type_enc": project_type_enc,
        "contract_value":   float(proj["contract_value"]),
        "planned_duration": max(int(proj["planned_duration"]), 1),
        "square_footage":   float(proj["square_footage"]),
        "early_phase_cpi":  round(early_cpi, 4),
    }

    X    = pd.DataFrame([features])[artifact["feature_names"]]
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    if prob >= 0.70:
        recommendation = (
            "High overrun risk. Immediately review budget allocations, enforce "
            "strict change-order controls, and consider value-engineering options."
        )
    elif prob >= 0.45:
        recommendation = (
            "Moderate overrun risk. Monitor cost categories weekly, activate "
            "contingency reserves, and flag any scope creep early."
        )
    else:
        recommendation = (
            "Low overrun risk. Maintain current cost controls and continue "
            "regular budget reviews through project completion."
        )

    return {
        "project_id":          project_id,
        "project_name":        str(proj["project_name"]),
        "overrun_probability": round(prob, 4),
        "prediction":          "Will Overrun" if pred == 1 else "On Track",
        "recommendation":      recommendation,
        "features":            features,
    }


# ===========================================================================
# MODEL 2 — Equipment Maintenance Prediction
# ===========================================================================


def _build_equipment_maintenance_dataset(engine) -> pd.DataFrame:
    """
    Build a training dataset for equipment maintenance prediction.

    One row per (equipment × month-end snapshot). Features are computed from
    usage history up to the snapshot date; the binary target is whether the
    equipment had at least one maintenance event in the following 30 days.

    Features
    --------
    equipment_type_enc        — label-encoded equipment type
    age_days                  — days since purchase at snapshot date
    total_hours_used          — cumulative hours operated
    avg_hours_per_day         — total_hours / age_days
    fuel_cost_per_hour        — cumulative fuel / cumulative hours
    recent_fuel_cost_trend    — fuel cost last 90 days ÷ prior 90 days
    maintenance_event_count   — number of maintenance events in history
    days_since_maintenance    — days since most recent maintenance (age if never)
    avg_maint_cost_per_event  — mean maintenance cost per event

    Target
    ------
    needs_maintenance_soon — 1 if maintenance_cost > 0 in next 30 days
    """
    # Fetch equipment dimension
    equipment = _query(engine, """
        SELECT id, equipment_type, purchase_date, status
        FROM   dim_equipment
        WHERE  purchase_date IS NOT NULL
    """)

    # Fetch all usage in one shot
    usage = _query(engine, """
        SELECT
            equipment_id,
            date_id,
            hours_used,
            COALESCE(fuel_cost, 0)        AS fuel_cost,
            COALESCE(maintenance_cost, 0) AS maintenance_cost
        FROM fact_equipment_usage
        ORDER BY equipment_id, date_id
    """)

    if equipment.empty or usage.empty:
        raise ValueError(
            "No equipment usage data found. Run data_generator.py first."
        )

    usage["date_id"]           = pd.to_datetime(usage["date_id"])
    equipment["purchase_date"] = pd.to_datetime(equipment["purchase_date"])

    rows = []
    for _, eq in equipment.iterrows():
        eid   = int(eq["id"])
        etype = eq["equipment_type"]
        pdate = eq["purchase_date"]

        eq_usage = usage[usage["equipment_id"] == eid].sort_values("date_id")
        if len(eq_usage) < 5:
            continue

        min_date = eq_usage["date_id"].min()
        max_date = eq_usage["date_id"].max()

        # Monthly snapshot dates — need at least 30 days of future data each
        snapshot_dates = pd.date_range(
            start=min_date + pd.DateOffset(months=2),
            end=max_date  - pd.Timedelta(days=30),
            freq="MS",
        )

        for snap in snapshot_dates:
            history   = eq_usage[eq_usage["date_id"] < snap]
            future_30 = eq_usage[
                (eq_usage["date_id"] >= snap)
                & (eq_usage["date_id"] <  snap + pd.Timedelta(days=30))
            ]

            if history.empty:
                continue

            age_days      = max((snap - pdate).days, 1)
            total_hours   = float(history["hours_used"].sum())
            avg_hrs_day   = total_hours / age_days

            fuel_per_hour = (
                history["fuel_cost"].sum() / total_hours
                if total_hours > 0 else 0.0
            )

            # Recent fuel trend: last 90 days vs prior 90 days
            r_cut = snap - pd.Timedelta(days=90)
            o_cut = snap - pd.Timedelta(days=180)
            recent_fuel = float(history[history["date_id"] >= r_cut]["fuel_cost"].sum())
            older_fuel  = float(
                history[
                    (history["date_id"] >= o_cut) & (history["date_id"] < r_cut)
                ]["fuel_cost"].sum()
            )
            fuel_trend = (recent_fuel / older_fuel) if older_fuel > 0 else 1.0

            # Maintenance history
            maint = history[history["maintenance_cost"] > 0]
            maint_count = len(maint)
            if not maint.empty:
                last_maint_date     = maint["date_id"].max()
                days_since_maint    = (snap - last_maint_date).days
                avg_maint_cost      = float(maint["maintenance_cost"].mean())
            else:
                days_since_maint = age_days
                avg_maint_cost   = 0.0

            # Target
            target = int((future_30["maintenance_cost"] > 0).any())

            rows.append({
                "equipment_id":             eid,
                "equipment_type":           etype,
                "age_days":                 age_days,
                "total_hours_used":         round(total_hours, 1),
                "avg_hours_per_day":        round(avg_hrs_day, 3),
                "fuel_cost_per_hour":       round(fuel_per_hour, 4),
                "recent_fuel_cost_trend":   round(min(fuel_trend, 10.0), 4),
                "maintenance_event_count":  maint_count,
                "days_since_maintenance":   days_since_maint,
                "avg_maint_cost_per_event": round(avg_maint_cost, 2),
                "needs_maintenance_soon":   target,
            })

    return pd.DataFrame(rows)


def train_equipment_maintenance_model(engine) -> dict:
    """
    Train a RandomForestClassifier to predict whether equipment will need
    maintenance within the next 30 days.

    Returns
    -------
    dict with keys:
        model              — fitted RandomForestClassifier
        label_encoder      — fitted LabelEncoder for equipment_type
        feature_names      — ordered list of feature column names
        train_metrics      — accuracy, precision, recall, F1, ROC-AUC on train set
        test_metrics       — same metrics on held-out test set
        feature_importance — DataFrame sorted by importance descending
        equipment_types    — list of equipment type strings known to the encoder
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("  Building equipment maintenance dataset...")
    df = _build_equipment_maintenance_dataset(engine)
    n_maint = int(df["needs_maintenance_soon"].sum())
    print(
        f"  Samples: {len(df):,}  |  maintenance: {n_maint} ({n_maint/len(df):.1%})  |  "
        f"no-maintenance: {len(df)-n_maint} ({(len(df)-n_maint)/len(df):.1%})"
    )

    # ── Feature encoding ───────────────────────────────────────────────────
    le = LabelEncoder()
    df["equipment_type_enc"] = le.fit_transform(df["equipment_type"])

    FEATURES = [
        "equipment_type_enc",
        "age_days",
        "total_hours_used",
        "avg_hours_per_day",
        "fuel_cost_per_hour",
        "recent_fuel_cost_trend",
        "maintenance_event_count",
        "days_since_maintenance",
        "avg_maint_cost_per_event",
    ]

    X = df[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = df["needs_maintenance_soon"]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # ── Train ──────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Metrics ────────────────────────────────────────────────────────────
    train_metrics = _metrics(
        y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1]
    )
    test_metrics = _metrics(
        y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]
    )

    # ── Feature importance ─────────────────────────────────────────────────
    fi_df = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["importance"] = fi_df["importance"].round(4)

    # ── Persist ────────────────────────────────────────────────────────────
    artifact = {
        "model":              model,
        "label_encoder":      le,
        "feature_names":      FEATURES,
        "train_metrics":      train_metrics,
        "test_metrics":       test_metrics,
        "feature_importance": fi_df,
        "equipment_types":    le.classes_.tolist(),
    }
    joblib.dump(artifact, EQUIPMENT_MAINTENANCE_MODEL_PATH)
    print(f"  Model saved → {EQUIPMENT_MAINTENANCE_MODEL_PATH}")

    return artifact


def predict_maintenance_needs(engine, equipment_id: int) -> dict:
    """
    Predict whether a piece of equipment needs maintenance within 30 days.

    Parameters
    ----------
    engine       — SQLAlchemy engine
    equipment_id — dim_equipment.id

    Returns
    -------
    dict with keys:
        equipment_id                    — echo of input
        equipment_name                  — display name
        equipment_type                  — type string
        maintenance_probability         — float 0–1
        prediction                      — "Maintenance Needed" | "No Immediate Maintenance"
        predicted_days_until_maintenance — estimated days (30 × (1 – probability))
        recommendation                  — actionable guidance string
        features                        — dict of feature values used
    """
    if not EQUIPMENT_MAINTENANCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {EQUIPMENT_MAINTENANCE_MODEL_PATH}. "
            "Run train_equipment_maintenance_model() first."
        )

    artifact = joblib.load(EQUIPMENT_MAINTENANCE_MODEL_PATH)
    model    = artifact["model"]
    le       = artifact["label_encoder"]

    # Fetch equipment
    eq_df = _query(engine, """
        SELECT id, equipment_name, equipment_type, purchase_date, status
        FROM   dim_equipment
        WHERE  id = :equipment_id
    """, {"equipment_id": equipment_id})

    if eq_df.empty:
        raise ValueError(f"Equipment {equipment_id} not found.")

    eq    = eq_df.iloc[0]
    snap  = pd.Timestamp(TODAY)
    pdate = pd.to_datetime(eq["purchase_date"])

    # Fetch usage history
    usage = _query(engine, """
        SELECT
            date_id,
            hours_used,
            COALESCE(fuel_cost, 0)        AS fuel_cost,
            COALESCE(maintenance_cost, 0) AS maintenance_cost
        FROM fact_equipment_usage
        WHERE equipment_id = :equipment_id
        ORDER BY date_id
    """, {"equipment_id": equipment_id})

    age_days = max((snap - pdate).days, 1)

    if usage.empty:
        features = {
            "equipment_type_enc":       0,
            "age_days":                 age_days,
            "total_hours_used":         0.0,
            "avg_hours_per_day":        0.0,
            "fuel_cost_per_hour":       0.0,
            "recent_fuel_cost_trend":   1.0,
            "maintenance_event_count":  0,
            "days_since_maintenance":   age_days,
            "avg_maint_cost_per_event": 0.0,
        }
    else:
        usage["date_id"] = pd.to_datetime(usage["date_id"])
        total_hours = float(usage["hours_used"].sum())

        fuel_per_hour = (
            usage["fuel_cost"].sum() / total_hours if total_hours > 0 else 0.0
        )

        r_cut = snap - pd.Timedelta(days=90)
        o_cut = snap - pd.Timedelta(days=180)
        recent_fuel = float(usage[usage["date_id"] >= r_cut]["fuel_cost"].sum())
        older_fuel  = float(
            usage[
                (usage["date_id"] >= o_cut) & (usage["date_id"] < r_cut)
            ]["fuel_cost"].sum()
        )
        fuel_trend = min((recent_fuel / older_fuel) if older_fuel > 0 else 1.0, 10.0)

        maint = usage[usage["maintenance_cost"] > 0]
        maint_count = len(maint)
        if not maint.empty:
            last_maint = maint["date_id"].max()
            days_since = (snap - last_maint).days
            avg_maint  = float(maint["maintenance_cost"].mean())
        else:
            days_since = age_days
            avg_maint  = 0.0

        try:
            etype_enc = int(le.transform([eq["equipment_type"]])[0])
        except ValueError:
            etype_enc = 0

        features = {
            "equipment_type_enc":       etype_enc,
            "age_days":                 age_days,
            "total_hours_used":         round(total_hours, 1),
            "avg_hours_per_day":        round(total_hours / age_days, 3),
            "fuel_cost_per_hour":       round(fuel_per_hour, 4),
            "recent_fuel_cost_trend":   round(fuel_trend, 4),
            "maintenance_event_count":  maint_count,
            "days_since_maintenance":   days_since,
            "avg_maint_cost_per_event": round(avg_maint, 2),
        }

    X    = pd.DataFrame([features])[artifact["feature_names"]]
    X    = X.replace([np.inf, -np.inf], 0).fillna(0)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    predicted_days = max(1, int(30 * (1.0 - prob)))

    if prob >= 0.70:
        recommendation = (
            f"High maintenance probability ({prob:.0%}). Schedule a full service "
            f"within {predicted_days} days. Inspect hydraulics, engine hours, "
            f"and investigate recent fuel-cost increases."
        )
    elif prob >= 0.40:
        recommendation = (
            f"Moderate maintenance probability ({prob:.0%}). Plan a preventive "
            f"inspection within {predicted_days} days and monitor fuel efficiency."
        )
    else:
        recommendation = (
            f"Low maintenance probability ({prob:.0%}). Continue standard monitoring. "
            f"Next routine check recommended in approximately {predicted_days} days."
        )

    return {
        "equipment_id":                      equipment_id,
        "equipment_name":                    str(eq["equipment_name"]),
        "equipment_type":                    str(eq["equipment_type"]),
        "maintenance_probability":           round(prob, 4),
        "prediction":                        (
            "Maintenance Needed" if pred == 1 else "No Immediate Maintenance"
        ),
        "predicted_days_until_maintenance":  predicted_days,
        "recommendation":                    recommendation,
        "features":                          features,
    }


# ===========================================================================
# main() — train, evaluate, and demo both models
# ===========================================================================


def main():
    BAR = "=" * 70

    def section(title: str) -> None:
        print(f"\n{BAR}\n  {title}\n{BAR}")

    def print_metrics(label: str, metrics: dict) -> None:
        print(f"\n  [{label}]")
        for k, v in metrics.items():
            val = f"{v:.4f}" if v is not None else "N/A"
            print(f"    {k:12s} {val}")

    engine = get_engine()

    # ── MODEL 1: Cost Overrun ──────────────────────────────────────────────
    section("MODEL 1: Cost Overrun Prediction")

    print()
    result1 = train_cost_overrun_model(engine)

    print_metrics("Train", result1["train_metrics"])
    print_metrics("Test",  result1["test_metrics"])

    print("\n  [Feature Importance]")
    print(result1["feature_importance"].to_string(index=False))

    # Sample predictions on 5 active/completed projects
    print("\n  [Sample Predictions]")
    with engine.connect() as conn:
        sample_projects = conn.execute(text("""
            SELECT p.id, p.project_name
            FROM   dim_projects p
            JOIN   fact_project_costs fpc ON p.id = fpc.project_id
            WHERE  p.status IN ('ACTIVE', 'COMPLETED')
              AND  p.actual_start_date IS NOT NULL
            GROUP  BY p.id
            HAVING COUNT(*) > 0
            LIMIT  5
        """)).fetchall()

    for pid, pname in sample_projects:
        try:
            r = predict_cost_overrun(engine, int(pid))
            print(
                f"    [{pid:3d}] {pname[:38]:<38}  "
                f"prob={r['overrun_probability']:.3f}  "
                f"{r['prediction']}"
            )
        except Exception as exc:
            print(f"    [{pid:3d}] {pname[:38]:<38}  Error: {exc}")

    # ── MODEL 2: Equipment Maintenance ────────────────────────────────────
    section("MODEL 2: Equipment Maintenance Prediction")

    print()
    result2 = train_equipment_maintenance_model(engine)

    print_metrics("Train", result2["train_metrics"])
    print_metrics("Test",  result2["test_metrics"])

    print("\n  [Feature Importance]")
    print(result2["feature_importance"].to_string(index=False))

    # Sample predictions on 5 active equipment pieces
    print("\n  [Sample Predictions]")
    with engine.connect() as conn:
        sample_equip = conn.execute(text("""
            SELECT e.id, e.equipment_name, e.equipment_type
            FROM   dim_equipment e
            JOIN   fact_equipment_usage u ON e.id = u.equipment_id
            WHERE  e.status != 'RETIRED'
            GROUP  BY e.id
            HAVING COUNT(*) > 0
            LIMIT  5
        """)).fetchall()

    for eid, ename, etype in sample_equip:
        try:
            r = predict_maintenance_needs(engine, int(eid))
            print(
                f"    [{eid:3d}] {ename[:28]:<28}  ({etype:10s})  "
                f"prob={r['maintenance_probability']:.3f}  "
                f"est.{r['predicted_days_until_maintenance']:3d}d  "
                f"{r['prediction']}"
            )
        except Exception as exc:
            print(f"    [{eid:3d}] {ename[:28]:<28}  Error: {exc}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{BAR}")
    print("  Both models trained and saved successfully.")
    print(f"    {COST_OVERRUN_MODEL_PATH}")
    print(f"    {EQUIPMENT_MAINTENANCE_MODEL_PATH}")
    print(f"{BAR}\n")


if __name__ == "__main__":
    main()
