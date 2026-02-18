"""
conftest.py — Shared pytest fixtures for the construction analytics test suite.

All heavy fixtures are session-scoped so that the database is seeded once and
both ML models are trained once per test run.

Fixture dependency graph
------------------------
  tmp_models_dir   (session)
       └─► patch_model_paths  (session, autouse=True)
                 └─► (active for entire session)

  db_engine        (session) — in-memory SQLite with minimal seed data
       └─► trained_artifacts  (session) — train both models once
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Make src/ importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from database import Base  # noqa: E402  (after sys.path insert)
import ml_models            # noqa: E402


# ===========================================================================
# Model-path patching
# ===========================================================================


@pytest.fixture(scope="session")
def tmp_models_dir(tmp_path_factory):
    """Session-scoped temporary directory that holds test .pkl artifacts."""
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session", autouse=True)
def patch_model_paths(tmp_models_dir):
    """
    Redirect ml_models module-level path constants to *tmp_models_dir* for
    the entire test session.

    train_*/predict_* resolve MODELS_DIR, COST_OVERRUN_MODEL_PATH, and
    EQUIPMENT_MAINTENANCE_MODEL_PATH from the module's global namespace at
    call time, so patching them here is sufficient — no per-function mocking
    required.
    """
    orig_dir   = ml_models.MODELS_DIR
    orig_cost  = ml_models.COST_OVERRUN_MODEL_PATH
    orig_maint = ml_models.EQUIPMENT_MAINTENANCE_MODEL_PATH

    ml_models.MODELS_DIR                       = tmp_models_dir
    ml_models.COST_OVERRUN_MODEL_PATH          = tmp_models_dir / "cost_overrun_model.pkl"
    ml_models.EQUIPMENT_MAINTENANCE_MODEL_PATH = tmp_models_dir / "equipment_maintenance_model.pkl"

    yield  # all tests run here

    # Restore originals (good practice even for session fixtures)
    ml_models.MODELS_DIR                       = orig_dir
    ml_models.COST_OVERRUN_MODEL_PATH          = orig_cost
    ml_models.EQUIPMENT_MAINTENANCE_MODEL_PATH = orig_maint


# ===========================================================================
# Test-database seed helpers
# ===========================================================================


def _seed_dim_dates(conn) -> None:
    """Insert all 365 days of 2022 into dim_dates."""
    start = date(2022, 1, 1)
    rows = []
    for i in range(365):
        d   = start + timedelta(days=i)
        dow = d.weekday()   # 0 = Mon … 6 = Sun
        rows.append({
            "dt":    d.isoformat(),
            "yr":    d.year,
            "q":     (d.month - 1) // 3 + 1,
            "mo":    d.month,
            "mn":    d.strftime("%B"),
            "wk":    d.isocalendar()[1],
            "dow":   dow,
            "is_we": 1 if dow >= 5 else 0,
            "is_hol": 0,
        })
    conn.execute(
        text("""
            INSERT INTO dim_dates
                (date, year, quarter, month, month_name,
                 week, day_of_week, is_weekend, is_holiday)
            VALUES
                (:dt, :yr, :q, :mo, :mn, :wk, :dow, :is_we, :is_hol)
        """),
        rows,
    )


def _seed_projects(conn) -> None:
    """
    Insert 11 test projects.

    IDs 1–4   COMPLETED COMMERCIAL  — cost overrun  (+25 % variance)
    IDs 5–8   COMPLETED RESIDENTIAL — under-budget  (−20 % variance)
    ID  9     ACTIVE    COMMERCIAL  — has cost data (for prediction tests)
    ID  10    ACTIVE    RESIDENTIAL — no cost data  (for missing-data tests)
    ID  11    ACTIVE    COMMERCIAL  — equipment host (FK for fact_equipment_usage)
    """
    for i in range(1, 9):
        ptype = "COMMERCIAL" if i <= 4 else "RESIDENTIAL"
        conn.execute(
            text("""
                INSERT INTO dim_projects
                    (id, project_name, project_type, contract_value,
                     planned_start_date, planned_end_date,
                     actual_start_date,  actual_end_date,
                     status, client_name, square_footage)
                VALUES
                    (:id, :name, :ptype, 1000000.0,
                     '2022-01-01', '2022-12-31',
                     '2022-01-03', '2022-12-30',
                     'COMPLETED', :client, 10000.0)
            """),
            {"id": i, "name": f"Test Project {i}", "ptype": ptype, "client": f"Client {i}"},
        )

    # Active project with cost data
    conn.execute(text("""
        INSERT INTO dim_projects
            (id, project_name, project_type, contract_value,
             planned_start_date, planned_end_date, actual_start_date,
             status, client_name)
        VALUES
            (9, 'Test Project Alpha', 'COMMERCIAL', 500000.0,
             '2022-01-01', '2022-12-31', '2022-01-03',
             'ACTIVE', 'Client Alpha')
    """))

    # Active project with NO cost data (tests missing-data handling)
    conn.execute(text("""
        INSERT INTO dim_projects
            (id, project_name, project_type, contract_value,
             planned_start_date, planned_end_date, actual_start_date,
             status, client_name)
        VALUES
            (10, 'Test Project No Data', 'RESIDENTIAL', 300000.0,
             '2022-01-01', '2022-12-31', '2022-01-03',
             'ACTIVE', 'Client NoData')
    """))

    # Equipment host project (only needed so fact_equipment_usage FK is valid)
    conn.execute(text("""
        INSERT INTO dim_projects
            (id, project_name, project_type, contract_value,
             planned_start_date, planned_end_date, actual_start_date,
             status, client_name)
        VALUES
            (11, 'Equipment Host Project', 'COMMERCIAL', 2000000.0,
             '2022-01-01', '2022-12-31', '2022-01-03',
             'ACTIVE', 'Client Host')
    """))


def _seed_project_costs(conn) -> None:
    """
    Insert one cost record per month (on the 15th) for Jan–Dec 2022.

    Projects 1–4 (overrun):     actual = 1.25 × budgeted  → variance = +25 %
    Projects 5–8 (under):       actual = 0.80 × budgeted  → variance = −20 %
    Project 9   (active/alpha): actual = 50 000, budgeted = 45 000 (Jan–Jun only)

    With 12 months × $80 K budgeted per project:
      overrun total variance  = (960 K − 1 200 K) / 960 K = not right, let me recheck
      actual overrun  = 1.25 × 80 000 × 12 = 1 200 000
      budgeted        =         80 000 × 12 =   960 000
      variance_pct    = (1 200 000 − 960 000) / 960 000 ≈ +25 % > 10 % → will_overrun = 1

      actual under    = 0.80 × 80 000 × 12 = 768 000
      variance_pct    = (768 000 − 960 000) / 960 000 ≈ −20 % < 10 % → will_overrun = 0
    """
    BUDGETED = 80_000.0
    months   = [date(2022, m, 15) for m in range(1, 13)]

    rows: list[dict] = []
    for pid in range(1, 9):
        factor = 1.25 if pid <= 4 else 0.80
        for d in months:
            rows.append({
                "pid": pid,
                "did": d.isoformat(),
                "cat": "LABOR",
                "amt": BUDGETED * factor,
                "bud": BUDGETED,
            })

    # Project 9 — slightly over budget (Jan–Jun), used for live prediction test
    for d in months[:6]:
        rows.append({
            "pid": 9,
            "did": d.isoformat(),
            "cat": "LABOR",
            "amt": 50_000.0,
            "bud": 45_000.0,
        })

    conn.execute(
        text("""
            INSERT INTO fact_project_costs
                (project_id, date_id, cost_category, amount, budgeted_amount)
            VALUES (:pid, :did, :cat, :amt, :bud)
        """),
        rows,
    )


def _seed_equipment(conn) -> None:
    """
    Insert 5 equipment pieces.

    IDs 1–4  will have full daily usage records for 2022.
    ID  5    (FORKLIFT) has no usage records — tests missing-data handling.
    """
    equipment = [
        (1, "Test Excavator 1",      "EXCAVATOR", "2021-01-01"),
        (2, "Test Crane 1",          "CRANE",     "2021-01-01"),
        (3, "Test Bulldozer 1",      "BULLDOZER", "2021-01-01"),
        (4, "Test Truck 1",          "TRUCK",     "2021-01-01"),
        (5, "Test Forklift No Data", "FORKLIFT",  "2021-01-01"),
    ]
    conn.execute(
        text("""
            INSERT INTO dim_equipment
                (id, equipment_name, equipment_type, purchase_date, status)
            VALUES (:id, :name, :etype, :pdate, 'ACTIVE')
        """),
        [{"id": e[0], "name": e[1], "etype": e[2], "pdate": e[3]} for e in equipment],
    )


# Maintenance event dates per equipment (all fall on weekdays within 2022).
# Chosen so that every equipment has ≥ 2 snapshot months where target = 1
# and ≥ 2 snapshot months where target = 0 (satisfies stratified-split guard).
_MAINTENANCE_DATES: dict[int, frozenset[str]] = {
    1: frozenset({"2022-03-15", "2022-06-15", "2022-09-15", "2022-12-15"}),  # every ~3 months
    2: frozenset({"2022-04-15", "2022-08-15", "2022-12-15"}),                # every ~4 months
    3: frozenset({"2022-02-15", "2022-05-15", "2022-08-15", "2022-11-15"}),  # every ~3 months
    4: frozenset({"2022-06-15", "2022-10-15"}),                              # every ~4 months
}


def _seed_equipment_usage(conn) -> None:
    """
    Insert one fact_equipment_usage record per calendar day of 2022 for
    equipment IDs 1–4, linked to the equipment host project (ID 11).

    Regular days:    hours_used = 8,  fuel_cost = 50.0,  maintenance_cost = 0
    Maintenance days: hours_used = 4, fuel_cost = 25.0,  maintenance_cost = 500
    """
    start = date(2022, 1, 1)
    rows: list[dict] = []

    for eid in range(1, 5):
        maint_dates = _MAINTENANCE_DATES[eid]
        for i in range(365):
            d  = start + timedelta(days=i)
            ds = d.isoformat()
            if ds in maint_dates:
                rows.append({"pid": 11, "eid": eid, "did": ds,
                             "hrs": 4.0, "fuel": 25.0, "maint": 500.0})
            else:
                rows.append({"pid": 11, "eid": eid, "did": ds,
                             "hrs": 8.0, "fuel": 50.0, "maint": 0.0})

    conn.execute(
        text("""
            INSERT INTO fact_equipment_usage
                (project_id, equipment_id, date_id,
                 hours_used, fuel_cost, maintenance_cost)
            VALUES (:pid, :eid, :did, :hrs, :fuel, :maint)
        """),
        rows,
    )


# ===========================================================================
# Main fixtures
# ===========================================================================


@pytest.fixture(scope="session")
def db_engine():
    """
    Session-scoped in-memory SQLite engine pre-seeded with minimal test data:

      • 8 COMPLETED projects (4 overrun, 4 under-budget) with monthly costs
      • 1 ACTIVE project with 6 months of cost data
      • 1 ACTIVE project with no cost data
      • 4 equipment pieces with 365 daily usage records each
      • 1 equipment piece with no usage records

    Foreign-key constraints are intentionally left unenforced (no PRAGMA
    foreign_keys=ON) to keep the seed logic order-independent.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        _seed_dim_dates(conn)
        _seed_projects(conn)
        _seed_project_costs(conn)
        _seed_equipment(conn)
        _seed_equipment_usage(conn)

    return engine


@pytest.fixture(scope="session")
def trained_artifacts(db_engine):
    """
    Train both ML models exactly once using the test database and the patched
    (temporary) model paths.  Returns::

        {
            "cost_overrun": <artifact dict>,   # from train_cost_overrun_model
            "maintenance":  <artifact dict>,   # from train_equipment_maintenance_model
        }

    All subsequent tests that need model artifacts use this fixture rather
    than re-training, keeping the suite fast.
    """
    from ml_models import (
        train_cost_overrun_model,
        train_equipment_maintenance_model,
    )

    cost_artifact  = train_cost_overrun_model(db_engine)
    maint_artifact = train_equipment_maintenance_model(db_engine)

    return {
        "cost_overrun": cost_artifact,
        "maintenance":  maint_artifact,
    }
