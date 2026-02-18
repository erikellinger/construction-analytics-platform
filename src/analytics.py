"""
analytics.py — Business intelligence queries for the Construction Analytics Platform.

Each function executes parameterised SQL against the SQLite database
and returns a pandas DataFrame ready for dashboarding or further analysis.

Run directly to execute all six analyses and print sample results:
    python src/analytics.py
"""

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
from database import get_engine

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _query(engine, sql: str, params: dict = None) -> pd.DataFrame:
    """Execute *sql* with optional bound *params* and return a DataFrame."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


# ---------------------------------------------------------------------------
# 1. Project Performance Summary
# ---------------------------------------------------------------------------


def get_project_performance_summary(engine, project_id: int) -> pd.DataFrame:
    """
    Cost vs budget by category, CPI, SPI proxy, and schedule variance
    for a single project.

    CPI > 1.0  →  under budget    |  CPI < 1.0  →  over budget
    SPI > 1.0  →  ahead of schedule (proxy: planned_days / elapsed_days)
    """
    sql = """
    WITH project_meta AS (
        SELECT
            id,
            project_name,
            project_type,
            contract_value,
            status,
            planned_start_date,
            planned_end_date,
            actual_start_date,
            actual_end_date,
            CAST(
                julianday(planned_end_date) - julianday(planned_start_date)
            AS INTEGER) AS planned_duration_days,
            CAST(
                julianday(COALESCE(actual_end_date, '2026-02-17'))
                - julianday(actual_start_date)
            AS INTEGER) AS elapsed_days
        FROM dim_projects
        WHERE id = :project_id
    ),
    cost_by_category AS (
        -- Aggregate actuals and budget per cost category with per-category CPI
        SELECT
            cost_category,
            ROUND(SUM(amount), 2)          AS actual_cost,
            ROUND(SUM(budgeted_amount), 2) AS budgeted_cost,
            ROUND(
                SUM(budgeted_amount) - SUM(amount), 2
            )                              AS variance,
            ROUND(
                100.0 * (SUM(budgeted_amount) - SUM(amount))
                / NULLIF(SUM(budgeted_amount), 0), 2
            )                              AS variance_pct,
            ROUND(
                SUM(budgeted_amount) / NULLIF(SUM(amount), 0), 3
            )                              AS cpi
        FROM fact_project_costs
        WHERE project_id = :project_id
        GROUP BY cost_category
    ),
    overall AS (
        SELECT
            ROUND(SUM(amount), 2)                                      AS total_actual,
            ROUND(SUM(budgeted_amount), 2)                             AS total_budgeted,
            ROUND(SUM(budgeted_amount) / NULLIF(SUM(amount), 0), 3)   AS overall_cpi
        FROM fact_project_costs
        WHERE project_id = :project_id
    )
    SELECT
        pm.project_name,
        pm.project_type,
        pm.status,
        pm.contract_value,
        pm.planned_start_date,
        pm.planned_end_date,
        pm.actual_start_date,
        pm.actual_end_date,
        pm.planned_duration_days,
        pm.elapsed_days,
        -- Positive = behind schedule (took longer than planned)
        pm.elapsed_days - pm.planned_duration_days          AS schedule_variance_days,
        -- SPI proxy: >1.0 means work finished faster than planned
        ROUND(
            CAST(pm.planned_duration_days AS REAL)
            / NULLIF(pm.elapsed_days, 0), 3
        )                                                   AS spi,
        ov.total_actual,
        ov.total_budgeted,
        ROUND(ov.total_budgeted - ov.total_actual, 2)      AS total_variance,
        ov.overall_cpi,
        cb.cost_category,
        cb.actual_cost,
        cb.budgeted_cost,
        cb.variance,
        cb.variance_pct,
        cb.cpi                                              AS category_cpi,
        CASE
            WHEN cb.cpi >= 1.00 THEN 'Under Budget'
            WHEN cb.cpi >= 0.90 THEN 'Slightly Over'
            WHEN cb.cpi >= 0.75 THEN 'Over Budget'
            ELSE                     'Significantly Over'
        END                                                 AS budget_status,
        -- Category's share of total spend
        ROUND(100.0 * cb.actual_cost / NULLIF(ov.total_actual, 0), 2) AS pct_of_total_spend
    FROM project_meta   pm
    CROSS JOIN overall  ov
    CROSS JOIN cost_by_category cb
    ORDER BY cb.actual_cost DESC
    """
    return _query(engine, sql, {"project_id": project_id})


# ---------------------------------------------------------------------------
# 2. Equipment Utilization Analysis
# ---------------------------------------------------------------------------


def get_equipment_utilization_analysis(
    engine,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Utilization %, revenue generated, and underutilised flag per active
    equipment piece for the given date window.

    Available hours = working (non-weekend, non-holiday) days × 8 hours/day.
    Underutilised  = utilization < 50%.
    """
    sql = """
    WITH date_range AS (
        -- Total available working days in the analysis window
        SELECT COUNT(*) AS working_days
        FROM dim_dates
        WHERE date      BETWEEN :start_date AND :end_date
          AND is_weekend = 0
          AND is_holiday = 0
    ),
    usage_per_equipment AS (
        SELECT
            equipment_id,
            SUM(hours_used)                    AS hours_used,
            SUM(COALESCE(fuel_cost, 0))        AS fuel_cost,
            SUM(COALESCE(maintenance_cost, 0)) AS maintenance_cost,
            COUNT(DISTINCT project_id)         AS projects_served
        FROM fact_equipment_usage
        WHERE date_id BETWEEN :start_date AND :end_date
        GROUP BY equipment_id
    ),
    base AS (
        SELECT
            e.id,
            e.equipment_name,
            e.equipment_type,
            e.hourly_rate,
            e.status,
            e.purchase_cost,
            e.purchase_date,
            COALESCE(u.hours_used, 0)        AS hours_used,
            COALESCE(u.fuel_cost, 0)         AS fuel_cost,
            COALESCE(u.maintenance_cost, 0)  AS maintenance_cost,
            COALESCE(u.projects_served, 0)   AS projects_served,
            COALESCE(u.hours_used, 0) * e.hourly_rate AS revenue_generated,
            dr.working_days * 8              AS available_hours
        FROM dim_equipment  e
        CROSS JOIN date_range dr
        LEFT JOIN usage_per_equipment u ON e.id = u.equipment_id
        WHERE e.status != 'RETIRED'
    )
    SELECT
        equipment_name,
        equipment_type,
        status,
        ROUND(hourly_rate, 2)                   AS hourly_rate,
        purchase_cost,
        purchase_date,
        ROUND(hours_used, 1)                    AS hours_used,
        available_hours,
        -- Core utilisation metric
        ROUND(
            100.0 * hours_used / NULLIF(available_hours, 0), 2
        )                                       AS utilization_pct,
        CASE
            WHEN 100.0 * hours_used / NULLIF(available_hours, 0) >= 75 THEN 'High (≥75%)'
            WHEN 100.0 * hours_used / NULLIF(available_hours, 0) >= 50 THEN 'Moderate (50–75%)'
            WHEN hours_used > 0                                         THEN 'Low (<50%)'
            ELSE                                                             'Idle'
        END                                     AS utilization_tier,
        ROUND(revenue_generated, 2)             AS revenue_generated,
        ROUND(fuel_cost, 2)                     AS fuel_cost,
        ROUND(maintenance_cost, 2)              AS maintenance_cost,
        -- Net contribution after running costs
        ROUND(revenue_generated - fuel_cost - maintenance_cost, 2) AS net_revenue,
        projects_served,
        -- Rank within equipment type (1 = most utilised in its class)
        RANK() OVER (
            PARTITION BY equipment_type
            ORDER BY 100.0 * hours_used / NULLIF(available_hours, 0) DESC
        )                                       AS rank_in_type,
        -- Fleet-wide utilisation percentile
        ROUND(
            100.0 * RANK() OVER (ORDER BY hours_used / NULLIF(available_hours, 0))
            / COUNT(*) OVER (), 1
        )                                       AS fleet_utilization_percentile,
        -- Binary flag for asset review
        CASE
            WHEN 100.0 * hours_used / NULLIF(available_hours, 0) < 50 THEN 1
            ELSE 0
        END                                     AS is_underutilized
    FROM base
    ORDER BY utilization_pct DESC
    """
    return _query(engine, sql, {
        "start_date": str(start_date),
        "end_date":   str(end_date),
    })


# ---------------------------------------------------------------------------
# 3. Subcontractor Performance Scorecard
# ---------------------------------------------------------------------------


def get_subcontractor_performance_scorecard(engine) -> pd.DataFrame:
    """
    Ranks all subcontractors by a composite score weighting:
      50% financial CPI, 30% client rating, 20% project volume (capped at 10).

    Returns overall rank, rank-within-trade, performance tier,
    and a flag identifying the top performer in each trade.
    """
    sql = """
    WITH sub_costs AS (
        -- Roll up subcontractor cost entries to the project level
        SELECT
            subcontractor_id,
            project_id,
            SUM(amount)          AS actual_cost,
            SUM(budgeted_amount) AS budgeted_cost
        FROM fact_project_costs
        WHERE cost_category    = 'SUBCONTRACTOR'
          AND subcontractor_id IS NOT NULL
        GROUP BY subcontractor_id, project_id
    ),
    sub_metrics AS (
        SELECT
            subcontractor_id,
            COUNT(DISTINCT project_id)                               AS projects_count,
            ROUND(SUM(actual_cost), 2)                               AS total_actual,
            ROUND(SUM(budgeted_cost), 2)                             AS total_budgeted,
            ROUND(
                SUM(budgeted_cost) / NULLIF(SUM(actual_cost), 0), 3
            )                                                        AS cpi,
            ROUND(
                100.0 * (SUM(budgeted_cost) - SUM(actual_cost))
                / NULLIF(SUM(budgeted_cost), 0), 2
            )                                                        AS budget_adherence_pct
        FROM sub_costs
        GROUP BY subcontractor_id
    ),
    scored AS (
        SELECT
            s.id,
            s.company_name,
            s.trade,
            s.rating                        AS client_rating,
            s.years_in_business,
            sm.projects_count,
            sm.total_actual,
            sm.total_budgeted,
            sm.cpi,
            sm.budget_adherence_pct,
            -- Composite score — 50/30/20 weighting, projects capped at 10
            ROUND(
                (COALESCE(sm.cpi, 0) * 50.0)
                + (s.rating / 5.0 * 30.0)
                + (
                    CASE WHEN sm.projects_count > 10 THEN 10.0
                         ELSE CAST(sm.projects_count AS REAL)
                    END / 10.0 * 20.0
                  ),
                2
            ) AS composite_score
        FROM dim_subcontractors s
        JOIN sub_metrics sm ON s.id = sm.subcontractor_id
    )
    SELECT
        company_name,
        trade,
        ROUND(client_rating, 1)    AS client_rating,
        years_in_business,
        projects_count,
        total_actual,
        total_budgeted,
        cpi,
        budget_adherence_pct,
        composite_score,
        -- Fleet-wide rank (1 = best overall performer)
        RANK() OVER (
            ORDER BY composite_score DESC
        )                          AS overall_rank,
        -- Within-trade rank for apples-to-apples comparison
        RANK() OVER (
            PARTITION BY trade
            ORDER BY composite_score DESC
        )                          AS rank_in_trade,
        -- Performance tier label
        CASE
            WHEN RANK() OVER (ORDER BY composite_score DESC) <= 10 THEN 'Top Performer'
            WHEN cpi >= 1.0 AND client_rating >= 4.0              THEN 'Recommended'
            WHEN cpi >= 0.90                                       THEN 'Above Average'
            WHEN cpi < 0.80                                        THEN 'Needs Improvement'
            ELSE                                                        'Average'
        END                        AS performance_tier,
        -- Flag the single best subcontractor in each trade
        CASE
            WHEN ROW_NUMBER() OVER (
                PARTITION BY trade ORDER BY composite_score DESC
            ) = 1 THEN 'Yes'
            ELSE 'No'
        END                        AS top_in_trade
    FROM scored
    ORDER BY overall_rank
    """
    return _query(engine, sql)


# ---------------------------------------------------------------------------
# 4. Labor Productivity by Trade
# ---------------------------------------------------------------------------


def get_labor_productivity_by_trade(
    engine,
    project_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Total hours, cost, effective rate, and month-over-month trend per trade.

    project_id=None  → fleet-wide view across all projects.
    project_id=<int> → scoped to a single project.

    Uses LAG() to compute MoM change for the most recent data month.
    """
    # Safe conditional: only a static SQL clause is interpolated (never user data)
    project_filter = "AND lh.project_id = :project_id" if project_id else ""
    params = {"project_id": project_id} if project_id else {}

    sql = f"""
    WITH labor_agg AS (
        SELECT
            e.trade,
            ROUND(SUM(lh.hours_worked), 1)                           AS total_hours,
            ROUND(SUM(lh.hours_worked * lh.hourly_rate), 2)          AS total_labor_cost,
            ROUND(AVG(lh.hourly_rate), 2)                            AS avg_hourly_rate,
            COUNT(DISTINCT lh.employee_id)                           AS unique_employees,
            COUNT(DISTINCT lh.project_id)                            AS projects_worked
        FROM fact_labor_hours lh
        JOIN dim_employees e ON lh.employee_id = e.id
        WHERE 1=1 {project_filter}
        GROUP BY e.trade
    ),
    monthly AS (
        -- Monthly hours and cost per trade — foundation for MoM trend
        SELECT
            e.trade,
            strftime('%Y-%m', lh.date_id)              AS yr_month,
            SUM(lh.hours_worked)                        AS monthly_hours,
            SUM(lh.hours_worked * lh.hourly_rate)       AS monthly_cost
        FROM fact_labor_hours lh
        JOIN dim_employees e ON lh.employee_id = e.id
        WHERE 1=1 {project_filter}
        GROUP BY e.trade, strftime('%Y-%m', lh.date_id)
    ),
    monthly_with_lag AS (
        -- Attach prior-month values using LAG window function
        SELECT
            trade,
            yr_month,
            monthly_hours,
            monthly_cost,
            LAG(monthly_hours) OVER (PARTITION BY trade ORDER BY yr_month) AS prev_hours,
            LAG(monthly_cost)  OVER (PARTITION BY trade ORDER BY yr_month) AS prev_cost
        FROM monthly
    ),
    latest_month AS (
        -- Extract only the most recent month's MoM delta per trade
        SELECT
            trade,
            yr_month                AS latest_month,
            ROUND(
                100.0 * (monthly_hours - prev_hours) / NULLIF(prev_hours, 0), 2
            )                       AS mom_hours_pct,
            ROUND(
                100.0 * (monthly_cost - prev_cost) / NULLIF(prev_cost, 0), 2
            )                       AS mom_cost_pct
        FROM monthly_with_lag
        WHERE yr_month = (
            SELECT MAX(yr_month)
            FROM monthly_with_lag m2
            WHERE m2.trade = monthly_with_lag.trade
        )
    )
    SELECT
        la.trade,
        la.total_hours,
        la.total_labor_cost,
        la.avg_hourly_rate,
        -- Effective rate may differ from avg_hourly_rate due to OT / blended crew
        ROUND(la.total_labor_cost / NULLIF(la.total_hours, 0), 2)  AS effective_cost_per_hour,
        la.unique_employees,
        ROUND(la.total_hours / NULLIF(la.unique_employees, 0), 1)  AS avg_hours_per_employee,
        la.projects_worked,
        lm.latest_month,
        lm.mom_hours_pct,
        lm.mom_cost_pct,
        -- Fleet-wide rankings
        RANK() OVER (ORDER BY la.total_labor_cost DESC)            AS cost_rank,
        RANK() OVER (ORDER BY la.total_hours DESC)                 AS hours_rank,
        -- Share of total workforce effort
        ROUND(
            100.0 * la.total_hours / SUM(la.total_hours) OVER (), 2
        )                                                           AS pct_of_total_hours,
        ROUND(
            100.0 * la.total_labor_cost / SUM(la.total_labor_cost) OVER (), 2
        )                                                           AS pct_of_total_cost
    FROM labor_agg la
    LEFT JOIN latest_month lm ON la.trade = lm.trade
    ORDER BY la.total_labor_cost DESC
    """
    return _query(engine, sql, params)


# ---------------------------------------------------------------------------
# 5. Safety Metrics
# ---------------------------------------------------------------------------


def get_safety_metrics(engine, start_date: date, end_date: date) -> pd.DataFrame:
    """
    OSHA-standard incident rate, severity/type breakdown, days lost,
    and count of zero-incident projects for the given date window.

    OSHA Incident Rate = (incidents × 200,000) / total man-hours worked.
    (200,000 = 100 workers × 50 weeks × 40 hrs — the OSHA base equivalent.)
    """
    sql = """
    WITH period_hours AS (
        SELECT COALESCE(SUM(hours_worked), 0) AS total_hours
        FROM fact_labor_hours
        WHERE date_id BETWEEN :start_date AND :end_date
    ),
    incident_breakdown AS (
        -- Granular breakdown: one row per (severity × incident_type) combination
        SELECT
            severity,
            incident_type,
            COUNT(*)       AS incident_count,
            SUM(days_lost) AS days_lost
        FROM fact_safety_incidents
        WHERE date_id BETWEEN :start_date AND :end_date
        GROUP BY severity, incident_type
    ),
    incident_totals AS (
        SELECT
            COUNT(*)                                                        AS total_incidents,
            SUM(days_lost)                                                  AS total_days_lost,
            SUM(CASE WHEN incident_type = 'INJURY'    THEN 1 ELSE 0 END)   AS injuries,
            SUM(CASE WHEN incident_type = 'NEAR_MISS' THEN 1 ELSE 0 END)   AS near_misses,
            SUM(CASE WHEN incident_type = 'VIOLATION' THEN 1 ELSE 0 END)   AS violations,
            SUM(CASE WHEN severity = 'SEVERE'         THEN 1 ELSE 0 END)   AS severe_count,
            SUM(CASE WHEN severity = 'MODERATE'       THEN 1 ELSE 0 END)   AS moderate_count,
            SUM(CASE WHEN severity = 'MINOR'          THEN 1 ELSE 0 END)   AS minor_count
        FROM fact_safety_incidents
        WHERE date_id BETWEEN :start_date AND :end_date
    ),
    affected_projects AS (
        SELECT DISTINCT project_id
        FROM fact_safety_incidents
        WHERE date_id BETWEEN :start_date AND :end_date
    ),
    zero_incident_projects AS (
        -- Active/completed projects that had NO incidents in the window
        SELECT COUNT(*) AS cnt
        FROM dim_projects p
        WHERE p.status          IN ('ACTIVE', 'COMPLETED')
          AND p.actual_start_date IS NOT NULL
          AND p.id NOT IN (SELECT project_id FROM affected_projects)
    )
    SELECT
        ib.severity,
        ib.incident_type,
        ib.incident_count,
        ib.days_lost,
        -- Per-type OSHA rate
        ROUND(200000.0 * ib.incident_count / NULLIF(ph.total_hours, 0), 4)
                                                 AS osha_rate_this_type,
        -- Aggregates repeated across all rows for dashboard convenience
        it.total_incidents,
        it.total_days_lost,
        it.injuries,
        it.near_misses,
        it.violations,
        it.severe_count,
        it.moderate_count,
        it.minor_count,
        ROUND(200000.0 * it.total_incidents / NULLIF(ph.total_hours, 0), 4)
                                                 AS osha_total_incident_rate,
        ROUND(200000.0 * it.injuries / NULLIF(ph.total_hours, 0), 4)
                                                 AS osha_injury_rate,
        ROUND(ph.total_hours)                    AS total_labor_hours,
        zip.cnt                                  AS zero_incident_projects
    FROM incident_breakdown     ib
    CROSS JOIN period_hours     ph
    CROSS JOIN incident_totals  it
    CROSS JOIN zero_incident_projects zip
    ORDER BY
        CASE ib.severity
            WHEN 'SEVERE'   THEN 1
            WHEN 'MODERATE' THEN 2
            ELSE                 3
        END,
        ib.incident_type
    """
    return _query(engine, sql, {
        "start_date": str(start_date),
        "end_date":   str(end_date),
    })


# ---------------------------------------------------------------------------
# 6. Budget Variance Analysis
# ---------------------------------------------------------------------------


def get_budget_variance_analysis(engine) -> pd.DataFrame:
    """
    Budget performance for every project with type-level benchmarks.

    Uses RANK() to surface top overruns, CASE to classify status,
    and per-category overrun columns to identify the primary cost driver.
    """
    sql = """
    WITH project_totals AS (
        -- Total actual vs budget per project, plus category-level overruns
        SELECT
            project_id,
            SUM(amount)                                                    AS total_actual,
            SUM(budgeted_amount)                                           AS total_budgeted,
            SUM(CASE WHEN cost_category = 'LABOR'
                     THEN amount - budgeted_amount ELSE 0 END)            AS labor_overrun,
            SUM(CASE WHEN cost_category = 'MATERIALS'
                     THEN amount - budgeted_amount ELSE 0 END)            AS materials_overrun,
            SUM(CASE WHEN cost_category = 'EQUIPMENT'
                     THEN amount - budgeted_amount ELSE 0 END)            AS equipment_overrun,
            SUM(CASE WHEN cost_category = 'SUBCONTRACTOR'
                     THEN amount - budgeted_amount ELSE 0 END)            AS subcontractor_overrun
        FROM fact_project_costs
        GROUP BY project_id
    ),
    project_variance AS (
        SELECT
            p.id,
            p.project_name,
            p.project_type,
            p.contract_value,
            p.status,
            ROUND(pt.total_actual, 2)                                      AS total_actual,
            ROUND(pt.total_budgeted, 2)                                    AS total_budgeted,
            ROUND(pt.total_actual - pt.total_budgeted, 2)                  AS cost_variance,
            ROUND(
                100.0 * (pt.total_actual - pt.total_budgeted)
                / NULLIF(pt.total_budgeted, 0), 2
            )                                                               AS variance_pct,
            ROUND(pt.labor_overrun, 2)                                     AS labor_overrun,
            ROUND(pt.materials_overrun, 2)                                 AS materials_overrun,
            ROUND(pt.equipment_overrun, 2)                                 AS equipment_overrun,
            ROUND(pt.subcontractor_overrun, 2)                             AS subcontractor_overrun,
            CASE
                WHEN pt.total_actual <= pt.total_budgeted              THEN 'Under Budget'
                WHEN pt.total_actual <= pt.total_budgeted * 1.05       THEN 'On Budget (±5%)'
                WHEN pt.total_actual <= pt.total_budgeted * 1.15       THEN 'Over Budget (5–15%)'
                ELSE                                                         'Significantly Over (>15%)'
            END AS budget_status,
            -- Identify the cost category driving the largest absolute overrun
            CASE
                WHEN ABS(pt.labor_overrun) >= ABS(pt.materials_overrun)
                 AND ABS(pt.labor_overrun) >= ABS(pt.equipment_overrun)
                 AND ABS(pt.labor_overrun) >= ABS(pt.subcontractor_overrun) THEN 'Labor'
                WHEN ABS(pt.materials_overrun) >= ABS(pt.equipment_overrun)
                 AND ABS(pt.materials_overrun) >= ABS(pt.subcontractor_overrun) THEN 'Materials'
                WHEN ABS(pt.equipment_overrun) >= ABS(pt.subcontractor_overrun) THEN 'Equipment'
                ELSE                                                               'Subcontractor'
            END AS primary_overrun_driver
        FROM dim_projects     p
        JOIN project_totals   pt ON p.id = pt.project_id
    ),
    type_benchmarks AS (
        -- Aggregate statistics per project type for peer comparison
        SELECT
            project_type,
            ROUND(AVG(variance_pct), 2)                                    AS avg_variance_pct,
            ROUND(MAX(variance_pct), 2)                                    AS max_variance_pct,
            ROUND(MIN(variance_pct), 2)                                    AS min_variance_pct,
            COUNT(*)                                                        AS type_project_count,
            SUM(CASE WHEN cost_variance > 0 THEN 1 ELSE 0 END)            AS type_over_budget_count
        FROM project_variance
        GROUP BY project_type
    )
    SELECT
        pv.project_name,
        pv.project_type,
        pv.status,
        pv.contract_value,
        pv.total_actual,
        pv.total_budgeted,
        pv.cost_variance,
        pv.variance_pct,
        pv.budget_status,
        pv.primary_overrun_driver,
        pv.labor_overrun,
        pv.materials_overrun,
        pv.equipment_overrun,
        pv.subcontractor_overrun,
        -- Type-level benchmarks for contextual comparison
        tb.avg_variance_pct           AS type_avg_variance_pct,
        tb.max_variance_pct           AS type_max_variance_pct,
        tb.type_project_count,
        tb.type_over_budget_count,
        -- Fleet-wide rank: 1 = worst cost overrun
        RANK() OVER (ORDER BY pv.variance_pct DESC)                       AS overrun_rank,
        -- Within project type rank
        RANK() OVER (
            PARTITION BY pv.project_type
            ORDER BY pv.variance_pct DESC
        )                                                                   AS type_overrun_rank,
        -- Lead/lag to show next/prev project variance in ranked order
        LEAD(pv.variance_pct) OVER (ORDER BY pv.variance_pct DESC)        AS next_project_variance_pct,
        LAG(pv.variance_pct)  OVER (ORDER BY pv.variance_pct DESC)        AS prev_project_variance_pct,
        -- Highlight the five most egregious overruns
        CASE
            WHEN RANK() OVER (ORDER BY pv.variance_pct DESC) <= 5 THEN 'Yes'
            ELSE 'No'
        END                                                                 AS top_5_overrun
    FROM project_variance pv
    JOIN type_benchmarks  tb ON pv.project_type = tb.project_type
    ORDER BY pv.variance_pct DESC
    """
    return _query(engine, sql)


# ---------------------------------------------------------------------------
# CLI test runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    BAR = "=" * 70

    def section(title: str) -> None:
        print(f"\n{BAR}\n  {title}\n{BAR}")

    def show(df: pd.DataFrame, cols: list = None, n: int = 5) -> None:
        display = df[cols] if cols else df
        print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}\n")
        print(display.head(n).to_string(index=False))

    engine = get_engine()

    # Resolve a sample project that has actual cost data
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT p.id, p.project_name
            FROM   dim_projects p
            JOIN   fact_project_costs fpc ON p.id = fpc.project_id
            WHERE  p.status IN ('ACTIVE', 'COMPLETED')
            LIMIT  1
        """)).fetchone()

    if not row:
        print("No project data found — run data_generator.py first.")
        sys.exit(1)

    pid, pname = row[0], row[1]
    DATE_START = date(2023, 1, 1)
    DATE_END   = date(2025, 12, 31)

    # ── 1. Project Performance Summary ────────────────────────────────
    section(f"1. Project Performance Summary — [{pid}] {pname}")
    df1 = get_project_performance_summary(engine, pid)
    show(df1, ["cost_category", "actual_cost", "budgeted_cost",
                "variance_pct", "category_cpi", "budget_status"])
    if not df1.empty:
        r = df1.iloc[0]
        print(f"\n  Overall CPI={r['overall_cpi']}  "
              f"SPI={r['spi']}  "
              f"Schedule Δ={r['schedule_variance_days']} days  "
              f"Total actual=${r['total_actual']:,.0f}")

    # ── 2. Equipment Utilization ───────────────────────────────────────
    section(f"2. Equipment Utilization Analysis  ({DATE_START} → {DATE_END})")
    df2 = get_equipment_utilization_analysis(engine, DATE_START, DATE_END)
    show(df2, ["equipment_name", "equipment_type", "utilization_pct",
                "utilization_tier", "hours_used", "revenue_generated", "net_revenue"])
    underutil = int((df2["is_underutilized"] == 1).sum())
    avg_util  = df2["utilization_pct"].mean()
    print(f"\n  Fleet avg utilization: {avg_util:.1f}%  |  "
          f"Underutilized assets: {underutil} of {len(df2)}")

    # ── 3. Subcontractor Scorecard ─────────────────────────────────────
    section("3. Subcontractor Performance Scorecard")
    df3 = get_subcontractor_performance_scorecard(engine)
    show(df3, ["company_name", "trade", "cpi", "client_rating",
                "composite_score", "overall_rank", "performance_tier", "top_in_trade"])
    top_per_trade = df3[df3["top_in_trade"] == "Yes"][
        ["trade", "company_name", "composite_score", "cpi"]
    ].sort_values("trade")
    print(f"\n  Recommended top performer per trade:")
    print(top_per_trade.to_string(index=False))

    # ── 4. Labor Productivity by Trade ────────────────────────────────
    section("4. Labor Productivity by Trade  (fleet-wide)")
    df4 = get_labor_productivity_by_trade(engine)
    show(df4, ["trade", "total_hours", "total_labor_cost", "effective_cost_per_hour",
                "unique_employees", "pct_of_total_hours", "mom_hours_pct"])
    print(f"\n  Most labour-intensive trade: {df4.iloc[0]['trade']}  "
          f"({df4.iloc[0]['pct_of_total_hours']:.1f}% of total hours)")

    # ── 5. Safety Metrics ─────────────────────────────────────────────
    section(f"5. Safety Metrics  ({DATE_START} → {DATE_END})")
    df5 = get_safety_metrics(engine, DATE_START, DATE_END)
    show(df5, ["severity", "incident_type", "incident_count", "days_lost", "osha_rate_this_type"])
    if not df5.empty:
        r5 = df5.iloc[0]
        print(f"\n  Total incidents : {r5['total_incidents']}  "
              f"(injuries={r5['injuries']}, near-misses={r5['near_misses']}, "
              f"violations={r5['violations']})")
        print(f"  OSHA rate       : {r5['osha_total_incident_rate']}  |  "
              f"Days lost: {r5['total_days_lost']}  |  "
              f"Zero-incident projects: {r5['zero_incident_projects']}")

    # ── 6. Budget Variance Analysis ────────────────────────────────────
    section("6. Budget Variance Analysis  (all projects)")
    df6 = get_budget_variance_analysis(engine)
    show(df6, ["project_name", "project_type", "variance_pct", "budget_status",
                "primary_overrun_driver", "overrun_rank", "top_5_overrun"])

    print("\n  Average variance % by project type:")
    type_summary = (
        df6.groupby("project_type")
           .agg(avg_variance=("variance_pct", "mean"),
                over_budget=("type_over_budget_count", "first"),
                total=("type_project_count", "first"))
           .round(2)
    )
    print(type_summary.to_string())

    top5 = df6[df6["top_5_overrun"] == "Yes"][
        ["project_name", "variance_pct", "budget_status", "primary_overrun_driver"]
    ]
    print(f"\n  Top-5 most over-budget projects:")
    print(top5.to_string(index=False))

    print(f"\n{BAR}")
    print("  All 6 analytics functions completed successfully.")
    print(f"{BAR}\n")
