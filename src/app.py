"""
app.py — Streamlit dashboard for the Construction Analytics Platform.

Run with:
    streamlit run src/app.py
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))

from analytics import (
    get_budget_variance_analysis,
    get_equipment_utilization_analysis,
    get_labor_productivity_by_trade,
    get_project_performance_summary,
    get_safety_metrics,
    get_subcontractor_performance_scorecard,
)
from database import Base, get_engine
from ml_models import (
    COST_OVERRUN_MODEL_PATH,
    EQUIPMENT_MAINTENANCE_MODEL_PATH,
    predict_cost_overrun,
    predict_maintenance_needs,
)

# ---------------------------------------------------------------------------
# App-wide config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Construction Analytics Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOOD       = "#71717a"      # Medium gray
WARNING    = "#ca8a04"      # Muted gold/yellow
BAD        = "#52525b"      # Dark gray
PRIMARY    = "#27272a"      # Almost black
MUTED      = "#d4d4d8"      # Light gray
ACCENT     = "#a1a1aa"      # Medium-light gray
BACKGROUND = "#fafafa"      # Off-white
THEME      = "plotly"       # Clean white theme
TODAY     = date(2026, 2, 17)
DEF_START = date(2023, 1, 1)
DEF_END   = date(2025, 12, 31)

STATUS_COLORS = {
    "ACTIVE":    "#27272a",  # Almost black
    "COMPLETED": "#71717a",  # Medium gray
    "PLANNING":  "#d4d4d8",  # Light gray
    "ON_HOLD":   "#ca8a04",  # Muted yellow
}

TIER_COLORS = {
    "Top Performer":     "#27272a",  # Almost black
    "Recommended":       "#52525b",  # Dark gray
    "Above Average":     "#71717a",  # Medium gray
    "Average":           "#a1a1aa",  # Medium-light gray
    "Needs Improvement": "#d4d4d8",  # Light gray
}

# ---------------------------------------------------------------------------
# Engine (cached as a resource — one connection pool per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_engine_cached():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


@st.cache_resource
def _load_cost_overrun_artifact():
    """Load the cost overrun model artifact once and share across sessions."""
    if not COST_OVERRUN_MODEL_PATH.exists():
        return None
    return joblib.load(COST_OVERRUN_MODEL_PATH)


@st.cache_resource
def _load_maintenance_artifact():
    """Load the maintenance model artifact once and share across sessions."""
    if not EQUIPMENT_MAINTENANCE_MODEL_PATH.exists():
        return None
    return joblib.load(EQUIPMENT_MAINTENANCE_MODEL_PATH)


# ---------------------------------------------------------------------------
# Cached data loaders  (_engine prefix skips hashing the engine object)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_kpis(_engine):
    sql = """
    WITH ps AS (
        SELECT
            COUNT(CASE WHEN status = 'ACTIVE'    THEN 1 END) AS active_projects,
            COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) AS completed_projects,
            COUNT(CASE WHEN status = 'PLANNING'  THEN 1 END) AS planning_projects,
            COUNT(CASE WHEN status = 'ON_HOLD'   THEN 1 END) AS on_hold_projects,
            SUM(CASE WHEN status = 'ACTIVE' THEN contract_value ELSE 0 END)
                AS active_contract_value,
            SUM(contract_value) AS total_contract_value
        FROM dim_projects
    ),
    cs AS (
        SELECT ROUND(SUM(budgeted_amount) / NULLIF(SUM(amount), 0), 3) AS portfolio_cpi
        FROM fact_project_costs
    )
    SELECT ps.*, cs.portfolio_cpi FROM ps CROSS JOIN cs
    """
    with _engine.connect() as conn:
        row = conn.execute(text(sql)).fetchone()
    return dict(zip(row._mapping.keys(), row)) if row else {}


@st.cache_data(ttl=300)
def load_status_dist(_engine):
    with _engine.connect() as conn:
        result = conn.execute(
            text("SELECT status, COUNT(*) AS count FROM dim_projects GROUP BY status")
        )
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


@st.cache_data(ttl=300)
def load_project_list(_engine):
    sql = """
    SELECT id, project_name, status, project_type, contract_value, location
    FROM   dim_projects
    WHERE  status IN ('ACTIVE', 'COMPLETED', 'ON_HOLD')
      AND  actual_start_date IS NOT NULL
    ORDER BY
        CASE status WHEN 'ACTIVE' THEN 1 WHEN 'ON_HOLD' THEN 2 ELSE 3 END,
        project_name
    """
    with _engine.connect() as conn:
        result = conn.execute(text(sql))
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


@st.cache_data(ttl=300)
def load_project_perf(_engine, project_id: int):
    return get_project_performance_summary(_engine, project_id)


@st.cache_data(ttl=300)
def load_budget_variance(_engine):
    return get_budget_variance_analysis(_engine)


@st.cache_data(ttl=300)
def load_equipment_util(_engine, start: date, end: date):
    return get_equipment_utilization_analysis(_engine, start, end)


@st.cache_data(ttl=300)
def load_subcontractors(_engine):
    df = get_subcontractor_performance_scorecard(_engine)
    return df


@st.cache_data(ttl=300)
def load_labor(_engine, project_id=None):
    return get_labor_productivity_by_trade(_engine, project_id)


@st.cache_data(ttl=300)
def load_safety(_engine, start: date, end: date):
    return get_safety_metrics(_engine, start, end)


@st.cache_data(ttl=300)
def load_cost_overrun_predictions(_engine):
    """
    Run cost overrun predictions for every ACTIVE project that has cost data.
    Returns a DataFrame ready for display; one row per project.
    """
    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT p.id, p.project_name, p.project_type, p.contract_value
            FROM   dim_projects p
            JOIN   fact_project_costs fpc ON p.id = fpc.project_id
            WHERE  p.status = 'ACTIVE'
              AND  p.actual_start_date IS NOT NULL
            GROUP  BY p.id
            ORDER  BY p.project_name
        """))
        projects = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

    if projects.empty:
        return pd.DataFrame()

    rows = []
    for _, proj in projects.iterrows():
        try:
            r = predict_cost_overrun(_engine, int(proj["id"]))
            prob = r["overrun_probability"]
            rows.append({
                "Project Name":        proj["project_name"],
                "Type":                proj["project_type"].title(),
                "Contract Value":      float(proj["contract_value"]),
                "Overrun Probability": prob,
                "Prediction":          r["prediction"],
                "Risk Level":          (
                    "High"   if prob > 0.70 else
                    "Medium" if prob >= 0.30 else
                    "Low"
                ),
                "Recommendation":      r["recommendation"],
                "_project_id":         int(proj["id"]),
            })
        except Exception:
            pass

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_maintenance_predictions(_engine):
    """
    Run maintenance predictions for every non-RETIRED equipment piece.
    Returns a DataFrame ready for display; one row per equipment.
    """
    with _engine.connect() as conn:
        result = conn.execute(text("""
            SELECT e.id, e.equipment_name, e.equipment_type,
                   e.purchase_date, e.status
            FROM   dim_equipment e
            WHERE  e.status != 'RETIRED'
            ORDER  BY e.equipment_name
        """))
        equip = pd.DataFrame(result.fetchall(), columns=list(result.keys()))

    if equip.empty:
        return pd.DataFrame()

    rows = []
    today_ts = pd.Timestamp(TODAY)
    for _, eq in equip.iterrows():
        try:
            r = predict_maintenance_needs(_engine, int(eq["id"]))
            prob = r["maintenance_probability"]
            days = r["predicted_days_until_maintenance"]
            feat = r["features"]

            # Age in years for display
            pdate    = pd.to_datetime(eq["purchase_date"])
            age_days = max((today_ts - pdate).days, 0)

            # Risk: Critical if prob>70% OR <7 days; Watch if 50-70% OR ≤14 days
            if prob > 0.70 or days < 7:
                risk = "Critical"
            elif prob >= 0.50 or days <= 14:
                risk = "Watch"
            else:
                risk = "Routine"

            rows.append({
                "Equipment Name":   eq["equipment_name"],
                "Type":             eq["equipment_type"].title(),
                "Age (days)":       age_days,
                "Total Hours":      float(feat.get("total_hours_used", 0)),
                "Maint. Prob.":     prob,
                "Est. Days":        days,
                "Risk":             risk,
                "Status":           eq["status"].replace("_", " ").title(),
                "_equipment_id":    int(eq["id"]),
            })
        except Exception:
            pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------


def _cpi_color(val):
    if pd.isna(val):
        return ""
    if val >= 1.0:
        return f"color: {GOOD}; font-weight:600"
    if val >= 0.90:
        return f"color: {WARNING}"
    return f"color: {BAD}; font-weight:600"


def _variance_color(val):
    if pd.isna(val):
        return ""
    if val <= 0:
        return f"color: {GOOD}"
    if val <= 15:
        return f"color: {WARNING}"
    return f"color: {BAD}; font-weight:600"


def _util_row_color(row):
    pct = row.get("utilization_pct", 100)
    if pct < 50:
        return [f"background-color: {BAD}18"] * len(row)
    if pct < 75:
        return [f"background-color: {WARNING}18"] * len(row)
    return [""] * len(row)


def fmt_currency(val):
    if pd.isna(val):
        return "—"
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.0f}K"
    return f"${val:.0f}"


def cpi_delta_color(cpi):
    """Return Streamlit metric delta_color based on CPI value."""
    return "normal"   # positive delta = green (under budget) by default


# ---------------------------------------------------------------------------
# PAGE 1 — Executive Dashboard
# ---------------------------------------------------------------------------


def page_executive(engine, date_start: date, date_end: date):
    st.title("📊 Executive Dashboard")
    st.caption(f"Portfolio overview · As of {TODAY.strftime('%B %d, %Y')}")

    kpis = load_kpis(engine)
    if not kpis:
        st.warning("No data found. Run `python src/data_generator.py` to seed the database.")
        return

    df_equip = load_equipment_util(engine, date_start, date_end)

    # ── KPI Row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Active Projects", kpis.get("active_projects", 0))
    with c2:
        val = kpis.get("active_contract_value", 0)
        st.metric("Active Contract Value", f"${val/1e6:.1f}M",
                  help="Sum of contract values for all active projects")
    with c3:
        cpi = kpis.get("portfolio_cpi") or 0.0
        st.metric("Portfolio CPI", f"{cpi:.3f}",
                  delta=round(cpi - 1.0, 3),
                  help="Cost Performance Index — >1.0 means under budget")
    with c4:
        if not df_equip.empty:
            avg_util = df_equip["utilization_pct"].mean()
            st.metric("Fleet Utilization", f"{avg_util:.1f}%",
                      delta=f"{avg_util - 65:.1f}%",
                      delta_color="normal",
                      help="Average utilization vs 65% internal target")
        else:
            st.metric("Fleet Utilization", "N/A")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Project Status Breakdown")
        df_status = load_status_dist(engine)
        if not df_status.empty:
            fig = px.pie(
                df_status, values="count", names="status",
                color="status", color_discrete_map=STATUS_COLORS,
                template=THEME, hole=0.45,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=13)
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Budget Performance by Project Type")
        df_budget = load_budget_variance(engine)
        if not df_budget.empty:
            type_avg = (
                df_budget.groupby("project_type", as_index=False)
                .agg(avg_variance=("variance_pct", "mean"),
                     project_count=("project_name", "count"))
                .round(2)
            )
            type_avg["bar_color"] = type_avg["avg_variance"].apply(
                lambda x: GOOD if x <= 0 else (WARNING if x <= 15 else BAD)
            )
            fig = px.bar(
                type_avg, x="project_type", y="avg_variance",
                color="bar_color", color_discrete_map="identity",
                text=type_avg["avg_variance"].apply(lambda x: f"{x:+.1f}%"),
                template=THEME,
                labels={"project_type": "Project Type",
                        "avg_variance": "Avg Budget Variance %"},
            )
            fig.update_traces(textposition="outside")
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10),
                              yaxis_title="Avg Variance %")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Bottom Row: Top overruns + Safety ─────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("🔴 Top 5 Over-Budget Projects")
        if not df_budget.empty:
            top5 = df_budget[df_budget["top_5_overrun"] == "Yes"][[
                "project_name", "project_type", "variance_pct",
                "cost_variance", "budget_status", "primary_overrun_driver",
            ]].copy()
            top5["cost_variance"] = top5["cost_variance"].apply(fmt_currency)
            styled = (
                top5.rename(columns={
                    "project_name": "Project", "project_type": "Type",
                    "variance_pct": "Variance %", "cost_variance": "Overrun $",
                    "budget_status": "Status", "primary_overrun_driver": "Driver",
                })
                .style
                .applymap(_variance_color, subset=["Variance %"])
                .format({"Variance %": "{:+.1f}%"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

    with col_r:
        st.subheader("⚠️ Safety Overview")
        df_safety = load_safety(engine, date_start, date_end)
        if not df_safety.empty:
            r = df_safety.iloc[0]
            sa, sb = st.columns(2)
            with sa:
                st.metric("Total Incidents", int(r["total_incidents"]))
                st.metric("Days Lost", int(r["total_days_lost"]))
            with sb:
                st.metric("OSHA Rate", f"{r['osha_total_incident_rate']:.2f}",
                          help="Incidents per 200,000 man-hours (OSHA standard)")
                st.metric("Zero-Incident Projects",
                          int(r["zero_incident_projects"]))
        else:
            st.info("No safety incidents recorded for the selected period.")


# ---------------------------------------------------------------------------
# PAGE 2 — Project Deep Dive
# ---------------------------------------------------------------------------


def page_project_deep_dive(engine):
    st.title("🔍 Project Deep Dive")

    df_projects = load_project_list(engine)
    if df_projects.empty:
        st.warning("No project data found. Run the data generator first.")
        return

    # Build label for dropdown: "Project Name (status)"
    df_projects["label"] = (
        df_projects["project_name"]
        + "  ["
        + df_projects["status"].str.upper()
        + "]"
    )
    selected_label = st.selectbox(
        "Select a project", df_projects["label"].tolist(), index=0
    )
    project_row = df_projects[df_projects["label"] == selected_label].iloc[0]
    project_id  = int(project_row["id"])

    st.divider()

    # ── Project info cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Project Type**")
        st.markdown(f"`{project_row['project_type'].title()}`")
    with c2:
        st.markdown("**Contract Value**")
        st.markdown(f"**{fmt_currency(project_row['contract_value'])}**")
    with c3:
        st.markdown("**Status**")
        color = STATUS_COLORS.get(project_row["status"], MUTED)
        st.markdown(
            f"<span style='color:{color}; font-weight:700'>"
            f"{project_row['status'].replace('_', ' ').title()}</span>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown("**Location**")
        st.markdown(project_row["location"] or "—")

    st.divider()

    df_perf = load_project_perf(engine, project_id)
    if df_perf.empty:
        st.warning("No cost data available for this project.")
        return

    r = df_perf.iloc[0]

    # ── KPI Metrics ───────────────────────────────────────────────────────
    st.subheader("Performance Indicators")
    m1, m2, m3, m4 = st.columns(4)
    cpi = float(r["overall_cpi"] or 0)
    spi = float(r["spi"] or 0)
    sv  = int(r["schedule_variance_days"] or 0)
    tv  = float(r["total_variance"] or 0)

    with m1:
        st.metric("Overall CPI", f"{cpi:.3f}", delta=round(cpi - 1.0, 3),
                  help="Budgeted / Actual cost  (>1.0 = under budget)")
    with m2:
        st.metric("SPI (proxy)", f"{spi:.3f}", delta=round(spi - 1.0, 3),
                  help="Planned duration / Elapsed days  (>1.0 = ahead of schedule)")
    with m3:
        st.metric("Schedule Variance", f"{sv:+d} days", delta=-sv,
                  delta_color="inverse",
                  help="Positive = behind schedule")
    with m4:
        st.metric("Total Cost Variance", fmt_currency(tv), delta=tv,
                  help="Positive = under budget (good)")

    st.divider()

    # ── Cost breakdown + Timeline ─────────────────────────────────────────
    col_chart, col_timeline = st.columns(2)

    with col_chart:
        st.subheader("Actual vs Budgeted Cost by Category")
        df_cost = df_perf[["cost_category", "actual_cost", "budgeted_cost"]].copy()
        df_melted = df_cost.melt(
            id_vars="cost_category", var_name="Type", value_name="Amount"
        )
        df_melted["Type"] = df_melted["Type"].map(
            {"actual_cost": "Actual", "budgeted_cost": "Budgeted"}
        )
        fig = px.bar(
            df_melted, x="cost_category", y="Amount", color="Type",
            barmode="group",
            color_discrete_map={"Actual": BAD, "Budgeted": PRIMARY},
            template=THEME,
            labels={"cost_category": "Category", "Amount": "Cost ($)"},
        )
        fig.update_layout(margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_timeline:
        st.subheader("Project Timeline")
        planned_start = r["planned_start_date"]
        planned_end   = r["planned_end_date"]
        actual_start  = r["actual_start_date"]
        actual_end    = r["actual_end_date"]

        tl_rows = []
        if planned_start and planned_end:
            tl_rows.append({
                "Phase": "Planned",
                "Start": pd.Timestamp(str(planned_start)),
                "End":   pd.Timestamp(str(planned_end)),
            })
        if actual_start:
            end_ts = pd.Timestamp(str(actual_end or TODAY))
            tl_rows.append({
                "Phase": "Actual",
                "Start": pd.Timestamp(str(actual_start)),
                "End":   end_ts,
            })

        if tl_rows:
            df_tl = pd.DataFrame(tl_rows)
            fig_tl = px.timeline(
                df_tl, x_start="Start", x_end="End", y="Phase",
                color="Phase",
                color_discrete_map={"Planned": PRIMARY, "Actual": GOOD},
                template=THEME,
            )
            fig_tl.update_yaxes(autorange="reversed")
            fig_tl.update_layout(margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("No date information available for timeline.")

    # ── Detailed cost table ───────────────────────────────────────────────
    st.subheader("Cost Detail by Category")
    display_cols = ["cost_category", "actual_cost", "budgeted_cost",
                    "variance_pct", "category_cpi", "budget_status", "pct_of_total_spend"]
    df_display = df_perf[display_cols].copy()
    styled = (
        df_display
        .rename(columns={
            "cost_category": "Category", "actual_cost": "Actual ($)",
            "budgeted_cost": "Budgeted ($)", "variance_pct": "Variance %",
            "category_cpi": "CPI", "budget_status": "Status",
            "pct_of_total_spend": "% of Total",
        })
        .style
        .applymap(_cpi_color, subset=["CPI"])
        .applymap(_variance_color, subset=["Variance %"])
        .format({
            "Actual ($)": "${:,.0f}", "Budgeted ($)": "${:,.0f}",
            "Variance %": "{:+.1f}%",  "CPI": "{:.3f}",
            "% of Total": "{:.1f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Equipment Analytics
# ---------------------------------------------------------------------------


def page_equipment(engine, date_start: date, date_end: date):
    st.title("🚧 Equipment Analytics")

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        date_start = st.date_input("From", value=date_start,
                                   min_value=date(2022, 1, 1), max_value=TODAY)
    with col_d2:
        date_end = st.date_input("To", value=date_end,
                                 min_value=date(2022, 1, 1), max_value=TODAY)

    df = load_equipment_util(engine, date_start, date_end)
    if df.empty:
        st.warning("No equipment data for this period.")
        return

    st.divider()

    # ── Summary KPIs ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    avg_util     = df["utilization_pct"].mean()
    total_rev    = df["revenue_generated"].sum()
    total_net    = df["net_revenue"].sum()
    underutil_n  = int((df["is_underutilized"] == 1).sum())

    with m1:
        st.metric("Fleet Avg Utilization", f"{avg_util:.1f}%",
                  delta=f"{avg_util - 65:.1f}%",
                  help="vs 65% internal target")
    with m2:
        st.metric("Total Revenue Generated", fmt_currency(total_rev))
    with m3:
        st.metric("Net Revenue (after costs)", fmt_currency(total_net))
    with m4:
        st.metric("Underutilized Assets (<50%)", underutil_n,
                  delta=f"-{underutil_n}" if underutil_n > 0 else "0",
                  delta_color="inverse")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────
    col_bar, col_pie = st.columns(2)

    with col_bar:
        st.subheader("Avg Utilization % by Equipment Type")
        type_summary = (
            df.groupby("equipment_type", as_index=False)
            .agg(avg_util=("utilization_pct", "mean"),
                 count=("equipment_name", "count"),
                 avg_revenue=("revenue_generated", "mean"))
            .round(2)
        )
        type_summary["bar_color"] = type_summary["avg_util"].apply(
            lambda x: GOOD if x >= 75 else (WARNING if x >= 50 else BAD)
        )
        fig = px.bar(
            type_summary.sort_values("avg_util", ascending=True),
            x="avg_util", y="equipment_type", orientation="h",
            color="bar_color", color_discrete_map="identity",
            text=type_summary.sort_values("avg_util")["avg_util"].apply(
                lambda x: f"{x:.1f}%"
            ),
            template=THEME,
            labels={"avg_util": "Avg Utilization %", "equipment_type": "Type"},
        )
        fig.add_vline(x=50, line_dash="dash", line_color=WARNING, opacity=0.7,
                      annotation_text="50% threshold")
        fig.update_traces(textposition="inside")
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_pie:
        st.subheader("Fleet Utilization Tier Distribution")
        tier_counts = df["utilization_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_color_map = {
            "High (≥75%)":       GOOD,
            "Moderate (50–75%)": PRIMARY,
            "Low (<50%)":        WARNING,
            "Idle":              BAD,
        }
        fig_pie = px.pie(
            tier_counts, values="Count", names="Tier",
            color="Tier", color_discrete_map=tier_color_map,
            template=THEME, hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Full Equipment Table ───────────────────────────────────────────────
    st.subheader("Full Equipment Utilization Report")
    st.caption("🔴 Red rows = underutilized (<50%)  ·  🟡 Yellow = moderate (50–75%)")

    display_df = df[[
        "equipment_name", "equipment_type", "status", "hourly_rate",
        "hours_used", "available_hours", "utilization_pct", "utilization_tier",
        "revenue_generated", "fuel_cost", "maintenance_cost", "net_revenue",
        "projects_served", "rank_in_type",
    ]].copy()

    styled = (
        display_df
        .rename(columns={
            "equipment_name": "Equipment", "equipment_type": "Type",
            "status": "Status", "hourly_rate": "$/hr",
            "hours_used": "Hours Used", "available_hours": "Available Hrs",
            "utilization_pct": "Util %", "utilization_tier": "Tier",
            "revenue_generated": "Revenue", "fuel_cost": "Fuel",
            "maintenance_cost": "Maintenance", "net_revenue": "Net Revenue",
            "projects_served": "Projects", "rank_in_type": "Rank in Type",
        })
        .style
        .apply(_util_row_color, axis=1)
        .format({
            "$/hr": "${:.0f}", "Hours Used": "{:.1f}", "Available Hrs": "{:,.0f}",
            "Util %": "{:.1f}%", "Revenue": "${:,.0f}",
            "Fuel": "${:,.0f}", "Maintenance": "${:,.0f}", "Net Revenue": "${:,.0f}",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=450)


# ---------------------------------------------------------------------------
# PAGE 4 — Subcontractor Performance
# ---------------------------------------------------------------------------


def page_subcontractor(engine):
    st.title("🤝 Subcontractor Performance")

    df = load_subcontractors(engine)
    if df.empty:
        st.warning("No subcontractor performance data found.")
        return

    # ── Filters ───────────────────────────────────────────────────────────
    trades = ["All Trades"] + sorted(df["trade"].unique().tolist())
    col_f1, col_f2 = st.columns([2, 4])
    with col_f1:
        selected_trade = st.selectbox("Filter by Trade", trades)

    df_filtered = df if selected_trade == "All Trades" else df[df["trade"] == selected_trade]

    st.divider()

    # ── KPI Row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Subcontractors Evaluated", len(df_filtered))
    with m2:
        avg_cpi = df_filtered["cpi"].mean()
        st.metric("Avg CPI", f"{avg_cpi:.3f}", delta=round(avg_cpi - 1.0, 3))
    with m3:
        top_perf_n = (df_filtered["performance_tier"] == "Top Performer").sum()
        st.metric("Top Performers", int(top_perf_n))
    with m4:
        needs_imp  = (df_filtered["performance_tier"] == "Needs Improvement").sum()
        st.metric("Needs Improvement", int(needs_imp),
                  delta=f"-{needs_imp}" if needs_imp > 0 else "0",
                  delta_color="inverse")

    st.divider()

    # ── Charts Row ────────────────────────────────────────────────────────
    col_hist, col_tier = st.columns(2)

    with col_hist:
        st.subheader("CPI Distribution")
        fig = px.histogram(
            df_filtered, x="cpi", nbins=25,
            color_discrete_sequence=[PRIMARY],
            template=THEME,
            labels={"cpi": "CPI (Budgeted / Actual)"},
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color=GOOD, opacity=0.8,
                      annotation_text="CPI = 1.0 (on budget)")
        fig.update_layout(margin=dict(t=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with col_tier:
        st.subheader("Performance Tier Breakdown")
        tier_counts = df_filtered["performance_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_pie = px.pie(
            tier_counts, values="Count", names="Tier",
            color="Tier", color_discrete_map=TIER_COLORS,
            template=THEME, hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
        fig_pie.update_layout(showlegend=True,
                              legend=dict(orientation="h", y=-0.2),
                              margin=dict(t=10, b=40))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Top Performer per Trade table ─────────────────────────────────────
    st.subheader("⭐ Recommended Top Performer per Trade")
    top_per_trade = (
        df[df["top_in_trade"] == "Yes"][[
            "trade", "company_name", "composite_score", "cpi",
            "client_rating", "projects_count", "performance_tier",
        ]]
        .sort_values("trade")
    )
    styled_top = (
        top_per_trade
        .rename(columns={
            "trade": "Trade", "company_name": "Company",
            "composite_score": "Score", "cpi": "CPI",
            "client_rating": "Rating", "projects_count": "Projects",
            "performance_tier": "Tier",
        })
        .style
        .applymap(_cpi_color, subset=["CPI"])
        .format({"Score": "{:.2f}", "CPI": "{:.3f}", "Rating": "{:.1f}"})
    )
    st.dataframe(styled_top, use_container_width=True, hide_index=True)

    st.divider()

    # ── Full Scorecard ─────────────────────────────────────────────────────
    st.subheader(f"Full Scorecard — {selected_trade}")
    display_cols = [
        "overall_rank", "company_name", "trade", "cpi", "budget_adherence_pct",
        "client_rating", "years_in_business", "projects_count",
        "composite_score", "performance_tier", "rank_in_trade",
    ]
    styled_full = (
        df_filtered[display_cols]
        .rename(columns={
            "overall_rank": "Rank", "company_name": "Company", "trade": "Trade",
            "cpi": "CPI", "budget_adherence_pct": "Budget Adherence %",
            "client_rating": "Rating", "years_in_business": "Years in Biz",
            "projects_count": "Projects", "composite_score": "Composite Score",
            "performance_tier": "Tier", "rank_in_trade": "Trade Rank",
        })
        .style
        .applymap(_cpi_color, subset=["CPI"])
        .format({
            "CPI": "{:.3f}", "Budget Adherence %": "{:+.1f}%",
            "Rating": "{:.1f}", "Composite Score": "{:.2f}",
        })
    )
    st.dataframe(styled_full, use_container_width=True, hide_index=True, height=400)


# ---------------------------------------------------------------------------
# PAGE 5 — Labor & Safety
# ---------------------------------------------------------------------------


def page_labor_safety(engine, date_start: date, date_end: date):
    st.title("👷 Labor & Safety")

    # ── Safety date range controls ────────────────────────────────────────
    st.subheader("📅 Analysis Period")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_start = st.date_input("From", value=date_start,
                                   min_value=date(2022, 1, 1), max_value=TODAY,
                                   key="ls_start")
    with col_d2:
        date_end = st.date_input("To", value=date_end,
                                 min_value=date(2022, 1, 1), max_value=TODAY,
                                 key="ls_end")

    df_labor  = load_labor(engine)
    df_safety = load_safety(engine, date_start, date_end)

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    #  LABOR section
    # ════════════════════════════════════════════════════════════════════
    st.header("Labor Productivity by Trade")

    if df_labor.empty:
        st.warning("No labor data available.")
    else:
        # ── Labor KPIs ────────────────────────────────────────────────────
        lm1, lm2, lm3, lm4 = st.columns(4)
        with lm1:
            st.metric("Total Labor Hours", f"{df_labor['total_hours'].sum():,.0f}")
        with lm2:
            st.metric("Total Labor Cost",
                      fmt_currency(df_labor["total_labor_cost"].sum()))
        with lm3:
            avg_rate = df_labor["effective_cost_per_hour"].mean()
            st.metric("Avg Effective Rate", f"${avg_rate:.2f}/hr")
        with lm4:
            st.metric("Trades Active", len(df_labor))

        col_labor_bar, col_labor_tbl = st.columns(2)

        with col_labor_bar:
            st.subheader("Labor Cost by Trade")
            fig_labor = px.bar(
                df_labor.sort_values("total_labor_cost"),
                x="total_labor_cost", y="trade", orientation="h",
                color="pct_of_total_cost",
                color_continuous_scale="Blues",
                template=THEME,
                text=df_labor.sort_values("total_labor_cost")["pct_of_total_cost"]
                    .apply(lambda x: f"{x:.1f}%"),
                labels={
                    "total_labor_cost": "Total Labor Cost ($)",
                    "trade": "Trade",
                    "pct_of_total_cost": "% of Total",
                },
            )
            fig_labor.update_traces(textposition="inside")
            fig_labor.update_layout(margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig_labor, use_container_width=True)

        with col_labor_tbl:
            st.subheader("Productivity Detail")
            tbl_cols = [
                "trade", "total_hours", "total_labor_cost",
                "effective_cost_per_hour", "unique_employees",
                "avg_hours_per_employee", "mom_hours_pct",
            ]
            # MoM trend indicator column
            df_tbl = df_labor[tbl_cols].copy()
            df_tbl["trend"] = df_tbl["mom_hours_pct"].apply(
                lambda x: "▲" if (pd.notna(x) and x > 0) else ("▼" if pd.notna(x) else "—")
            )

            def color_mom(val):
                if pd.isna(val):
                    return ""
                return f"color: {GOOD}" if val > 0 else f"color: {BAD}"

            styled_labor = (
                df_tbl
                .rename(columns={
                    "trade": "Trade", "total_hours": "Total Hrs",
                    "total_labor_cost": "Total Cost",
                    "effective_cost_per_hour": "Eff. Rate/hr",
                    "unique_employees": "Workers",
                    "avg_hours_per_employee": "Avg Hrs/Worker",
                    "mom_hours_pct": "MoM Hrs %",
                    "trend": "Trend",
                })
                .style
                .applymap(color_mom, subset=["MoM Hrs %"])
                .format({
                    "Total Hrs": "{:,.0f}",
                    "Total Cost": "${:,.0f}",
                    "Eff. Rate/hr": "${:.2f}",
                    "Avg Hrs/Worker": "{:.1f}",
                    "MoM Hrs %": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                })
            )
            st.dataframe(styled_labor, use_container_width=True,
                         hide_index=True, height=380)

    st.divider()

    # ════════════════════════════════════════════════════════════════════
    #  SAFETY section
    # ════════════════════════════════════════════════════════════════════
    st.header("Safety Performance")

    if df_safety.empty:
        st.info("No safety incidents recorded for the selected period. 🟢")
    else:
        r = df_safety.iloc[0]

        # ── Safety KPIs ───────────────────────────────────────────────────
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("Total Incidents", int(r["total_incidents"]),
                      delta=f"-{r['severe_count']} severe",
                      delta_color="inverse")
        with sm2:
            st.metric("OSHA Incident Rate", f"{r['osha_total_incident_rate']:.4f}",
                      help="Incidents per 200,000 man-hours (OSHA 300 Log standard)")
        with sm3:
            st.metric("Days Lost to Injuries", int(r["total_days_lost"]),
                      delta=f"-{r['total_days_lost']} productivity days" if r["total_days_lost"] > 0 else "0",
                      delta_color="inverse")
        with sm4:
            st.metric("Zero-Incident Projects", int(r["zero_incident_projects"]),
                      help="Active/completed projects with no incidents in this period")

        st.divider()

        col_sev, col_type = st.columns(2)

        with col_sev:
            st.subheader("Incidents by Severity")
            sev_order = {"severe": 1, "moderate": 2, "minor": 3}
            df_sev = (
                df_safety.groupby("severity", as_index=False)
                ["incident_count"].sum()
                .assign(order=lambda d: d["severity"].map(sev_order))
                .sort_values("order")
            )
            sev_colors = {"severe": BAD, "moderate": WARNING, "minor": GOOD}
            fig_sev = px.bar(
                df_sev, x="severity", y="incident_count",
                color="severity", color_discrete_map=sev_colors,
                template=THEME,
                text="incident_count",
                labels={"severity": "Severity", "incident_count": "Incidents"},
            )
            fig_sev.update_traces(textposition="outside")
            fig_sev.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig_sev, use_container_width=True)

        with col_type:
            st.subheader("Incidents by Type")
            df_type = df_safety.groupby("incident_type", as_index=False)[
                "incident_count"
            ].sum()
            type_colors = {
                "injury":    BAD,
                "near_miss": WARNING,
                "violation": PRIMARY,
            }
            fig_type = px.pie(
                df_type, values="incident_count", names="incident_type",
                color="incident_type", color_discrete_map=type_colors,
                template=THEME, hole=0.4,
            )
            fig_type.update_traces(textinfo="percent+label", textfont_size=13)
            fig_type.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_type, use_container_width=True)

        # ── Stacked bar: severity × type ─────────────────────────────────
        st.subheader("Incident Detail — Severity × Type")
        fig_stacked = px.bar(
            df_safety, x="incident_type", y="incident_count",
            color="severity",
            color_discrete_map=sev_colors,
            barmode="stack",
            template=THEME,
            text="incident_count",
            labels={"incident_type": "Incident Type",
                    "incident_count": "Count",
                    "severity": "Severity"},
        )
        fig_stacked.update_traces(textposition="inside")
        fig_stacked.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_stacked, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 6 — ML Predictions
# ---------------------------------------------------------------------------

# Risk colour palettes used in both sections
_RISK_COLORS = {
    "High":     "#c0392b",
    "Medium":   "#ca8a04",
    "Low":      "#2d6a4f",
    "Critical": "#c0392b",
    "Watch":    "#ca8a04",
    "Routine":  "#2d6a4f",
}


def _risk_text_style(val: str) -> str:
    color = _RISK_COLORS.get(val, "")
    return f"color: {color}; font-weight: 700" if color else ""


def _prob_bg_style(val: float) -> str:
    if pd.isna(val):
        return ""
    if val > 0.70:
        return "background-color: #c0392b1a; color: #c0392b; font-weight: 700"
    if val >= 0.30:
        return "background-color: #ca8a041a; color: #ca8a04"
    return "background-color: #2d6a4f1a; color: #2d6a4f"


def _days_style(val: float) -> str:
    if pd.isna(val):
        return ""
    if val < 7:
        return "color: #c0392b; font-weight: 700"
    if val <= 14:
        return "color: #ca8a04"
    return ""


def _fi_chart(fi_df: pd.DataFrame, name_map: dict) -> "go.Figure":
    """Horizontal bar chart for feature importances."""
    fi = fi_df.copy()
    fi["feature"] = fi["feature"].map(name_map).fillna(fi["feature"])
    fi = fi.sort_values("importance")
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale=[[0, MUTED], [0.5, ACCENT], [1, PRIMARY]],
        text=fi["importance"].apply(lambda x: f"{x:.3f}"),
        template=THEME,
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=10, b=10),
    )
    return fig


def _metrics_expander(label: str, artifact: dict, caption: str) -> None:
    """Render train/test metrics inside a Streamlit expander."""
    with st.expander(f"Model Performance Metrics — {label}"):
        col_tr, col_te = st.columns(2)
        with col_tr:
            st.markdown("**Train set**")
            for k, v in artifact["train_metrics"].items():
                val = f"{v:.4f}" if v is not None else "N/A"
                st.markdown(f"- **{k.replace('_', ' ').title()}:** `{val}`")
        with col_te:
            st.markdown("**Test set (held-out 25%)**")
            for k, v in artifact["test_metrics"].items():
                val = f"{v:.4f}" if v is not None else "N/A"
                st.markdown(f"- **{k.replace('_', ' ').title()}:** `{val}`")
        st.caption(caption)


def page_ml_predictions(engine):
    st.title("ML Predictions")
    st.caption(
        "Predictions generated by RandomForestClassifier models trained on historical project data. "
        "Run `python src/ml_models.py` to retrain."
    )

    # ── Guard: ensure models exist on disk ────────────────────────────────
    cost_ok  = COST_OVERRUN_MODEL_PATH.exists()
    maint_ok = EQUIPMENT_MAINTENANCE_MODEL_PATH.exists()

    if not cost_ok or not maint_ok:
        st.warning(
            "One or both ML models are not trained yet. "
            "Run `python src/ml_models.py` from the project root to train and save them."
        )
        if not cost_ok:
            st.error(f"Missing: `{COST_OVERRUN_MODEL_PATH}`")
        if not maint_ok:
            st.error(f"Missing: `{EQUIPMENT_MAINTENANCE_MODEL_PATH}`")
        return

    # Load cached artifacts (for fi charts + metrics; avoids repeated disk I/O)
    artifact_cost  = _load_cost_overrun_artifact()
    artifact_maint = _load_maintenance_artifact()

    if artifact_cost is None or artifact_maint is None:
        st.error("Failed to load model artifacts. Try retraining with `python src/ml_models.py`.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Cost Overrun Predictions
    # ════════════════════════════════════════════════════════════════════════
    st.header("Cost Overrun Risk — Active Projects")
    st.caption(
        "Predicts whether each active project will exceed its budget by >10%. "
        "Uses contract value, project type, planned duration, square footage, "
        "and early-phase CPI as features."
    )

    with st.spinner("Running cost overrun predictions…"):
        df_cost = load_cost_overrun_predictions(engine)

    if df_cost.empty:
        st.info(
            "No active projects with sufficient cost data found. "
            "Ensure data_generator.py has been run and active projects exist."
        )
    else:
        # ── KPI row ───────────────────────────────────────────────────────
        n_high = int((df_cost["Risk Level"] == "High").sum())
        n_med  = int((df_cost["Risk Level"] == "Medium").sum())
        n_low  = int((df_cost["Risk Level"] == "Low").sum())
        avg_p  = df_cost["Overrun Probability"].mean()

        ck1, ck2, ck3, ck4 = st.columns(4)
        with ck1:
            st.metric("Projects Analyzed", len(df_cost))
        with ck2:
            st.metric(
                "High Risk (>70%)", n_high,
                delta=f"-{n_high}" if n_high else "0",
                delta_color="inverse",
            )
        with ck3:
            st.metric("Medium Risk (30–70%)", n_med)
        with ck4:
            st.metric("Avg Overrun Probability", f"{avg_p:.1%}")

        st.divider()

        # ── Predictions table ─────────────────────────────────────────────
        st.subheader("Project Risk Assessment")

        display_cols = [
            "Project Name", "Type", "Contract Value",
            "Overrun Probability", "Prediction", "Risk Level",
        ]
        styled_cost = (
            df_cost[display_cols]
            .style
            .applymap(_risk_text_style, subset=["Risk Level"])
            .applymap(_prob_bg_style,   subset=["Overrun Probability"])
            .format({
                "Contract Value":      "${:,.0f}",
                "Overrun Probability": "{:.1%}",
            })
        )
        st.dataframe(styled_cost, use_container_width=True, hide_index=True)

        # High-risk recommendations
        high_risk_rows = df_cost[df_cost["Risk Level"] == "High"]
        if not high_risk_rows.empty:
            with st.expander(
                f"Recommendations for {len(high_risk_rows)} high-risk project(s)"
            ):
                for _, row in high_risk_rows.iterrows():
                    st.markdown(
                        f"**{row['Project Name']}** "
                        f"— `{row['Overrun Probability']:.1%}` overrun probability"
                    )
                    st.info(row["Recommendation"])
                    st.divider()

        st.divider()

        # ── Feature importance + risk distribution ─────────────────────────
        col_fi1, col_dist1 = st.columns(2)

        with col_fi1:
            st.subheader("Feature Importance")
            fig_fi1 = _fi_chart(
                artifact_cost["feature_importance"],
                name_map={
                    "project_type_enc": "Project Type",
                    "contract_value":   "Contract Value",
                    "planned_duration": "Planned Duration",
                    "square_footage":   "Square Footage",
                    "early_phase_cpi":  "Early Phase CPI",
                },
            )
            st.plotly_chart(fig_fi1, use_container_width=True)

        with col_dist1:
            st.subheader("Risk Distribution")
            vc1 = df_cost["Risk Level"].value_counts()
            risk_counts = pd.DataFrame({"Risk": vc1.index, "Count": vc1.values})
            fig_dist1 = px.pie(
                risk_counts, values="Count", names="Risk",
                color="Risk",
                color_discrete_map=_RISK_COLORS,
                template=THEME, hole=0.45,
            )
            fig_dist1.update_traces(textinfo="percent+label", textfont_size=13)
            fig_dist1.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_dist1, use_container_width=True)

        _metrics_expander(
            "Cost Overrun Classifier",
            artifact_cost,
            "RandomForestClassifier · 200 trees · max_depth=8 · "
            "class_weight=balanced · trained on completed projects only",
        )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Equipment Maintenance Predictions
    # ════════════════════════════════════════════════════════════════════════
    st.header("Equipment Maintenance Risk — Active Fleet")
    st.caption(
        "Predicts likelihood of a maintenance event within the next 30 days. "
        "Uses age, usage intensity, fuel cost trends, and maintenance history."
    )

    with st.spinner("Running maintenance predictions…"):
        df_maint = load_maintenance_predictions(engine)

    if df_maint.empty:
        st.info("No active equipment found for maintenance predictions.")
    else:
        # ── KPI row ───────────────────────────────────────────────────────
        n_crit    = int((df_maint["Risk"] == "Critical").sum())
        n_watch   = int((df_maint["Risk"] == "Watch").sum())
        n_routine = int((df_maint["Risk"] == "Routine").sum())
        avg_mp    = df_maint["Maint. Prob."].mean()

        ek1, ek2, ek3, ek4 = st.columns(4)
        with ek1:
            st.metric("Equipment Analyzed", len(df_maint))
        with ek2:
            st.metric(
                "Critical (Action Needed)", n_crit,
                delta=f"-{n_crit}" if n_crit else "0",
                delta_color="inverse",
            )
        with ek3:
            st.metric("On Watch", n_watch)
        with ek4:
            st.metric("Avg Maintenance Probability", f"{avg_mp:.1%}")

        st.divider()

        # ── Predictions table ─────────────────────────────────────────────
        st.subheader("Fleet Maintenance Risk Assessment")

        display_cols_m = [
            "Equipment Name", "Type", "Age (days)", "Total Hours",
            "Maint. Prob.", "Est. Days", "Risk", "Status",
        ]
        styled_maint = (
            df_maint[display_cols_m]
            .style
            .applymap(_risk_text_style, subset=["Risk"])
            .applymap(_prob_bg_style,   subset=["Maint. Prob."])
            .applymap(_days_style,      subset=["Est. Days"])
            .format({
                "Age (days)":  "{:,}",
                "Total Hours": "{:,.1f}",
                "Maint. Prob.": "{:.1%}",
                "Est. Days":   "{:.0f}",
            })
        )
        st.dataframe(styled_maint, use_container_width=True, hide_index=True, height=420)

        # Critical equipment callout
        critical_equip = df_maint[df_maint["Risk"] == "Critical"]
        if not critical_equip.empty:
            with st.expander(
                f"{len(critical_equip)} piece(s) flagged as Critical — view details"
            ):
                st.dataframe(
                    critical_equip[
                        ["Equipment Name", "Type", "Maint. Prob.", "Est. Days",
                         "Age (days)", "Total Hours", "Status"]
                    ].style
                    .applymap(_prob_bg_style, subset=["Maint. Prob."])
                    .applymap(_days_style,    subset=["Est. Days"])
                    .format({
                        "Maint. Prob.": "{:.1%}",
                        "Est. Days":    "{:.0f}",
                        "Age (days)":   "{:,}",
                        "Total Hours":  "{:,.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # ── Feature importance + risk distribution ─────────────────────────
        col_fi2, col_dist2 = st.columns(2)

        with col_fi2:
            st.subheader("Feature Importance")
            fig_fi2 = _fi_chart(
                artifact_maint["feature_importance"],
                name_map={
                    "equipment_type_enc":       "Equipment Type",
                    "age_days":                 "Age (days)",
                    "total_hours_used":         "Total Hours",
                    "avg_hours_per_day":        "Avg Hrs/Day",
                    "fuel_cost_per_hour":       "Fuel Cost/Hour",
                    "recent_fuel_cost_trend":   "Fuel Cost Trend",
                    "maintenance_event_count":  "Past Maint. Events",
                    "days_since_maintenance":   "Days Since Maint.",
                    "avg_maint_cost_per_event": "Avg Maint. Cost",
                },
            )
            st.plotly_chart(fig_fi2, use_container_width=True)

        with col_dist2:
            st.subheader("Risk Distribution")
            vc2 = df_maint["Risk"].value_counts()
            maint_risk_counts = pd.DataFrame({"Risk": vc2.index, "Count": vc2.values})
            fig_dist2 = px.pie(
                maint_risk_counts, values="Count", names="Risk",
                color="Risk",
                color_discrete_map=_RISK_COLORS,
                template=THEME, hole=0.45,
            )
            fig_dist2.update_traces(textinfo="percent+label", textfont_size=13)
            fig_dist2.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_dist2, use_container_width=True)

        _metrics_expander(
            "Equipment Maintenance Classifier",
            artifact_maint,
            "RandomForestClassifier · 200 trees · max_depth=10 · "
            "class_weight=balanced · monthly-snapshot training dataset",
        )


# ---------------------------------------------------------------------------
# Sidebar + Router
# ---------------------------------------------------------------------------


def main():
    engine = get_engine_cached()

    with st.sidebar:
        st.markdown("## 🏗️ Construction Analytics")
        st.divider()

        page = st.radio(
            "Navigate",
            [
                "Executive Dashboard",
                "Project Deep Dive",
                "Equipment Analytics",
                "Subcontractor Performance",
                "Labor & Safety",
                "ML Predictions",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**📅 Date Range**")
        date_start = st.date_input(
            "From", value=DEF_START,
            min_value=date(2022, 1, 1), max_value=TODAY,
            key="global_start",
        )
        date_end = st.date_input(
            "To", value=DEF_END,
            min_value=date(2022, 1, 1), max_value=TODAY,
            key="global_end",
        )
        st.caption("Applies to Equipment & Safety analyses")

        st.divider()
        st.markdown("**⚙️ Actions**")
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")

        st.divider()
        st.caption("SQLite · SQLAlchemy · Plotly · Streamlit")

    # ── Route to selected page ────────────────────────────────────────────
    if date_end < date_start:
        st.error("End date must be after start date.")
        return

    if "Executive"    in page:
        page_executive(engine, date_start, date_end)
    elif "Project"    in page:
        page_project_deep_dive(engine)
    elif "Equipment"  in page:
        page_equipment(engine, date_start, date_end)
    elif "Subcontract" in page:
        page_subcontractor(engine)
    elif "Labor"      in page:
        page_labor_safety(engine, date_start, date_end)
    elif "ML"         in page:
        page_ml_predictions(engine)


if __name__ == "__main__":
    main()
