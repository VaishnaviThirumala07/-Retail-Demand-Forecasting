"""
DS31 Streamlit Dashboard
Four tabs: Forecast Explorer | Accuracy Report | SQL Insights | Cost Simulation
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
REPORTS = ROOT / "reports"
FORECASTS_DIR = ROOT / "data" / "forecasts"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DS31 — Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background: #0f1117; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1d27;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    if WAREHOUSE.exists():
        return duckdb.connect(str(WAREHOUSE), read_only=True)
    return None


@st.cache_data(ttl=3600)
def load_reconciled() -> pd.DataFrame | None:
    path = FORECASTS_DIR / "reconciled_forecasts.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_results() -> pd.DataFrame | None:
    path = REPORTS / "results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_sim() -> pd.DataFrame | None:
    path = REPORTS / "inventory_simulation.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=3600)
def load_actuals_agg(level_col: str | None, level_val: str, con) -> pd.DataFrame | None:
    if con is None:
        return None
    try:
        if level_col is None:
            q = """
            SELECT sale_date AS ds, SUM(units) AS y
            FROM raw_sales_enriched
            WHERE state_id = 'CA'
            GROUP BY ds ORDER BY ds
            """
        else:
            q = f"""
            SELECT sale_date AS ds, SUM(units) AS y
            FROM raw_sales_enriched
            WHERE {level_col} = '{level_val}'
            GROUP BY ds ORDER BY ds
            """
        df = con.execute(q).df()
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    except Exception:
        return None


def run_sql_analysis(con, sql_file: str) -> pd.DataFrame | None:
    path = ROOT / "sql" / "analysis" / sql_file
    if not path.exists() or con is None:
        return None
    try:
        sql = path.read_text()
        # Replace dbt refs with actual table names
        sql = sql.replace("{{ ref('stg_sales') }}", "raw_sales_enriched")
        sql = sql.replace("fct_daily_sales", "raw_sales_enriched")
        return con.execute(sql).df()
    except Exception as e:
        return None


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem">
        <h1 style="background:linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2.5rem;font-weight:700;margin:0">
            📦 DS31 Retail Demand Forecasting
        </h1>
        <p style="color:#9ca3af;font-size:1.1rem;margin-top:0.5rem">
            Hierarchically coherent forecasts across 12 levels · M5 Walmart Dataset
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI row ──────────────────────────────────────────────────────────────────
con = get_connection()
results_df = load_results()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("📊 Hierarchy Levels", "12", help="Total → State → Store → Category → Dept → Item")
with k2:
    st.metric("📈 Bottom Series", "~12K", help="Item × Store series in CA scope")
with k3:
    best_rmsse = results_df["mean_rmsse"].min() if results_df is not None else None
    st.metric("🎯 Best RMSSE", f"{best_rmsse:.4f}" if best_rmsse else "—")
with k4:
    st.metric("🗓️ Forecast Horizon", "28 days")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Forecast Explorer",
    "📊 Accuracy Report",
    "💡 SQL Insights",
    "💰 Cost Simulation",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: Forecast Explorer
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Hierarchy Drill-Down: Actual vs Forecast")

    LEVEL_MAP = {
        "Total (CA)":   (None,       "CA"),
        "State":        ("state_id",  "CA"),
        "Store":        ("store_id",  "CA_1"),
        "Category":     ("cat_id",    "FOODS"),
        "Department":   ("dept_id",   "FOODS_1"),
    }

    col_sel, col_id = st.columns([1, 2])
    with col_sel:
        selected_level = st.selectbox("Hierarchy Level", list(LEVEL_MAP.keys()), key="level_sel")
    level_col, default_id = LEVEL_MAP[selected_level]

    with col_id:
        level_id = st.text_input("Series ID", value=default_id, key="id_inp")

    recon = load_reconciled()
    actuals = load_actuals_agg(level_col, level_id, con)

    if actuals is None or len(actuals) == 0:
        st.info("Load the M5 data and run `make all` to populate forecasts.")
    else:
        method_cols = []
        if recon is not None:
            method_cols = [c for c in recon.columns if c not in ("unique_id", "ds", "y")]

        # Plot actuals
        fig = go.Figure()
        # Training period (last 180 days for readability)
        train_actuals = actuals[actuals["ds"] <= actuals["ds"].max() - pd.Timedelta(days=28)]
        test_actuals = actuals[actuals["ds"] > actuals["ds"].max() - pd.Timedelta(days=28)]

        fig.add_trace(go.Scatter(
            x=train_actuals["ds"].tail(180),
            y=train_actuals["y"].tail(180),
            name="Actual (train)",
            line=dict(color="#94a3b8", width=1.5),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=test_actuals["ds"],
            y=test_actuals["y"],
            name="Actual (test)",
            line=dict(color="#f1f5f9", width=2),
            mode="lines+markers",
        ))

        # Add forecast traces from reconciled if available
        colors = ["#6366f1", "#22d3ee", "#f59e0b", "#10b981"]
        if recon is not None and method_cols:
            for i, method in enumerate(method_cols[:4]):
                if level_col and level_col in recon.columns:
                    m_df = recon[recon[level_col] == level_id] if level_col in recon.columns else recon
                elif "unique_id" in recon.columns:
                    m_df = recon[recon["unique_id"] == level_id]
                else:
                    m_df = recon

                if len(m_df) > 0:
                    m_df = m_df.groupby("ds")[method].sum().reset_index()
                    m_df["ds"] = pd.to_datetime(m_df["ds"])
                    fig.add_trace(go.Scatter(
                        x=m_df["ds"],
                        y=m_df[method].clip(lower=0),
                        name=method,
                        line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                        mode="lines",
                    ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.8)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
            xaxis=dict(gridcolor="#2d3155"),
            yaxis=dict(gridcolor="#2d3155", title="Units Sold"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 Show data"):
            st.dataframe(test_actuals.tail(28).style.format({"y": "{:.0f}"}))


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: Accuracy Report
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Per-Level RMSSE: Baseline vs LightGBM vs Reconciled")

    if results_df is None:
        st.info("Run `make evaluate` to generate accuracy results.")
    else:
        # Pivot for display
        pivot = results_df.pivot_table(
            index="method", columns="level", values="mean_rmsse", aggfunc="mean"
        ).round(4)

        # Color heatmap
        level_order = ["total", "state", "store", "category", "dept"]
        display_cols = [c for c in level_order if c in pivot.columns]
        pivot = pivot[display_cols]

        fig_heat = px.imshow(
            pivot,
            text_auto=".4f",
            color_continuous_scale="RdYlGn_r",
            title="RMSSE Heatmap (lower = better)",
            aspect="auto",
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Bar chart comparison
        if "level" in results_df.columns and "mean_rmsse" in results_df.columns:
            fig_bar = px.bar(
                results_df[results_df["level"].isin(level_order)],
                x="level",
                y="mean_rmsse",
                color="method",
                barmode="group",
                title="Mean RMSSE per Level by Method",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,29,39,0.8)",
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            results_df.sort_values(["level", "mean_rmsse"])
                      .style.format({"mean_rmsse": "{:.4f}", "mean_mae": "{:.2f}"}),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: SQL Insights
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("SQL Analytics — Key Insights from 59M+ Rows")

    insight_options = {
        "SNAP Day Lift":        "02_snap_day_lift.sql",
        "Weekday Seasonality":  "01_weekday_monthly_seasonality.sql",
        "Event Impact":         "03_event_holiday_impact.sql",
        "Price Elasticity":     "04_price_elasticity.sql",
        "Intermittency":        "05_intermittency.sql",
        "Top/Bottom Items":     "06_top_bottom_items_by_store.sql",
        "Pareto Analysis":      "08_pareto_analysis.sql",
    }

    selected_insight = st.selectbox("Choose Analysis", list(insight_options.keys()))
    sql_file = insight_options[selected_insight]

    # Show SQL
    sql_path = ROOT / "sql" / "analysis" / sql_file
    if sql_path.exists():
        with st.expander("📄 View SQL"):
            st.code(sql_path.read_text(), language="sql")

    # Run and display
    if con is not None:
        df_insight = run_sql_analysis(con, sql_file)
        if df_insight is not None and len(df_insight) > 0:
            # Auto-chart based on insight type
            if selected_insight == "SNAP Day Lift":
                fig = px.bar(
                    df_insight,
                    x="cat_id" if "cat_id" in df_insight.columns else df_insight.columns[0],
                    y="snap_lift_pct" if "snap_lift_pct" in df_insight.columns else df_insight.columns[-1],
                    color="state_id" if "state_id" in df_insight.columns else None,
                    title="SNAP Day Sales Lift % by Category and State",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif selected_insight == "Pareto Analysis":
                if "revenue_share_pct" in df_insight.columns:
                    fig = px.bar(
                        df_insight,
                        x="item_bucket",
                        y="revenue_share_pct",
                        title="Pareto: Revenue Share by Item Bucket",
                        color="revenue_share_pct",
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)

            elif selected_insight == "Intermittency":
                if "intermittent_share_pct" in df_insight.columns:
                    fig = px.bar(
                        df_insight,
                        x="cat_id" if "cat_id" in df_insight.columns else df_insight.columns[0],
                        y="intermittent_share_pct",
                        title="Intermittent Items (>50% zero sales) by Category",
                        color="avg_zero_pct",
                        color_continuous_scale="Reds",
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_insight, use_container_width=True)
        else:
            st.info("Run `make ingest` and `make dbt-run` to load data for SQL analysis.")
    else:
        st.warning("No DuckDB warehouse found. Run `make ingest` first.")

    # Static key insights (always shown)
    st.divider()
    st.markdown("### 📝 Key Findings (fill in with real numbers after running)")
    insights_path = REPORTS / "insights.md"
    if insights_path.exists():
        st.markdown(insights_path.read_text())
    else:
        st.markdown("""
        | # | Insight | Impact |
        |---|---------|--------|
        | 1 | SNAP days drive **X% higher** food sales in CA vs non-SNAP days | Demand planning |
        | 2 | Friday/Saturday sales are **X% above** weekday average | Staffing & replenishment |
        | 3 | Top 20% of items produce **~80% of revenue** (Pareto) | Assortment focus |
        | 4 | **X%** of items have >50% zero-sales days (intermittent demand) | Model selection |
        | 5 | Christmas week drives **X% uplift**, Super Bowl **X%** | Event forecasting |
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: Cost Simulation
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Inventory Cost Simulation")

    sim_df = load_sim()

    if sim_df is None or len(sim_df) == 0:
        st.info("Run `make evaluate` to generate cost simulation results.")
    else:
        # KPI cards
        cols = st.columns(len(sim_df))
        baseline_cost = sim_df.iloc[0]["total_cost"] if len(sim_df) > 0 else 1

        for i, (_, row) in enumerate(sim_df.iterrows()):
            with cols[i]:
                pct_delta = ((row["total_cost"] - baseline_cost) / baseline_cost * 100
                             if i > 0 else 0)
                st.metric(
                    label=row["method"],
                    value=f"${row['total_cost']:,.0f}",
                    delta=f"{pct_delta:+.1f}% vs baseline" if i > 0 else "Baseline",
                    delta_color="inverse",
                )

        # Cost breakdown chart
        fig_cost = go.Figure()
        for _, row in sim_df.iterrows():
            fig_cost.add_trace(go.Bar(
                name=row["method"],
                x=["Stockout Cost", "Holding Cost"],
                y=[row["stockout_cost"], row["holding_cost"]],
            ))
        fig_cost.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            title="Cost Breakdown by Forecast Method",
            height=400,
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Fill rate
        fig_fill = px.bar(
            sim_df,
            x="method",
            y="fill_rate",
            title="Fill Rate (Service Level) by Method",
            color="fill_rate",
            color_continuous_scale="Greens",
            text_auto=".1%",
        )
        fig_fill.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_fill, use_container_width=True)

        st.dataframe(sim_df.style.format({
            "stockout_cost": "${:,.2f}",
            "holding_cost": "${:,.2f}",
            "total_cost": "${:,.2f}",
            "fill_rate": "{:.1%}",
        }), use_container_width=True)

    # Cost assumption slider
    st.divider()
    st.markdown("### ⚙️ Sensitivity Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        stockout_mult = st.slider("Stockout cost multiplier", 0.5, 3.0, 1.0, 0.1)
        st.caption("Multiply the assumed $1.05/unit stockout cost")
    with col_b:
        z_score = st.slider("Safety stock z-score", 1.0, 2.5, 1.65, 0.05)
        st.caption(f"Service level: {100*(1 - (1 - 0.95)**(1/z_score)):.0f}% approx")
    st.info(
        "💡 Cost assumptions: avg unit price $3.50, stockout = 30% margin, "
        "holding = 0.5%/unit/day. Stated openly — not production costs."
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#4b5563;font-size:0.8rem;padding:1rem">
        DS31 · M5 Forecasting · LightGBM + MinTrace · DuckDB + dbt
        · <a href="http://localhost:8000/docs" style="color:#6366f1">API Docs</a>
    </div>
    """,
    unsafe_allow_html=True,
)
