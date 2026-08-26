"""
app.py
======
E-Commerce Delivery Risk Intelligence -- Streamlit dashboard.

This file is presentation-only. Every analytical computation (cleaning,
feature engineering, modelling, evaluation, risk scoring, managerial
insights) is performed in core.py and consumed here via run_pipeline().
Run with:  streamlit run app.py

UI/UX design system: a small set of render_* helper functions (defined
below, section "DESIGN SYSTEM") are reused across every tab instead of
scattering one-off HTML/CSS. No analytical calculation, filter, model,
or downstream number in this file was changed from the original app --
only how results are laid out and styled.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# COLOR TOKENS (kept from the existing navy / red / green / orange language)
# --------------------------------------------------------------------------
PRIMARY = "#1F3B57"      # navy
PRIMARY_DARK = "#14283C"
ACCENT = "#C0392B"       # red   -> high risk / late
GOOD = "#1E8449"         # green -> low risk / on-time
WARN = "#B9770E"         # orange -> medium risk
NEUTRAL = "#5D6D7E"      # muted slate for secondary text
BORDER = "#E2E6EB"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F5F7FA"
SIDEBAR_BG = "#EEF2F6"

RISK_COLORS = {"High Risk": ACCENT, "Medium Risk": WARN, "Low Risk": GOOD}

# --------------------------------------------------------------------------
# GLOBAL CSS -- single design system used by every render_* helper below
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAGE_BG}; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1300px; }}
    h1, h2, h3, h4 {{ color: {PRIMARY}; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* ---------- App header ---------- */
    .app-header {{
        background-color: {PRIMARY};
        border-radius: 12px;
        padding: 1.7rem 2.1rem;
        margin-bottom: 1.4rem;
    }}
    .app-header h1 {{
        color: #FFFFFF; font-size: 1.7rem; font-weight: 700; margin: 0;
        letter-spacing: 0.01em;
    }}
    .app-header p.subtitle {{
        color: #D7E1EA; font-size: 0.98rem; margin: 0.35rem 0 0 0;
    }}
    .app-header p.tag {{
        color: #93A9BF; font-size: 0.74rem; margin: 0.75rem 0 0 0;
        text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
    .sidebar-title {{
        color: {PRIMARY}; font-size: 1.02rem; font-weight: 700; margin-bottom: 0.1rem;
    }}
    .sidebar-caption {{ color: {NEUTRAL}; font-size: 0.78rem; margin-bottom: 0.6rem; }}
    .sidebar-section-label {{
        color: {PRIMARY}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; margin: 1.1rem 0 0.35rem 0; padding-bottom: 0.3rem;
        border-bottom: 1px solid {BORDER};
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background-color: #E7ECF1; padding: 6px; border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px; padding: 0 16px; border-radius: 7px; background-color: transparent;
        color: {NEUTRAL}; font-weight: 600; font-size: 0.85rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY} !important; color: #FFFFFF !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {CARD_BG}; border: 1px solid {BORDER}; border-top: none;
        border-radius: 0 0 10px 10px; padding: 1.4rem 1.6rem 1.1rem 1.6rem;
    }}

    /* ---------- Section headers ---------- */
    .section-header {{ border-left: 4px solid {PRIMARY}; padding-left: 0.7rem; margin: 1.5rem 0 0.8rem 0; }}
    .section-header h4 {{ margin: 0; font-size: 1.02rem; }}
    .section-header p {{ margin: 0.1rem 0 0 0; color: {NEUTRAL}; font-size: 0.83rem; }}

    /* ---------- KPI cards ---------- */
    .kpi-row {{ display: flex; gap: 0.85rem; flex-wrap: wrap; margin-bottom: 1.1rem; }}
    .kpi-card {{
        flex: 1 1 200px; background-color: {CARD_BG}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 0.95rem 1.15rem; box-shadow: 0 1px 3px rgba(16,24,40,0.05);
    }}
    .kpi-label {{
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        color: {NEUTRAL};
    }}
    .kpi-value {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.15rem; line-height: 1.15; }}
    .kpi-sub {{ font-size: 0.76rem; color: {NEUTRAL}; margin-top: 0.25rem; }}

    /* ---------- Insight / info cards ---------- */
    .insight-card {{
        background-color: #F4F6F8; border-left: 4px solid {PRIMARY}; padding: 0.6rem 0.95rem;
        border-radius: 6px; margin-bottom: 0.5rem; font-size: 0.9rem; color: #2C3E50;
    }}

    /* ---------- Risk / stage cards ---------- */
    .risk-card {{
        background-color: {CARD_BG}; border: 1px solid {BORDER}; border-top: 4px solid {PRIMARY};
        border-radius: 10px; padding: 0.95rem 1.1rem; box-shadow: 0 1px 3px rgba(16,24,40,0.05);
        height: 100%;
    }}
    .risk-card-title {{
        font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em;
        margin-bottom: 0.5rem;
    }}
    .risk-card-list {{ margin: 0; padding-left: 1.05rem; font-size: 0.85rem; color: #34495E; }}
    .risk-card-list li {{ margin-bottom: 0.2rem; }}

    /* ---------- Flow / pipeline diagrams ---------- */
    .flow-vertical {{ display: flex; flex-direction: column; align-items: center; }}
    .flow-horizontal {{ display: flex; flex-direction: row; align-items: center; flex-wrap: wrap; justify-content: center; gap: 0.25rem; }}
    .flow-node {{
        background-color: {CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {PRIMARY};
        border-radius: 8px; padding: 0.5rem 0.95rem; font-size: 0.82rem; font-weight: 600;
        color: {PRIMARY}; text-align: center; min-width: 140px;
    }}
    .flow-node.small {{ min-width: 110px; font-size: 0.74rem; padding: 0.4rem 0.65rem; }}
    .flow-node.alt {{ border-left-color: {ACCENT}; }}
    .flow-arrow {{ color: {NEUTRAL}; font-size: 1.05rem; margin: 0.05rem 0.3rem; }}
    .rel-branches {{ display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; margin: 0.35rem 0; }}
    .rel-arrow {{ color: {NEUTRAL}; font-size: 0.78rem; text-align: center; margin: 0.1rem 0; }}

    /* ---------- Mini info cards (methodology strip) ---------- */
    .mini-card {{
        background-color: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 0.75rem 0.85rem; height: 100%;
    }}
    .mini-card-title {{ font-size: 0.72rem; font-weight: 700; color: {PRIMARY}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem; }}
    .mini-card-body {{ font-size: 0.8rem; color: #34495E; }}
    .mini-card-body ul {{ margin: 0; padding-left: 1.05rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# DESIGN SYSTEM -- reusable render_* helpers (used across every tab)
# ==========================================================================
def render_header(title: str, subtitle: str, tag: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
            <p class="tag">{tag}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""<div class="section-header"><h4>{title}</h4>{sub_html}</div>""",
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, sub: str = "", color: str = PRIMARY) -> str:
    """Returns the HTML for a single KPI card (composed into rows by render_kpi_row)."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color};">{value}</div>{sub_html}</div>'
    )


def render_kpi_row(cards: list) -> None:
    """cards: list of dicts with keys label, value, sub (optional), color (optional)."""
    html = '<div class="kpi-row">' + "".join(
        render_kpi_card(c["label"], c["value"], c.get("sub", ""), c.get("color", PRIMARY)) for c in cards
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_info_card(text: str) -> None:
    st.markdown(f'<div class="insight-card">• {text}</div>', unsafe_allow_html=True)


def render_risk_card(title: str, color: str, lines: list) -> None:
    items = "".join(f"<li>{l}</li>" for l in lines) if lines else "<li>No data available.</li>"
    st.markdown(
        f"""
        <div class="risk-card" style="border-top-color:{color};">
            <div class="risk-card-title" style="color:{color};">{title}</div>
            <ul class="risk-card-list">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_node(text: str, size: str = "normal", variant: str = "") -> str:
    cls = "flow-node" + (" small" if size == "small" else "") + (f" {variant}" if variant else "")
    return f'<div class="{cls}">{text}</div>'


def render_flowchart(nodes: list, orientation: str = "vertical") -> None:
    arrow = "↓" if orientation == "vertical" else "→"
    wrap_class = "flow-vertical" if orientation == "vertical" else "flow-horizontal"
    parts = []
    for i, n in enumerate(nodes):
        parts.append(render_flow_node(n))
        if i < len(nodes) - 1:
            parts.append(f'<div class="flow-arrow">{arrow}</div>')
    st.markdown(f'<div class="{wrap_class}">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def render_mini_card(title: str, lines: list) -> None:
    items = "".join(f"<li>{l}</li>" for l in lines)
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-card-title">{title}</div>
            <div class="mini-card-body"><ul>{items}</ul></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_relationship_diagram() -> None:
    html = f"""
    <div class="flow-vertical">
        {render_flow_node('CUSTOMERS')}
        <div class="rel-arrow">↓ customer_id</div>
        {render_flow_node('ORDERS')}
        <div class="rel-arrow">↓ order_id</div>
        <div class="rel-branches">
            {render_flow_node('PAYMENTS', 'small')}
            {render_flow_node('REVIEWS', 'small')}
            {render_flow_node('ORDER ITEMS', 'small')}
        </div>
        <div class="rel-arrow">↓ product_id / seller_id (from ORDER ITEMS)</div>
        <div class="rel-branches">
            {render_flow_node('PRODUCTS', 'small')}
            {render_flow_node('SELLERS', 'small')}
        </div>
        <div class="rel-branches" style="margin-top:0.5rem;">
            {render_flow_node('PRODUCTS → CATEGORY TRANSLATION', 'small', 'alt')}
            {render_flow_node('CUSTOMERS / SELLERS → GEOLOCATION', 'small', 'alt')}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
render_header(
    "E-Commerce Delivery Risk Intelligence",
    "Predicting and preventing late deliveries using machine learning",
    "DAML · Working-with-AI · IIM Ranchi",
)

# --------------------------------------------------------------------------
# DATA AVAILABILITY CHECK (graceful failure, no raw tracebacks)
# --------------------------------------------------------------------------
available, info = core.check_data_availability()
if not available:
    st.error(
        "**Data files not found.** This app expects all nine Olist CSV files to sit in the "
        "same folder as `app.py` / `core.py` (no subfolder). Details below:"
    )
    st.code(info)
    st.stop()

data_dir = info

with st.spinner("Running the full analytical pipeline (data integration → cleaning → EDA → modelling)... "
                 "this runs once and is cached for the session."):
    try:
        results = core.run_pipeline(data_dir)
    except Exception as exc:  # graceful failure per spec section 25
        st.error("The analytical pipeline could not complete. This usually means a required "
                 "column is missing or a CSV is corrupted.")
        st.exception(exc)
        st.stop()

df = results.df_model

# --------------------------------------------------------------------------
# SIDEBAR — FILTERS (apply across relevant tabs; the underlying model/results
# are NOT retrained on filter change -- filters only slice display data)
# --------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">Dashboard Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-caption">Filters slice the descriptive views below. Model training itself is fixed and cached.</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-section-label">Geography</div>', unsafe_allow_html=True)
all_states = sorted(df["customer_state"].dropna().unique().tolist())
sel_cust_states = st.sidebar.multiselect("Customer state", all_states, default=[])

all_seller_states = sorted(df["seller_state"].dropna().unique().tolist())
sel_seller_states = st.sidebar.multiselect("Seller state", all_seller_states, default=[])

st.sidebar.markdown('<div class="sidebar-section-label">Product</div>', unsafe_allow_html=True)
all_categories = sorted(df["dominant_category"].dropna().unique().tolist())
sel_categories = st.sidebar.multiselect("Product category", all_categories, default=[])

st.sidebar.markdown('<div class="sidebar-section-label">Time Window</div>', unsafe_allow_html=True)
min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Purchase date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

st.sidebar.markdown('<div class="sidebar-section-label">Risk Segment</div>', unsafe_allow_html=True)
risk_levels = ["High Risk", "Medium Risk", "Low Risk"]
sel_risk = st.sidebar.multiselect("Risk segment (scored test orders)", risk_levels, default=[])


def apply_filters(frame: pd.DataFrame, has_risk_col: bool = False) -> pd.DataFrame:
    out = frame.copy()
    if sel_cust_states and "customer_state" in out.columns:
        out = out[out["customer_state"].isin(sel_cust_states)]
    if sel_seller_states and "seller_state" in out.columns:
        out = out[out["seller_state"].isin(sel_seller_states)]
    if sel_categories and "dominant_category" in out.columns:
        out = out[out["dominant_category"].isin(sel_categories)]
    if "order_purchase_timestamp" in out.columns and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        out = out[
            (out["order_purchase_timestamp"].dt.date >= start) & (out["order_purchase_timestamp"].dt.date <= end)
        ]
    if has_risk_col and sel_risk and "risk_segment" in out.columns:
        out = out[out["risk_segment"].isin(sel_risk)]
    return out


df_f = apply_filters(df)
pred_table_f = apply_filters(results.order_prediction_table, has_risk_col=True)

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Executive Control Tower",
    "Delivery Risk Intelligence",
    "Risk Drivers",
    "Order & Risk Drilldown",
    "Model Comparison",
    "Managerial Action Center",
    "Data & Methodology",
])

# ============================== TAB 1 =====================================
with tab1:
    render_section_header("Executive KPI Snapshot", "Headline volume, risk and model-quality metrics for the current filter selection.")

    avg_delay_days = None
    if len(df_f):
        delivered_delay = (df_f["order_delivered_customer_date"] - df_f["order_estimated_delivery_date"]).dt.days
        avg_delay_days = delivered_delay[delivered_delay > 0].mean()
    n_high_risk = (pred_table_f["risk_segment"] == "High Risk").sum() if len(pred_table_f) else 0

    render_kpi_row([
        {"label": "Total Orders (modelled)", "value": f"{len(df_f):,}"},
        {"label": "Late-Delivery Rate", "value": f"{df_f['late_delivery'].mean():.1%}" if len(df_f) else "N/A", "color": ACCENT},
        {"label": "High-Risk Orders (scored)", "value": f"{n_high_risk:,}", "color": ACCENT},
        {"label": "Best Model", "value": results.best_model_name},
    ])
    render_kpi_row([
        {"label": "Avg. Delay (late orders)", "value": f"{avg_delay_days:.1f} days" if avg_delay_days == avg_delay_days else "N/A"},
        {"label": "Recall (late class)", "value": f"{results.comparison_df.loc[results.best_model_name, 'Recall']:.1%}", "color": GOOD},
        {"label": "F1-score", "value": f"{results.comparison_df.loc[results.best_model_name, 'F1-score']:.3f}"},
        {"label": "ROC-AUC", "value": f"{results.comparison_df.loc[results.best_model_name, 'ROC-AUC']:.3f}"},
    ])

    render_section_header("Delivery Risk Overview")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        month_df = results.eda["late_rate_by_month"]
        fig = px.line(month_df, x="purchase_month_period", y="late_rate", markers=True,
                      labels={"purchase_month_period": "Purchase Month", "late_rate": "Late-Delivery Rate"},
                      title="Late-Delivery Rate Over Time")
        fig.update_traces(line_color=ACCENT)
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if len(pred_table_f):
            seg_counts = pred_table_f["risk_segment"].value_counts().reindex(risk_levels).fillna(0)
            fig2 = px.pie(values=seg_counts.values, names=seg_counts.index, hole=0.55,
                          color=seg_counts.index, color_discrete_map=RISK_COLORS,
                          title="Risk Distribution (scored orders)")
            fig2.update_layout(margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No scored orders match the current filters.")

    render_section_header("Risk by Region", "Highest late-delivery rate customer states (min. 5 orders).")
    if len(df_f):
        top_state_df = (
            df_f.groupby("customer_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        )
        top_state_df = top_state_df[top_state_df["count"] >= 5].sort_values("mean", ascending=False).head(8)
        fig = px.bar(top_state_df, x="customer_state", y="mean", color="mean",
                     color_continuous_scale=[GOOD, WARN, ACCENT],
                     labels={"mean": "Late-Delivery Rate", "customer_state": "Customer State"},
                     title="Top 8 Highest-Risk Customer States")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No orders match the current filters.")

    render_section_header("Key Drivers", "Strongest predictors of late delivery (permutation importance).")
    top_imp = results.feature_importance.head(5).sort_values("importance_mean")
    fig = px.bar(top_imp, x="importance_mean", y="feature", orientation="h",
                 labels={"importance_mean": "Mean Importance (Recall drop)", "feature": ""},
                 title="Top 5 Predictive Drivers")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    render_section_header("Management Priority", "What requires attention this week, based on current scored risk segments.")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_risk_card("High Risk", ACCENT, results.managerial_insights[:1] or ["Escalate and monitor daily."])
    with p2:
        render_risk_card("Medium Risk", WARN, ["Route to daily operational review queue."])
    with p3:
        render_risk_card("Low Risk", GOOD, ["Continue standard processing; no action required."])

# ============================== TAB 2 ======================================
with tab2:
    render_section_header("Delivery Risk Intelligence", "Descriptive risk patterns across geography, category and time (filters applied).")

    col1, col2 = st.columns(2)
    with col1:
        cust_state_df = df_f.groupby("customer_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        cust_state_df = cust_state_df[cust_state_df["count"] >= 5].sort_values("mean", ascending=False)
        fig = px.bar(cust_state_df, x="customer_state", y="mean", color="mean",
                     color_continuous_scale=[GOOD, WARN, ACCENT],
                     labels={"mean": "Late-Delivery Rate", "customer_state": "Customer State"},
                     title="Late-Delivery Rate by Customer State")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        seller_state_df = df_f.groupby("seller_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        seller_state_df = seller_state_df[seller_state_df["count"] >= 5].sort_values("mean", ascending=False)
        fig = px.bar(seller_state_df, x="seller_state", y="mean", color="mean",
                     color_continuous_scale=[GOOD, WARN, ACCENT],
                     labels={"mean": "Late-Delivery Rate", "seller_state": "Seller State"},
                     title="Late-Delivery Rate by Seller State")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat_df = df_f.groupby("dominant_category")["late_delivery"].agg(["mean", "count"]).reset_index()
        cat_df = cat_df[cat_df["count"] >= 15].sort_values("mean", ascending=False).head(15)
        fig = px.bar(cat_df, x="mean", y="dominant_category", orientation="h",
                     labels={"mean": "Late-Delivery Rate", "dominant_category": "Product Category"},
                     title="Top 15 Product Categories by Late-Delivery Rate", color="mean",
                     color_continuous_scale=[GOOD, WARN, ACCENT])
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(df_f.assign(**{"Delivery Outcome": df_f["late_delivery"].map({0: "On-Time", 1: "Late"})}),
                            x="distance_km", color="Delivery Outcome", barmode="overlay", nbins=40,
                            color_discrete_map={"On-Time": GOOD, "Late": ACCENT},
                            title="Seller–Customer Distance Distribution by Outcome")
        st.plotly_chart(fig, use_container_width=True)

    render_section_header("Delivery Delay Distribution", "Late orders only.")
    late_only = df_f[df_f["late_delivery"] == 1].copy()
    if len(late_only):
        late_only["delay_days"] = (late_only["order_delivered_customer_date"] - late_only["order_estimated_delivery_date"]).dt.days
        fig = px.box(late_only, y="delay_days", points="outliers", title="Distribution of Delay (days beyond promise)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No late orders in the current filter selection.")

# ============================== TAB 3 ======================================
with tab3:
    render_section_header("Risk Drivers", f"Feature relevance computed via permutation importance on the selected model ({results.best_model_name}).")

    imp = results.feature_importance.head(15)
    fig = px.bar(imp.sort_values("importance_mean"), x="importance_mean", y="feature", orientation="h",
                 error_x="importance_std", title="Top Predictive Drivers (Permutation Importance, Recall-based)",
                 labels={"importance_mean": "Mean Importance (Recall drop)", "feature": "Feature"})
    fig.update_traces(marker_color=PRIMARY)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Numeric Feature Relevance (Feature Selection Stage)")
        num_sel = results.selection_report[results.selection_report["type"] == "numeric"]
        st.dataframe(num_sel, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("##### Categorical Feature Relevance (Feature Selection Stage)")
        cat_sel = results.selection_report[results.selection_report["type"] == "categorical"]
        st.dataframe(cat_sel, use_container_width=True, hide_index=True)

    render_section_header("Dimensionality Reduction Assessment")
    st.write(results.dim_reduction["explanation"])
    if results.dim_reduction["recommended"]:
        st.dataframe(results.dim_reduction["explained_variance_table"], use_container_width=True, hide_index=True)
    if results.dim_reduction["high_corr_pairs"]:
        st.caption(f"High-correlation numeric pairs detected: {results.dim_reduction['high_corr_pairs']}")

    render_section_header("Value vs. Delay Relationship")
    col3, col4 = st.columns(2)
    with col3:
        sample_df = df_f.sample(min(3000, len(df_f)), random_state=42) if len(df_f) else df_f
        fig = px.scatter(sample_df, x="total_price", y="approval_time_hours",
                          color=sample_df["late_delivery"].map({0: "On-Time", 1: "Late"}) if len(sample_df) else None,
                          color_discrete_map={"On-Time": GOOD, "Late": ACCENT}, opacity=0.5,
                          title="Order Value vs. Approval Time (sampled)")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        pt_df = results.eda["processing_time_vs_late"]
        fig = go.Figure()
        fig.add_bar(x=["On-Time", "Late"], y=pt_df["approval_time_hours"], name="Avg Approval Time (hrs)", marker_color=PRIMARY)
        fig.add_bar(x=["On-Time", "Late"], y=pt_df["estimated_window_days"], name="Avg Estimated Window (days)", marker_color=ACCENT)
        fig.update_layout(title="Processing / Promise Timing by Outcome", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# ============================== TAB 4 ======================================
with tab4:
    render_section_header("Order & Risk Drilldown", "Scored orders come from the held-out test set used to evaluate the selected model.")

    search_id = st.text_input("Search by Order ID (exact match)")
    view_df = pred_table_f.copy()
    if search_id:
        view_df = view_df[view_df["order_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

    st.dataframe(
        view_df[[
            "order_id", "customer_state", "seller_state", "dominant_category", "total_price",
            "distance_km", "estimated_window_days", "risk_probability", "risk_segment",
            "actual_late_delivery", "predicted_late_delivery", "recommended_intervention",
        ]],
        use_container_width=True, hide_index=True, height=420,
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇ Download filtered high-risk orders (CSV)",
            data=view_df[view_df["risk_segment"] == "High Risk"].to_csv(index=False).encode("utf-8"),
            file_name="high_risk_orders.csv", mime="text/csv",
        )
    with dl2:
        st.download_button(
            "⬇ Download full filtered drilldown table (CSV)",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="order_risk_drilldown.csv", mime="text/csv",
        )

# ============================== TAB 5 ======================================
with tab5:
    render_section_header(
        "Model Comparison",
        f"Deterministic 75/25 stratified train/test split, random_state=42. KNN k was selected via "
        f"cross-validated recall search (k={results.model_meta['best_k']}).",
    )

    st.dataframe(results.comparison_df.style.format("{:.4f}"), use_container_width=True)
    st.success(results.best_model_justification)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Confusion Matrices")
        for name, res in results.model_results.items():
            cm = res.confusion_mat
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=["On-Time", "Late"], y=["On-Time", "Late"],
                             title=f"{res.name} — Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### ROC Curves")
        fig = go.Figure()
        for name, res in results.model_results.items():
            fig.add_trace(go.Scatter(x=res.fpr, y=res.tpr, mode="lines", name=f"{res.name} (AUC={res.metrics['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Class Balance (Training Data)")
        cb = results.model_meta["class_balance"]
        st.write(f"Late: {cb['late_pct']:.1%} · On-Time: {cb['on_time_pct']:.1%}")
        st.caption(
            "Class imbalance is addressed through probability-based risk segmentation (Tab 4) rather than "
            "artificial resampling, keeping the reported metrics representative of real-world order volumes."
        )

# ============================== TAB 6 ======================================
with tab6:
    render_section_header("Managerial Action Center", "Risk → action framework, ranked drivers and the recommended intervention roadmap.")

    # --- Risk -> Action framework, sourced from the actual scored predictions
    # (results.order_prediction_table["recommended_intervention"], produced by core.py) ---
    render_section_header("Risk → Action Framework")
    full_pred = results.order_prediction_table
    r1, r2, r3 = st.columns(3)
    for col, seg, color in zip((r1, r2, r3), risk_levels, (ACCENT, WARN, GOOD)):
        seg_rows = full_pred[full_pred["risk_segment"] == seg] if "risk_segment" in full_pred.columns else pd.DataFrame()
        if len(seg_rows) and "recommended_intervention" in seg_rows.columns:
            actions = seg_rows["recommended_intervention"].dropna().unique().tolist()[:4]
        else:
            actions = ["No scored orders in this segment."]
        with col:
            render_risk_card(seg, color, actions)

    render_section_header("Top Risk Drivers", "Ranked by permutation importance (see Risk Drivers tab for full detail).")
    top6 = results.feature_importance.head(6).sort_values("importance_mean")
    fig = px.bar(top6, x="importance_mean", y="feature", orientation="h",
                 labels={"importance_mean": "Mean Importance (Recall drop)", "feature": ""},
                 title="Strongest Predictors of Late Delivery")
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    render_section_header("Major Findings")
    for insight in results.managerial_insights:
        render_info_card(insight)

    render_section_header("Intervention Framework", "Full segment → action mapping produced by the scoring pipeline.")
    with st.expander("View full intervention framework table", expanded=False):
        st.dataframe(results.intervention_framework, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download intervention framework (CSV)",
        data=results.intervention_framework.to_csv(index=False).encode("utf-8"),
        file_name="intervention_framework.csv", mime="text/csv",
    )

    render_section_header("Suggested Implementation Roadmap")
    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        render_risk_card("Week 1–2", PRIMARY, ["Deploy scoring model in shadow mode.", "Validate risk segments against actual outcomes."])
    with rm2:
        render_risk_card("Week 3–4", PRIMARY, ["Route High-Risk orders to a daily proactive-monitoring queue."])
    with rm3:
        render_risk_card("Month 2", PRIMARY, ["Share risk drivers with Seller Management & Logistics.", "Target evidence-based conversations, not blanket penalties."])
    with rm4:
        render_risk_card("Month 3+", PRIMARY, ["Re-evaluate performance on fresh data; retrain if Recall degrades.", "Reassess High-Risk threshold vs. intervention capacity."])

    render_section_header("Responsible-AI / Analytical Limitations")
    for lim in core.DATA_REQUIREMENTS["data_limitations"]:
        render_info_card(lim)
    render_info_card("Model outputs are probabilistic risk scores, not certainties, and should support -- not replace -- managerial judgment.")
    render_info_card("The models establish association/prediction, not proven causal mechanisms; interventions should be piloted and measured before wide rollout.")
    render_info_card("Public historical data may not reflect current carrier partnerships, seller mix, or seasonal effects.")

    if results.review_narrative is not None and len(results.review_narrative) == 2:
        render_section_header(
            "Downstream Customer-Experience Signal",
            "Descriptive only, not used as a model input. Review data occurs after delivery.",
        )
        rn = results.review_narrative.copy()
        rn["late_delivery"] = rn["late_delivery"].map({0: "On-Time", 1: "Late"})
        fig = px.bar(rn, x="late_delivery", y="review_score", color="late_delivery",
                     color_discrete_map={"On-Time": GOOD, "Late": ACCENT},
                     title="Average Review Score by Delivery Outcome")
        st.plotly_chart(fig, use_container_width=True)

# ============================== TAB 7 ======================================
with tab7:
    render_section_header("Data Requirement Analysis & Methodology")

    dr = core.DATA_REQUIREMENTS

    render_section_header("Analytical Pipeline")
    render_flowchart(
        [
            "9 Olist CSVs", "Data Integration", "Data Cleaning", "Feature Engineering",
            "Target Creation", "Feature Selection", "Train / Test Split", "KNN + Naive Bayes",
            "Model Evaluation", "Risk Scoring", "Managerial Action",
        ],
        orientation="horizontal",
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_mini_card("Data", [
            "9 source tables",
            "Order-level unit of analysis",
            "Key relational joins",
        ])
    with m2:
        render_mini_card("Target", [f"{dr['target_variable']}"])
    with m3:
        render_mini_card("Features", [p for p in dr["predictor_families"]])
    with m4:
        render_mini_card("Models", ["KNN", "Naive Bayes"])
    with m5:
        render_mini_card("Output", ["Risk probability", "Risk segment", "Recommended intervention"])

    render_section_header("Data Relationship Map", "How the nine Olist tables connect for order-level integration.")
    render_data_relationship_diagram()

    with st.expander("Business & technical objective, assumptions and leakage policy", expanded=False):
        st.markdown(f"**Business objective:** {dr['business_objective']}")
        st.markdown(f"**Technical objective:** {dr['technical_objective']}")
        st.markdown(f"**Unit of analysis:** {dr['unit_of_analysis']}")
        st.markdown(f"**Target variable:** {dr['target_variable']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Predictor families used:**")
            for p in dr["predictor_families"]:
                st.markdown(f"- {p}")
        with col2:
            st.markdown("**Excluded as predictors (data-leakage policy):**")
            for e in dr["excluded_as_predictors_due_to_leakage"]:
                st.markdown(f"- {e}")

        st.markdown("**Keys & relationships used for multi-table integration:**")
        for k in dr["keys_and_relationships"]:
            st.markdown(f"- {k}")

        st.markdown("**Assumptions:**")
        for a in dr["assumptions"]:
            st.markdown(f"- {a}")

    render_section_header("Data Cleaning, Integration & Target Construction Logs")
    log1, log2 = st.columns(2)
    with log1:
        with st.expander("Multi-table integration log", expanded=False):
            st.json(results.integration_log)
        with st.expander("Target construction log", expanded=False):
            st.json(results.target_log)
            st.caption(
                f"{len(results.df_excluded):,} orders were excluded from modelling (not delivered, or missing "
                "actual/estimated dates) and are reported here for transparency, not silently dropped."
            )
    with log2:
        for table, log in results.cleaning_logs.items():
            with st.expander(f"`{table}` cleaning log", expanded=False):
                st.json(log)

    render_section_header("Missing-Value & Outlier Treatment")
    mv1, mv2 = st.columns(2)
    with mv1:
        st.markdown("##### Missing-Value Analysis (pre-treatment)")
        st.dataframe(results.missing_summary, use_container_width=True, hide_index=True)
        with st.expander("Missing-value treatment log", expanded=False):
            st.json(results.missing_treatment_log)
    with mv2:
        st.markdown("##### Outlier Analysis (IQR method)")
        st.dataframe(results.outlier_summary, use_container_width=True, hide_index=True)
        with st.expander("Outlier treatment log (winsorized/capped at 3×IQR, not deleted)", expanded=False):
            st.json(results.outlier_treatment_log)

    render_section_header("Final Modelling Feature Set")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.write("Numeric features:", results.numeric_features)
    with fcol2:
        st.write("Categorical features:", results.categorical_features)

    with st.expander("Feature documentation", expanded=False):
        feat_doc_df = pd.DataFrame(
            [{"feature": k, "description": v} for k, v in core.FEATURE_DOCUMENTATION.items()]
        )
        st.dataframe(feat_doc_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data source: Olist Brazilian E-Commerce Public Dataset (public, anonymised). This project is an academic "
    "analysis and does not represent access to Olist's internal/confidential systems."
)
