"""
app.py
======
E-Commerce Delivery Risk Intelligence -- Streamlit dashboard.

This file is presentation-only. Every analytical computation (cleaning,
feature engineering, modelling, evaluation, risk scoring, managerial
insights) is performed in core.py and consumed here via run_pipeline().
Run with:  streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

# --------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery Risk Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1F3B57"
ACCENT = "#C0392B"
GOOD = "#1E8449"
WARN = "#B9770E"
NEUTRAL = "#5D6D7E"

st.markdown(
    """
    <style>
    .main {background-color: #FAFBFC;}
    .metric-card {
        background-color: white; border-radius: 10px; padding: 1.1rem 1.2rem;
        border: 1px solid #E5E8EC; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    h1, h2, h3 { color: #1F3B57; }
    .insight-box {
        background-color: #F4F6F8; border-left: 4px solid #1F3B57;
        padding: 0.7rem 1rem; margin-bottom: 0.6rem; border-radius: 4px; font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# DATA AVAILABILITY CHECK (graceful failure, no raw tracebacks)
# --------------------------------------------------------------------------
st.title("📦 E-Commerce Delivery Risk Intelligence")
st.caption(
    "Predicting Late Orders Using Machine Learning · Olist Brazilian E-Commerce Public Dataset · "
    "DAML / Working-with-AI Project, IIM Ranchi"
)

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
st.sidebar.header("🔎 Dashboard Filters")
st.sidebar.caption("Filters slice the descriptive views below. Model training itself is fixed and cached.")

all_states = sorted(df["customer_state"].dropna().unique().tolist())
sel_cust_states = st.sidebar.multiselect("Customer state", all_states, default=[])

all_seller_states = sorted(df["seller_state"].dropna().unique().tolist())
sel_seller_states = st.sidebar.multiselect("Seller state", all_seller_states, default=[])

all_categories = sorted(df["dominant_category"].dropna().unique().tolist())
sel_categories = st.sidebar.multiselect("Product category", all_categories, default=[])

min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Purchase date range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

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
    "🎯 Executive Control Tower",
    "🌎 Delivery Risk Intelligence",
    "📊 Risk Drivers",
    "🔍 Order & Risk Drilldown",
    "⚖️ Model Comparison",
    "🧭 Managerial Action Center",
    "📋 Data & Methodology",
])

# ============================== TAB 1 ====================================
with tab1:
    st.subheader("Executive Control Tower")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders (modelled)", f"{len(df_f):,}")
    c2.metric("Late-Delivery Rate", f"{df_f['late_delivery'].mean():.1%}" if len(df_f) else "N/A")
    n_high_risk = (pred_table_f["risk_segment"] == "High Risk").sum() if len(pred_table_f) else 0
    c3.metric("High-Risk Orders (scored sample)", f"{n_high_risk:,}")
    c4.metric("Best Model", results.best_model_name)

    c5, c6, c7, c8 = st.columns(4)
    avg_delay_days = None
    if len(df_f):
        delivered_delay = (df_f["order_delivered_customer_date"] - df_f["order_estimated_delivery_date"]).dt.days
        avg_delay_days = delivered_delay[delivered_delay > 0].mean()
    c5.metric("Avg. Delay (late orders only)", f"{avg_delay_days:.1f} days" if avg_delay_days == avg_delay_days else "N/A")
    c6.metric("Recall (late class)", f"{results.comparison_df.loc[results.best_model_name, 'Recall']:.1%}")
    c7.metric("F1-score", f"{results.comparison_df.loc[results.best_model_name, 'F1-score']:.3f}")
    c8.metric("ROC-AUC", f"{results.comparison_df.loc[results.best_model_name, 'ROC-AUC']:.3f}")

    st.markdown("#### Delivery Trend")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        month_df = results.eda["late_rate_by_month"]
        fig = px.line(month_df, x="purchase_month_period", y="late_rate", markers=True,
                      labels={"purchase_month_period": "Purchase Month", "late_rate": "Late-Delivery Rate"},
                      title="Late-Delivery Rate Over Time")
        fig.update_traces(line_color=ACCENT)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if len(pred_table_f):
            seg_counts = pred_table_f["risk_segment"].value_counts().reindex(risk_levels).fillna(0)
            fig2 = px.pie(values=seg_counts.values, names=seg_counts.index, hole=0.5,
                          color=seg_counts.index,
                          color_discrete_map={"High Risk": ACCENT, "Medium Risk": WARN, "Low Risk": GOOD},
                          title="Risk Distribution (scored orders)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No scored orders match the current filters.")

    st.markdown("#### Key Management Callouts")
    for insight in results.managerial_insights[:4]:
        st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)

# ============================== TAB 2 ====================================
with tab2:
    st.subheader("Delivery Risk Intelligence")
    st.caption("Descriptive risk patterns across geography, category and time (filters applied).")

    col1, col2 = st.columns(2)
    with col1:
        cust_state_df = df_f.groupby("customer_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        cust_state_df = cust_state_df[cust_state_df["count"] >= 5].sort_values("mean", ascending=False)
        fig = px.bar(cust_state_df, x="customer_state", y="mean", color="mean",
                     color_continuous_scale=["#1E8449", "#B9770E", "#C0392B"],
                     labels={"mean": "Late-Delivery Rate", "customer_state": "Customer State"},
                     title="Late-Delivery Rate by Customer State")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        seller_state_df = df_f.groupby("seller_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        seller_state_df = seller_state_df[seller_state_df["count"] >= 5].sort_values("mean", ascending=False)
        fig = px.bar(seller_state_df, x="seller_state", y="mean", color="mean",
                     color_continuous_scale=["#1E8449", "#B9770E", "#C0392B"],
                     labels={"mean": "Late-Delivery Rate", "seller_state": "Seller State"},
                     title="Late-Delivery Rate by Seller State")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cat_df = df_f.groupby("dominant_category")["late_delivery"].agg(["mean", "count"]).reset_index()
        cat_df = cat_df[cat_df["count"] >= 15].sort_values("mean", ascending=False).head(15)
        fig = px.bar(cat_df, x="mean", y="dominant_category", orientation="h",
                     labels={"mean": "Late-Delivery Rate", "dominant_category": "Product Category"},
                     title="Top 15 Product Categories by Late-Delivery Rate", color="mean",
                     color_continuous_scale=["#1E8449", "#B9770E", "#C0392B"])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(df_f.assign(**{"Delivery Outcome": df_f["late_delivery"].map({0: "On-Time", 1: "Late"})}),
                            x="distance_km", color="Delivery Outcome", barmode="overlay", nbins=40,
                            color_discrete_map={"On-Time": GOOD, "Late": ACCENT},
                            title="Seller–Customer Distance Distribution by Outcome")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Delivery Delay Distribution (late orders)")
    late_only = df_f[df_f["late_delivery"] == 1].copy()
    if len(late_only):
        late_only["delay_days"] = (late_only["order_delivered_customer_date"] - late_only["order_estimated_delivery_date"]).dt.days
        fig = px.box(late_only, y="delay_days", points="outliers", title="Distribution of Delay (days beyond promise)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No late orders in the current filter selection.")

# ============================== TAB 3 ====================================
with tab3:
    st.subheader("Risk Drivers")
    st.caption(f"Feature relevance computed via permutation importance on the selected model ({results.best_model_name}).")

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

    st.markdown("##### Dimensionality Reduction Assessment")
    st.write(results.dim_reduction["explanation"])
    if results.dim_reduction["recommended"]:
        st.dataframe(results.dim_reduction["explained_variance_table"], use_container_width=True, hide_index=True)
    if results.dim_reduction["high_corr_pairs"]:
        st.caption(f"High-correlation numeric pairs detected: {results.dim_reduction['high_corr_pairs']}")

    st.markdown("##### Value vs. Delay Relationship")
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

# ============================== TAB 4 ====================================
with tab4:
    st.subheader("Order & Risk Drilldown")
    st.caption("Scored orders come from the held-out test set used to evaluate the selected model.")

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

    st.download_button(
        "⬇️ Download filtered high-risk orders (CSV)",
        data=view_df[view_df["risk_segment"] == "High Risk"].to_csv(index=False).encode("utf-8"),
        file_name="high_risk_orders.csv", mime="text/csv",
    )
    st.download_button(
        "⬇️ Download full filtered drilldown table (CSV)",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="order_risk_drilldown.csv", mime="text/csv",
    )

# ============================== TAB 5 ====================================
with tab5:
    st.subheader("Model Comparison")
    st.caption(f"Deterministic 75/25 stratified train/test split, random_state=42. KNN k was selected via cross-validated "
               f"recall search (k={results.model_meta['best_k']}).")

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

# ============================== TAB 6 ====================================
with tab6:
    st.subheader("Managerial Action Center")

    st.markdown("##### Major Findings")
    for insight in results.managerial_insights:
        st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)

    st.markdown("##### Intervention Framework")
    st.dataframe(results.intervention_framework, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download intervention framework (CSV)",
        data=results.intervention_framework.to_csv(index=False).encode("utf-8"),
        file_name="intervention_framework.csv", mime="text/csv",
    )

    st.markdown("##### Suggested Implementation Roadmap")
    st.markdown(
        """
        1. **Week 1–2:** Deploy the scoring model in shadow mode alongside existing operations; validate risk
           segments against actual outcomes without changing workflows yet.
        2. **Week 3–4:** Route High-Risk orders to a daily proactive-monitoring queue for fulfilment operations.
        3. **Month 2:** Share seller/state-level risk drivers with Seller Management and Logistics Coordination
           for targeted, evidence-based conversations (not blanket penalties).
        4. **Month 3+:** Re-evaluate model performance on fresh data; retrain if Recall degrades materially,
           and reassess the High-Risk probability threshold against actual intervention capacity.
        """
    )

    st.markdown("##### Responsible-AI / Analytical Limitations")
    for lim in core.DATA_REQUIREMENTS["data_limitations"]:
        st.markdown(f"- {lim}")
    st.markdown(
        "- Model outputs are **probabilistic risk scores**, not certainties, and should support -- not replace -- "
        "managerial judgment.\n"
        "- The models establish **association/prediction**, not proven causal mechanisms; interventions should be "
        "piloted and measured before wide rollout.\n"
        "- Public historical data may not reflect current carrier partnerships, seller mix, or seasonal effects."
    )

    if results.review_narrative is not None and len(results.review_narrative) == 2:
        st.markdown("##### Downstream Customer-Experience Signal (descriptive only, not used as a model input)")
        rn = results.review_narrative.copy()
        rn["late_delivery"] = rn["late_delivery"].map({0: "On-Time", 1: "Late"})
        fig = px.bar(rn, x="late_delivery", y="review_score", color="late_delivery",
                     color_discrete_map={"On-Time": GOOD, "Late": ACCENT},
                     title="Average Review Score by Delivery Outcome")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shown for managerial context only. Review data occurs after delivery and is never used as a model predictor.")

# ============================== TAB 7 ====================================
with tab7:
    st.subheader("Data Requirement Analysis & Methodology")

    dr = core.DATA_REQUIREMENTS
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

    st.divider()
    st.markdown("##### Data Cleaning Log")
    for table, log in results.cleaning_logs.items():
        with st.expander(f"`{table}` cleaning log"):
            st.json(log)

    st.markdown("##### Multi-Table Integration Log")
    st.json(results.integration_log)

    st.markdown("##### Target Construction Log")
    st.json(results.target_log)
    st.caption(
        f"{len(results.df_excluded):,} orders were excluded from modelling (not delivered, or missing "
        "actual/estimated dates) and are reported here for transparency, not silently dropped."
    )

    st.markdown("##### Missing-Value Analysis (pre-treatment)")
    st.dataframe(results.missing_summary, use_container_width=True, hide_index=True)
    st.markdown("##### Missing-Value Treatment Log")
    st.json(results.missing_treatment_log)

    st.markdown("##### Outlier Analysis (IQR method)")
    st.dataframe(results.outlier_summary, use_container_width=True, hide_index=True)
    st.markdown("##### Outlier Treatment Log (winsorized/capped at 3×IQR, not deleted)")
    st.json(results.outlier_treatment_log)

    st.markdown("##### Final Modelling Feature Set")
    st.write("Numeric features:", results.numeric_features)
    st.write("Categorical features:", results.categorical_features)

    st.caption("Feature documentation:")
    feat_doc_df = pd.DataFrame(
        [{"feature": k, "description": v} for k, v in core.FEATURE_DOCUMENTATION.items()]
    )
    st.dataframe(feat_doc_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data source: Olist Brazilian E-Commerce Public Dataset (public, anonymised). This project is an academic "
    "analysis and does not represent access to Olist's internal/confidential systems."
)
