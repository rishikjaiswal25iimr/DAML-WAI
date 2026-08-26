"""
core.py
=======
E-Commerce Delivery Risk Intelligence: Predicting Late Orders Using Machine Learning
Single analytical engine (single source of truth) for the Olist late-delivery project.

Course   : Data Analytics and Machine Learning Techniques (DAML), IIM Ranchi
Context  : MBA Working-with-AI (WAI) assignment

DESIGN PRINCIPLE
-----------------
Every piece of analytical logic (data integration, cleaning, missing-value
treatment, outlier treatment, feature engineering, feature selection, EDA
aggregation, modelling, evaluation, risk profiling and managerial
recommendations) lives HERE. app.py only calls these functions and renders
the results. There is exactly one analytical pipeline.

DATA LEAKAGE POLICY (read before editing predictors)
-----------------------------------------------------
The prediction problem is framed as: "at the time an order is placed and
paid for, can we flag it as high risk of arriving after the promised date?"
Consequently the following are NEVER used as predictors:
    * order_delivered_customer_date (defines the target itself)
    * order_delivered_carrier_date  (only known once the parcel has shipped;
      using it would mean the model needs information from deep inside the
      fulfilment process, which defeats the purpose of an *early* warning
      system and would artificially inflate performance)
    * review_score / review_comment_* (reviews are written after the
      customer has experienced -- or not experienced -- a delay)
The only date-derived predictor taken from the post-purchase, pre-shipping
window is order_approved_at (payment approval), which happens very close to
checkout and is a legitimate "early operations" signal, not a delivery
signal.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover - core.py must still import without streamlit
    _HAS_STREAMLIT = False


def _cache(func):
    """Use Streamlit's data cache when available (so the pipeline runs once per
    session and filters do not retrain models), otherwise run uncached."""
    if _HAS_STREAMLIT:
        return st.cache_data(show_spinner=False)(func)
    return func


RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. DATA REQUIREMENT ANALYSIS (exposed for the dashboard "Data Requirements"
#    panel -- this is a first-class deliverable of the course, not decoration)
# --------------------------------------------------------------------------
DATA_REQUIREMENTS = {
    "business_objective": (
        "Enable Olist's operations and seller-management teams to identify, at "
        "order-placement time, which orders carry an elevated risk of arriving "
        "after the promised delivery date, so that proactive intervention "
        "(carrier escalation, seller follow-up, customer communication) can be "
        "targeted rather than applied uniformly."
    ),
    "technical_objective": (
        "Build and compare two supervised classification models (K-Nearest "
        "Neighbours and Naive Bayes) that predict a binary late_delivery label "
        "from order-, product-, payment-, seller- and geography-level features "
        "that are available early in the order lifecycle, and translate model "
        "output into an operational risk-scoring and intervention framework."
    ),
    "unit_of_analysis": "One row = one delivered order (order_id).",
    "target_variable": (
        "late_delivery (1 = order_delivered_customer_date > "
        "order_estimated_delivery_date, 0 = otherwise), computed only for "
        "orders with status 'delivered' and non-missing actual/estimated dates."
    ),
    "predictor_families": [
        "Order composition (item count, seller count, product count, order value, freight)",
        "Payment characteristics (payment value, installments, payment type mix)",
        "Product characteristics (category, weight, dimensions/volume, listing richness)",
        "Timing known at/near purchase (purchase calendar features, approval time, promised delivery window)",
        "Geography (customer state/region, seller state/region, same-state flag, seller-customer distance)",
    ],
    "excluded_as_predictors_due_to_leakage": [
        "order_delivered_customer_date (defines the target)",
        "order_delivered_carrier_date (only known after dispatch)",
        "review_score, review_comment_title, review_comment_message, "
        "review_creation_date, review_answer_timestamp (post-experience)",
    ],
    "source_tables": [
        "olist_orders_dataset.csv", "olist_customers_dataset.csv",
        "olist_order_items_dataset.csv", "olist_products_dataset.csv",
        "olist_sellers_dataset.csv", "olist_order_payments_dataset.csv",
        "olist_geolocation_dataset.csv", "product_category_name_translation.csv",
        "olist_order_reviews_dataset.csv (secondary/descriptive use only)",
    ],
    "keys_and_relationships": [
        "orders.customer_id -> customers.customer_id (1:1)",
        "orders.order_id -> order_items.order_id (1:many; aggregated to order level)",
        "order_items.product_id -> products.product_id (many:1)",
        "order_items.seller_id -> sellers.seller_id (many:1)",
        "products.product_category_name -> translation.product_category_name (many:1)",
        "customers/sellers zip_code_prefix -> geolocation.geolocation_zip_code_prefix (many:1, deduplicated by mean lat/lng)",
        "orders.order_id -> payments.order_id (1:many; aggregated to order level)",
        "orders.order_id -> reviews.order_id (1:many, used only for descriptive narrative)",
    ],
    "data_limitations": [
        "Public, anonymised, historical (2016-2018) Brazilian marketplace data -- "
        "no access to Olist's live/internal systems.",
        "Geolocation is resolved at zip-code-prefix granularity (not exact address), "
        "so distance is approximate.",
        "No macro/weather/traffic/holiday-calendar data is available to explain "
        "some delays.",
        "A small number of orders are missing item, payment or product records "
        "and are excluded from modelling with the exclusion counted and reported.",
    ],
    "assumptions": [
        "Only orders with status 'delivered' and complete actual/estimated dates "
        "define a valid target and enter the modelling dataset.",
        "Order-level aggregation from item-level records (e.g. sum of price, mean "
        "of weight) is a reasonable simplification of multi-item / multi-seller orders.",
        "Payment information is assumed to be finalised at/near checkout and is "
        "therefore treated as an early, non-leaky signal.",
    ],
}

FILE_NAMES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}

STATE_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}


# --------------------------------------------------------------------------
# 2. FILE DISCOVERY (robust, no hard-coded local paths)
# --------------------------------------------------------------------------
def discover_data_directory(explicit_dir: Optional[str] = None) -> str:
    """Search a short list of sensible candidate directories for all nine
    Olist CSV files and return the first directory that has them all.

    This lets the same code run unmodified on a laptop, on Streamlit Cloud,
    or from a GitHub-cloned repo, as long as the CSVs sit next to app.py.
    """
    candidates = []
    if explicit_dir:
        candidates.append(explicit_dir)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(here)
    except NameError:
        pass
    candidates.append(os.getcwd())
    candidates.append(".")

    seen = set()
    ordered_candidates = []
    for c in candidates:
        c_norm = os.path.abspath(c)
        if c_norm not in seen:
            seen.add(c_norm)
            ordered_candidates.append(c_norm)

    required = list(FILE_NAMES.values())
    for directory in ordered_candidates:
        if all(os.path.isfile(os.path.join(directory, fname)) for fname in required):
            return directory

    # Nothing matched fully -- raise a clear, actionable error rather than a
    # cryptic pandas FileNotFoundError deep inside the pipeline.
    missing_report = {}
    best_dir = ordered_candidates[0] if ordered_candidates else "."
    for directory in ordered_candidates:
        missing = [f for f in required if not os.path.isfile(os.path.join(directory, f))]
        missing_report[directory] = missing
        if len(missing) < len(missing_report.get(best_dir, required)):
            best_dir = directory

    raise FileNotFoundError(
        "Could not locate all nine Olist CSV files. Place the CSVs in the same "
        "folder as app.py / core.py. Directories checked and missing files:\n"
        + "\n".join(f"  - {d}: missing {miss}" for d, miss in missing_report.items())
    )


def check_data_availability(explicit_dir: Optional[str] = None) -> tuple[bool, str]:
    """Non-raising helper used by app.py to show a friendly banner instead of
    crashing on startup when data is not yet present."""
    try:
        directory = discover_data_directory(explicit_dir)
        return True, directory
    except FileNotFoundError as exc:
        return False, str(exc)


# --------------------------------------------------------------------------
# 3. GEO HELPERS
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# 4. RAW LOAD
# --------------------------------------------------------------------------
@_cache
def load_raw_tables(data_dir: str) -> dict:
    """Read all nine CSVs with appropriate dtypes/date parsing. Returns a
    dict of raw (uncleaned) DataFrames keyed like FILE_NAMES."""

    def path(key):
        return os.path.join(data_dir, FILE_NAMES[key])

    orders = pd.read_csv(
        path("orders"),
        parse_dates=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    customers = pd.read_csv(path("customers"), dtype={"customer_zip_code_prefix": str})
    items = pd.read_csv(path("items"), parse_dates=["shipping_limit_date"])
    products = pd.read_csv(path("products")).rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    )
    sellers = pd.read_csv(path("sellers"), dtype={"seller_zip_code_prefix": str})
    payments = pd.read_csv(path("payments"))
    geolocation = pd.read_csv(path("geolocation"), dtype={"geolocation_zip_code_prefix": str})
    category_translation = pd.read_csv(path("category_translation"))
    reviews = pd.read_csv(
        path("reviews"),
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )

    return {
        "orders": orders, "customers": customers, "items": items,
        "products": products, "sellers": sellers, "payments": payments,
        "geolocation": geolocation, "category_translation": category_translation,
        "reviews": reviews,
    }


# --------------------------------------------------------------------------
# 5. TABLE-LEVEL CLEANING
# --------------------------------------------------------------------------
def _zfill_zip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(5)


def clean_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(orders)}
    df = orders.drop_duplicates(subset="order_id").copy()
    df["order_status"] = df["order_status"].str.lower().str.strip()
    # Impossible chronology: approved before purchase, or delivered before approved.
    # These are data errors, not business signal -- we null the offending
    # timestamp rather than drop the whole order, so the order can still be
    # used if it is otherwise a valid 'delivered' record.
    bad_approval = df["order_approved_at"] < df["order_purchase_timestamp"]
    df.loc[bad_approval, "order_approved_at"] = pd.NaT
    log["rows_with_impossible_approval_time_nulled"] = int(bad_approval.sum())
    log["final_rows"] = len(df)
    log["duplicates_removed"] = log["initial_rows"] - len(df)
    return df, log


def clean_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(customers)}
    df = customers.drop_duplicates(subset="customer_id").copy()
    df["customer_state"] = df["customer_state"].str.upper().str.strip()
    df["customer_city"] = df["customer_city"].str.strip().str.title()
    df["customer_zip_code_prefix"] = _zfill_zip(df["customer_zip_code_prefix"])
    log["final_rows"] = len(df)
    return df, log


def clean_sellers(sellers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(sellers)}
    df = sellers.drop_duplicates(subset="seller_id").copy()
    df["seller_state"] = df["seller_state"].str.upper().str.strip()
    df["seller_city"] = df["seller_city"].str.strip().str.title()
    df["seller_zip_code_prefix"] = _zfill_zip(df["seller_zip_code_prefix"])
    log["final_rows"] = len(df)
    return df, log


def clean_products(products: pd.DataFrame, category_translation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(products)}
    df = products.drop_duplicates(subset="product_id").copy()
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df = df.merge(category_translation, on="product_category_name", how="left")
    df["product_category_name_english"] = (
        df["product_category_name_english"].fillna(df["product_category_name"]).fillna("unknown")
    )
    numeric_cols = [
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        "product_name_length", "product_description_length", "product_photos_qty",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Impossible physical values (<=0) treated as missing, to be imputed later.
    for c in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        invalid = df[c] <= 0
        log[f"{c}_invalid_set_to_missing"] = int(invalid.sum())
        df.loc[invalid, c] = np.nan
    df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    log["final_rows"] = len(df)
    return df, log


def clean_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(items)}
    df = items.drop_duplicates(subset=["order_id", "order_item_id"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce")
    invalid = (df["price"] <= 0) | df["price"].isna()
    log["rows_with_non_positive_or_missing_price_removed"] = int(invalid.sum())
    df = df[~invalid].copy()
    df["freight_value"] = df["freight_value"].fillna(0).clip(lower=0)
    log["final_rows"] = len(df)
    log["rows_removed"] = log["initial_rows"] - log["final_rows"]
    return df, log


def clean_payments(payments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(payments)}
    df = payments.drop_duplicates().copy()
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce").fillna(0)
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce").fillna(1).clip(lower=1)
    df["payment_type"] = df["payment_type"].replace("not_defined", "unknown").fillna("unknown")
    log["final_rows"] = len(df)
    return df, log


def clean_geolocation(geolocation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate to one lat/lng/state per zip prefix and drop coordinates
    that fall outside Brazil's approximate bounding box (data errors)."""
    log = {"initial_rows": len(geolocation)}
    df = geolocation.copy()
    df["geolocation_zip_code_prefix"] = _zfill_zip(df["geolocation_zip_code_prefix"])
    in_brazil = df["geolocation_lat"].between(-34, 6) & df["geolocation_lng"].between(-74, -32)
    log["rows_outside_brazil_bbox_removed"] = int((~in_brazil).sum())
    df = df[in_brazil]
    agg = (
        df.groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            state=("geolocation_state", lambda s: s.mode().iat[0] if not s.mode().empty else np.nan),
        )
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    log["final_zip_prefixes"] = len(agg)
    return agg, log


def clean_reviews(reviews: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"initial_rows": len(reviews)}
    df = reviews.drop_duplicates(subset="review_id").copy()
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    log["final_rows"] = len(df)
    return df, log


@_cache
def run_all_cleaning(data_dir: str) -> tuple[dict, dict]:
    raw = load_raw_tables(data_dir)
    cleaned = {}
    logs = {}
    cleaned["orders"], logs["orders"] = clean_orders(raw["orders"])
    cleaned["customers"], logs["customers"] = clean_customers(raw["customers"])
    cleaned["sellers"], logs["sellers"] = clean_sellers(raw["sellers"])
    cleaned["products"], logs["products"] = clean_products(raw["products"], raw["category_translation"])
    cleaned["items"], logs["items"] = clean_items(raw["items"])
    cleaned["payments"], logs["payments"] = clean_payments(raw["payments"])
    cleaned["geolocation"], logs["geolocation"] = clean_geolocation(raw["geolocation"])
    cleaned["reviews"], logs["reviews"] = clean_reviews(raw["reviews"])
    return cleaned, logs


# --------------------------------------------------------------------------
# 6. MULTI-TABLE INTEGRATION (item-level -> order-level aggregation)
# --------------------------------------------------------------------------
def aggregate_order_items(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    merged = items.merge(products, on="product_id", how="left")

    def _mode_or_na(s: pd.Series):
        m = s.mode()
        return m.iat[0] if not m.empty else np.nan

    agg = merged.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        product_count=("product_id", "nunique"),
        distinct_category_count=("product_category_name_english", "nunique"),
        total_price=("price", "sum"),
        avg_item_price=("price", "mean"),
        max_item_price=("price", "max"),
        total_freight=("freight_value", "sum"),
        avg_freight=("freight_value", "mean"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_volume_cm3=("product_volume_cm3", "mean"),
        max_product_volume_cm3=("product_volume_cm3", "max"),
        avg_photos_qty=("product_photos_qty", "mean"),
        avg_name_length=("product_name_length", "mean"),
        avg_description_length=("product_description_length", "mean"),
        dominant_category=("product_category_name_english", _mode_or_na),
        dominant_seller_id=("seller_id", _mode_or_na),
    ).reset_index()
    return agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    def _mode_or_na(s: pd.Series):
        m = s.mode()
        return m.iat[0] if not m.empty else "unknown"

    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
        dominant_payment_type=("payment_type", _mode_or_na),
    ).reset_index()
    return agg


@_cache
def integrate_full_dataset(data_dir: str) -> tuple[pd.DataFrame, dict]:
    """Build the order-level analytical dataset by joining every table on
    its correct key, respecting the 1:many relationships in the raw data."""
    cleaned, cleaning_logs = run_all_cleaning(data_dir)
    orders, customers, sellers = cleaned["orders"], cleaned["customers"], cleaned["sellers"]
    items, products, payments = cleaned["items"], cleaned["products"], cleaned["payments"]
    geo = cleaned["geolocation"]

    items_agg = aggregate_order_items(items, products)
    payments_agg = aggregate_payments(payments)

    df = orders.merge(customers, on="customer_id", how="left")
    log = {"after_customers_merge": len(df)}

    df = df.merge(items_agg, on="order_id", how="left")
    log["orders_without_any_item_record"] = int(df["item_count"].isna().sum())

    df = df.merge(payments_agg, on="order_id", how="left")
    log["orders_without_any_payment_record"] = int(df["total_payment_value"].isna().sum())

    df = df.merge(
        sellers[["seller_id", "seller_state", "seller_city", "seller_zip_code_prefix"]],
        left_on="dominant_seller_id", right_on="seller_id", how="left",
    )

    df = df.merge(
        geo.rename(columns={"zip_code_prefix": "customer_zip_code_prefix",
                             "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left",
    )
    df = df.merge(
        geo.rename(columns={"zip_code_prefix": "seller_zip_code_prefix",
                             "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left",
    )
    log["orders_missing_customer_geo"] = int(df["customer_lat"].isna().sum())
    log["orders_missing_seller_geo"] = int(df["seller_lat"].isna().sum())

    df["distance_km"] = haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])

    log["final_integrated_rows"] = len(df)
    return df, {**cleaning_logs, "integration": log}


# --------------------------------------------------------------------------
# 7. TARGET DEFINITION (leakage-checked)
# --------------------------------------------------------------------------
def compute_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log = {"total_orders": len(df)}
    valid_status = df["order_status"] == "delivered"
    has_dates = df["order_delivered_customer_date"].notna() & df["order_estimated_delivery_date"].notna()
    usable = valid_status & has_dates

    log["excluded_not_delivered_status"] = int((~valid_status).sum())
    log["excluded_missing_dates_among_delivered"] = int((valid_status & ~has_dates).sum())
    log["usable_for_modelling"] = int(usable.sum())

    df_model = df[usable].copy()
    df_excluded = df[~usable].copy()
    df_model["late_delivery"] = (
        df_model["order_delivered_customer_date"] > df_model["order_estimated_delivery_date"]
    ).astype(int)
    log["late_rate_in_modelling_data"] = float(df_model["late_delivery"].mean())
    return df_model, df_excluded, log


# --------------------------------------------------------------------------
# 8. MISSING VALUES
# --------------------------------------------------------------------------
def missing_value_analysis(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in columns:
        miss = int(df[c].isna().sum())
        rows.append({"column": c, "missing_count": miss, "missing_pct": round(100 * miss / n, 2)})
    out = pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    return out


def handle_missing_values(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    log = {}
    for c in numeric_cols:
        if c in df.columns and df[c].isna().any():
            filled = int(df[c].isna().sum())
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)
            log[c] = {"strategy": "median_imputation", "value_used": round(float(median_val), 2), "rows_filled": filled}
    for c in categorical_cols:
        if c in df.columns and df[c].isna().any():
            filled = int(df[c].isna().sum())
            df[c] = df[c].fillna("Unknown")
            log[c] = {"strategy": "explicit_unknown_category", "rows_filled": filled}
    # distance_km missing (no geo match) -> keep as median + explicit missing flag,
    # since "missingness" here is itself potentially informative.
    if "distance_km" in df.columns:
        df["distance_missing_flag"] = df["distance_km"].isna().astype(int)
    return df, log


# --------------------------------------------------------------------------
# 9. OUTLIERS
# --------------------------------------------------------------------------
def outlier_analysis(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append({
            "column": c, "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
            "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
            "outlier_count": n_out, "outlier_pct": round(100 * n_out / len(s), 2),
        })
    return pd.DataFrame(rows)


def treat_outliers(df: pd.DataFrame, numeric_cols: list[str], outlier_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winsorize (cap, do not delete) at 3x IQR beyond Q1/Q3. Capping rather
    than deleting preserves sample size and acknowledges that extreme order
    values / weights are often legitimate business events (bulk orders,
    heavy furniture, etc.) rather than data errors."""
    df = df.copy()
    log = {}
    for _, row in outlier_summary.iterrows():
        c = row["column"]
        if c not in df.columns:
            continue
        iqr = row["IQR"]
        lower = row["Q1"] - 3 * iqr
        upper = row["Q3"] + 3 * iqr
        n_capped = int(((df[c] < lower) | (df[c] > upper)).sum())
        df[c] = df[c].clip(lower=lower, upper=upper)
        log[c] = {"capped_at": [round(lower, 2), round(upper, 2)], "rows_capped": n_capped}
    return df, log


# --------------------------------------------------------------------------
# 10. FEATURE ENGINEERING
# --------------------------------------------------------------------------
FEATURE_DOCUMENTATION = {
    "purchase_month": "Calendar month of purchase (seasonality effects on fulfilment load).",
    "purchase_weekday": "Day of week of purchase (0=Mon).",
    "purchase_hour": "Hour of day of purchase.",
    "approval_time_hours": "Hours between purchase and payment approval; early operational friction signal.",
    "estimated_window_days": "Days between purchase and the promised delivery date (the delivery promise itself).",
    "item_count": "Number of line items in the order.",
    "seller_count": "Number of distinct sellers fulfilling the order (multi-seller orders are operationally harder to coordinate).",
    "product_count": "Number of distinct products in the order.",
    "distinct_category_count": "Number of distinct product categories in the order.",
    "total_price": "Sum of item prices (order merchandise value).",
    "avg_item_price": "Average price per item.",
    "total_freight": "Sum of freight charges across items.",
    "freight_ratio": "total_freight / total_price; a high ratio can indicate bulky/remote shipments.",
    "avg_product_weight_g": "Average product weight in the order.",
    "avg_product_volume_cm3": "Average product volume (L x H x W) in the order.",
    "avg_photos_qty": "Average number of listing photos (proxy for listing/seller quality).",
    "distance_km": "Approximate great-circle distance between dominant seller and customer zip-code centroids.",
    "same_state_flag": "1 if the dominant seller and the customer are in the same state.",
    "customer_region": "Brazilian macro-region of the customer (derived from state).",
    "seller_region": "Brazilian macro-region of the dominant seller (derived from state).",
    "total_payment_value": "Total amount paid for the order.",
    "max_installments": "Maximum number of instalments chosen across payment records.",
    "n_payment_methods": "Number of distinct payment methods used for the order.",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    approval_hours = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["approval_time_hours"] = approval_hours.clip(lower=0)

    df["estimated_window_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days.clip(lower=0)

    df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
    df["freight_ratio"] = df["freight_ratio"].fillna(0)

    df["same_state_flag"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["customer_region"] = df["customer_state"].map(STATE_TO_REGION).fillna("Unknown")
    df["seller_region"] = df["seller_state"].map(STATE_TO_REGION).fillna("Unknown")

    return df


# --------------------------------------------------------------------------
# 11. FEATURE SELECTION
# --------------------------------------------------------------------------
CANDIDATE_NUMERIC = [
    "item_count", "seller_count", "product_count", "distinct_category_count",
    "total_price", "avg_item_price", "max_item_price", "total_freight", "avg_freight",
    "freight_ratio", "avg_product_weight_g", "avg_product_volume_cm3", "max_product_volume_cm3",
    "avg_photos_qty", "avg_name_length", "avg_description_length",
    "approval_time_hours", "estimated_window_days", "distance_km",
    "total_payment_value", "max_installments", "n_payment_methods",
    "purchase_month", "purchase_weekday", "purchase_hour", "same_state_flag",
]
CANDIDATE_CATEGORICAL = [
    "customer_state", "seller_state", "customer_region", "seller_region",
    "dominant_category", "dominant_payment_type",
]


def select_features(df: pd.DataFrame, target_col: str = "late_delivery") -> tuple[list, list, pd.DataFrame]:
    rows = []
    y = df[target_col].values

    # --- numeric: correlation + mutual information ---
    numeric_scores = {}
    for c in CANDIDATE_NUMERIC:
        if c not in df.columns:
            continue
        x = df[c].fillna(df[c].median())
        corr = np.corrcoef(x, y)[0, 1] if x.std() > 0 else 0.0
        numeric_scores[c] = corr
    mi_input = df[[c for c in CANDIDATE_NUMERIC if c in df.columns]].fillna(0)
    mi_values = mutual_info_classif(mi_input, y, discrete_features=False, random_state=RANDOM_STATE)
    mi_scores = dict(zip(mi_input.columns, mi_values))

    # Drop highly collinear pairs (|corr| > 0.9), keeping the one with higher MI.
    corr_matrix = mi_input.corr().abs()
    to_drop = set()
    cols = list(mi_input.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.iloc[i, j] > 0.9:
                a, b = cols[i], cols[j]
                weaker = a if mi_scores.get(a, 0) < mi_scores.get(b, 0) else b
                to_drop.add(weaker)

    final_numeric = []
    for c in CANDIDATE_NUMERIC:
        if c not in df.columns:
            continue
        keep = c not in to_drop
        rows.append({
            "feature": c, "type": "numeric",
            "correlation_with_target": round(numeric_scores.get(c, 0), 4),
            "mutual_information": round(mi_scores.get(c, 0), 4),
            "decision": "keep" if keep else "drop",
            "reason": "retained: informative and non-redundant" if keep else "dropped: highly collinear with a stronger feature",
        })
        if keep:
            final_numeric.append(c)

    # --- categorical: mutual information on label-encoded values, keep low-cardinality/high-signal ---
    final_categorical = []
    for c in CANDIDATE_CATEGORICAL:
        if c not in df.columns:
            continue
        codes = df[c].astype("category").cat.codes.values.reshape(-1, 1)
        mi_val = mutual_info_classif(codes, y, discrete_features=True, random_state=RANDOM_STATE)[0]
        cardinality = df[c].nunique()
        # Very high-cardinality raw state fields are kept (business-relevant),
        # but if a feature effectively duplicates region information with a
        # sub-1e-4 MI gain over its region counterpart it is still retained
        # here since regional AND state-level views are both used in the
        # dashboard's Delivery Risk Intelligence tab.
        keep = True
        rows.append({
            "feature": c, "type": "categorical", "correlation_with_target": np.nan,
            "mutual_information": round(mi_val, 4),
            "decision": "keep" if keep else "drop",
            "reason": f"retained: business-relevant categorical driver (cardinality={cardinality})",
        })
        if keep:
            final_categorical.append(c)

    selection_report = pd.DataFrame(rows).sort_values(
        by=["type", "mutual_information"], ascending=[True, False]
    ).reset_index(drop=True)
    return final_numeric, final_categorical, selection_report


def assess_dimensionality_reduction(df: pd.DataFrame, numeric_features: list[str]) -> dict:
    """Dimensionality reduction (PCA) is only warranted when many numeric
    predictors are strongly correlated with each other. We check this
    explicitly rather than forcing PCA into the pipeline."""
    corr = df[numeric_features].fillna(0).corr().abs()
    n = len(numeric_features)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] > 0.8:
                high_corr_pairs.append((numeric_features[i], numeric_features[j], round(corr.iloc[i, j], 3)))

    if len(high_corr_pairs) < 3:
        return {
            "recommended": False,
            "high_corr_pairs": high_corr_pairs,
            "explanation": (
                f"Only {len(high_corr_pairs)} numeric feature pair(s) exceed |r|>0.8 out of "
                f"{n} candidate numeric features already narrowed down by feature selection. "
                "Multicollinearity is limited (collinear pairs were already pruned during "
                "feature selection), so PCA is not required: applying it here would sacrifice "
                "interpretability -- a key requirement for a managerial dashboard -- for "
                "negligible dimensionality benefit."
            ),
        }
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as _SS

    X = _SS().fit_transform(df[numeric_features].fillna(0))
    pca = PCA(random_state=RANDOM_STATE).fit(X)
    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_.round(4),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).round(4),
    })
    return {
        "recommended": True,
        "high_corr_pairs": high_corr_pairs,
        "explained_variance_table": explained,
        "explanation": (
            f"{len(high_corr_pairs)} numeric feature pairs exceed |r|>0.8. PCA is shown here as "
            "an exploratory diagnostic; the production model still uses the original, "
            "interpretable features so that risk drivers remain explainable to managers."
        ),
    }


# --------------------------------------------------------------------------
# 12. EDA (business-question-driven aggregations; charts are built in app.py)
# --------------------------------------------------------------------------
def eda_summary(df: pd.DataFrame) -> dict:
    out = {}
    out["overall_late_rate"] = float(df["late_delivery"].mean())
    out["n_orders"] = len(df)

    ts = df.copy()
    ts["purchase_month_period"] = ts["order_purchase_timestamp"].dt.to_period("M").astype(str)
    out["late_rate_by_month"] = (
        ts.groupby("purchase_month_period")["late_delivery"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "late_rate", "count": "n_orders"})
    )

    out["late_rate_by_customer_state"] = (
        df.groupby("customer_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "late_rate", "count": "n_orders"})
        .query("n_orders >= 20").sort_values("late_rate", ascending=False)
    )

    out["late_rate_by_seller_state"] = (
        df.groupby("seller_state")["late_delivery"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "late_rate", "count": "n_orders"})
        .query("n_orders >= 20").sort_values("late_rate", ascending=False)
    )

    out["late_rate_by_category"] = (
        df.groupby("dominant_category")["late_delivery"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "late_rate", "count": "n_orders"})
        .query("n_orders >= 30").sort_values("late_rate", ascending=False)
    )

    out["freight_vs_late"] = df.groupby("late_delivery")["total_freight"].mean().reset_index()
    out["order_value_vs_late"] = df.groupby("late_delivery")["total_price"].mean().reset_index()
    out["distance_vs_late"] = df.groupby("late_delivery")["distance_km"].mean().reset_index()
    out["processing_time_vs_late"] = (
        df.groupby("late_delivery")[["approval_time_hours", "estimated_window_days"]].mean().reset_index()
    )
    return out


def secondary_review_narrative(df_model: pd.DataFrame, reviews: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Descriptive-only: reviews are NEVER used as a model predictor, but a
    manager will want to know the downstream customer-experience consequence
    of a late delivery. This merge happens strictly for narrative purposes."""
    merged = df_model[["order_id", "late_delivery"]].merge(
        reviews[["order_id", "review_score"]].dropna(), on="order_id", how="inner"
    )
    if merged.empty:
        return None
    return merged.groupby("late_delivery")["review_score"].mean().reset_index()


# --------------------------------------------------------------------------
# 13. MODELLING
# --------------------------------------------------------------------------
def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def _select_best_k(X_train, y_train, preprocessor, k_grid=(5, 11, 21, 31)) -> int:
    """Small, fast hyper-parameter search. To keep runtime reasonable on
    Streamlit Cloud with ~100k rows, cross-validation for k-selection is
    performed on a stratified subsample; the final chosen k is then fit on
    the full training set."""
    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_train) > 15000:
        sample_idx, _ = train_test_split(
            np.arange(len(X_train)), train_size=15000, stratify=y_train, random_state=RANDOM_STATE
        )
        X_sub, y_sub = X_train.iloc[sample_idx], y_train[sample_idx]
    else:
        X_sub, y_sub = X_train, y_train

    best_k, best_score = k_grid[0], -1
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for k in k_grid:
        pipe = Pipeline([("prep", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1))])
        scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="recall", n_jobs=-1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_k = mean_score, k
    return best_k


@dataclass
class ModelResult:
    name: str
    pipeline: object
    metrics: dict
    confusion_mat: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


def evaluate_predictions(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate_models(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str],
                               target_col: str = "late_delivery") -> dict:
    X = df[numeric_features + categorical_features]
    y = df[target_col].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    best_k = _select_best_k(X_train, y_train, preprocessor)

    knn_pipe = Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features)),
        ("clf", KNeighborsClassifier(n_neighbors=best_k, weights="distance", n_jobs=-1)),
    ])
    knn_pipe.fit(X_train, y_train)
    knn_pred = knn_pipe.predict(X_test)
    knn_proba = knn_pipe.predict_proba(X_test)[:, 1]
    knn_metrics = evaluate_predictions(y_test, knn_pred, knn_proba)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_proba)

    nb_pipe = Pipeline([
        ("prep", build_preprocessor(numeric_features, categorical_features)),
        ("clf", GaussianNB()),
    ])
    nb_pipe.fit(X_train, y_train)
    nb_pred = nb_pipe.predict(X_test)
    nb_proba = nb_pipe.predict_proba(X_test)[:, 1]
    nb_metrics = evaluate_predictions(y_test, nb_pred, nb_proba)
    nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_proba)

    results = {
        "KNN": ModelResult("K-Nearest Neighbours", knn_pipe, knn_metrics,
                            confusion_matrix(y_test, knn_pred), knn_fpr, knn_tpr, knn_pred, knn_proba),
        "NaiveBayes": ModelResult("Naive Bayes (Gaussian)", nb_pipe, nb_metrics,
                                   confusion_matrix(y_test, nb_pred), nb_fpr, nb_tpr, nb_pred, nb_proba),
    }
    meta = {
        "best_k": best_k, "X_test": X_test, "y_test": y_test, "idx_test": idx_test,
        "X_train": X_train, "y_train": y_train,
        "class_balance": {"late_pct": float(np.mean(y)), "on_time_pct": float(1 - np.mean(y))},
    }
    return {"results": results, "meta": meta}


def compare_models(results: dict) -> tuple[pd.DataFrame, str, str]:
    comp = pd.DataFrame({
        name: {
            "Accuracy": r.metrics["accuracy"], "Precision": r.metrics["precision"],
            "Recall": r.metrics["recall"], "F1-score": r.metrics["f1"], "ROC-AUC": r.metrics["roc_auc"],
        }
        for name, r in results.items()
    }).T.round(4)

    # Because a missed late order is operationally costlier than an
    # unnecessary monitoring flag, model selection prioritises Recall on the
    # late-delivery class, with F1 as a tie-breaker against excessive false alarms.
    best_name = comp["Recall"].idxmax()
    if comp["Recall"].nunique() == 1:
        best_name = comp["F1-score"].idxmax()
    justification = (
        f"{results[best_name].name} is selected as the preferred model. Given that failing to "
        f"flag a genuinely late order (a false negative) costs Olist a missed intervention "
        f"opportunity -- while an unnecessary monitoring flag (a false positive) only costs a "
        f"small amount of operational attention -- Recall on the late-delivery class is the "
        f"primary selection criterion. {results[best_name].name} achieves a Recall of "
        f"{comp.loc[best_name, 'Recall']:.1%} versus "
        f"{comp.loc[[n for n in comp.index if n != best_name][0], 'Recall']:.1%} for the "
        f"alternative model, with an F1-score of {comp.loc[best_name, 'F1-score']:.3f}."
    )
    return comp, best_name, justification


def compute_feature_importance(pipeline, X_test, y_test, max_sample: int = 3000) -> pd.DataFrame:
    if len(X_test) > max_sample:
        X_sub = X_test.sample(max_sample, random_state=RANDOM_STATE)
        y_sub = pd.Series(y_test, index=X_test.index).loc[X_sub.index].values
    else:
        X_sub, y_sub = X_test, y_test
    result = permutation_importance(
        pipeline, X_sub, y_sub, scoring="recall", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = pd.DataFrame({
        "feature": X_sub.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp


# --------------------------------------------------------------------------
# 14. RISK PROFILING
# --------------------------------------------------------------------------
def risk_segmentation(probabilities: np.ndarray) -> tuple[np.ndarray, dict]:
    """Data-driven thresholds: the top ~15% of predicted probability is
    flagged High Risk (the volume an operations team can realistically
    intervene on daily), the next ~25% Medium Risk, the remainder Low Risk."""
    p85 = float(np.quantile(probabilities, 0.85))
    p60 = float(np.quantile(probabilities, 0.60))
    segments = np.where(probabilities >= p85, "High Risk", np.where(probabilities >= p60, "Medium Risk", "Low Risk"))
    thresholds = {"medium_threshold_p60": round(p60, 3), "high_threshold_p85": round(p85, 3)}
    return segments, thresholds


INTERVENTION_MAP = {
    "High Risk": "Immediate proactive outreach: escalate to seller/carrier, notify fulfilment ops for expedite review, pre-emptively inform customer of possible delay.",
    "Medium Risk": "Add to daily monitoring queue: flag for seller performance tracking and route/carrier review if delay signal persists.",
    "Low Risk": "Standard processing; no additional intervention required.",
}


def build_order_prediction_table(df: pd.DataFrame, idx_test, y_test, y_pred, y_proba, risk_segments) -> pd.DataFrame:
    base = df.loc[idx_test, [
        "order_id", "customer_state", "seller_state", "dominant_category",
        "total_price", "distance_km", "estimated_window_days",
    ]].copy()
    base["actual_late_delivery"] = y_test
    base["predicted_late_delivery"] = y_pred
    base["risk_probability"] = np.round(y_proba, 4)
    base["risk_segment"] = risk_segments
    base["recommended_intervention"] = base["risk_segment"].map(INTERVENTION_MAP)
    return base.sort_values("risk_probability", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 15. MANAGERIAL INSIGHTS + INTERVENTION FRAMEWORK
# --------------------------------------------------------------------------
def generate_managerial_insights(eda: dict, importance_df: pd.DataFrame, comparison_df: pd.DataFrame,
                                  best_model_name: str, thresholds: dict) -> list[str]:
    insights = []
    insights.append(
        f"Across {eda['n_orders']:,} delivered orders with complete records, the overall "
        f"late-delivery rate is {eda['overall_late_rate']:.1%}."
    )
    top_cust_state = eda["late_rate_by_customer_state"].iloc[0]
    insights.append(
        f"The customer state with the highest late-delivery rate (min. 20 orders) is "
        f"{top_cust_state['customer_state']} at {top_cust_state['late_rate']:.1%}, "
        f"based on {int(top_cust_state['n_orders'])} orders."
    )
    top_seller_state = eda["late_rate_by_seller_state"].iloc[0]
    insights.append(
        f"The seller state most associated with late delivery is {top_seller_state['seller_state']} "
        f"at {top_seller_state['late_rate']:.1%} (n={int(top_seller_state['n_orders'])})."
    )
    if not eda["late_rate_by_category"].empty:
        top_cat = eda["late_rate_by_category"].iloc[0]
        insights.append(
            f"Among product categories with at least 30 orders, '{top_cat['dominant_category']}' "
            f"shows the highest late-delivery rate at {top_cat['late_rate']:.1%}."
        )
    fv = eda["freight_vs_late"]
    if len(fv) == 2:
        delta = fv.loc[fv['late_delivery'] == 1, 'total_freight'].values[0] - fv.loc[fv['late_delivery'] == 0, 'total_freight'].values[0]
        direction = "higher" if delta > 0 else "lower"
        insights.append(
            f"Late orders carry, on average, {direction} freight cost than on-time orders "
            f"(difference of R$ {abs(delta):.2f} per order), consistent with longer/heavier shipments "
            f"being more exposed to delay."
        )
    dv = eda["distance_vs_late"]
    if len(dv) == 2:
        delta_d = dv.loc[dv['late_delivery'] == 1, 'distance_km'].values[0] - dv.loc[dv['late_delivery'] == 0, 'distance_km'].values[0]
        insights.append(
            f"Late orders travel on average {delta_d:.0f} km {'further' if delta_d > 0 else 'less'} "
            f"between seller and customer than on-time orders, supporting a geographic risk component."
        )
    if not importance_df.empty:
        top_feat = importance_df.iloc[0]
        insights.append(
            f"The strongest predictive driver identified via permutation importance on the selected "
            f"model is '{top_feat['feature']}'."
        )
    insights.append(
        f"{best_model_name} is the recommended model for operational deployment "
        f"(Recall={comparison_df.loc[best_model_name, 'Recall']:.1%}, "
        f"F1={comparison_df.loc[best_model_name, 'F1-score']:.3f})."
    )
    insights.append(
        f"Orders scoring at or above a predicted probability of {thresholds['high_threshold_p85']} "
        f"are classified High Risk (top ~15% of the risk distribution) and are the recommended "
        f"focus for daily proactive intervention given realistic operational capacity."
    )
    return insights


def generate_intervention_framework(eda: dict, importance_df: pd.DataFrame) -> pd.DataFrame:
    top_cust_state = eda["late_rate_by_customer_state"].iloc[0]["customer_state"]
    top_seller_state = eda["late_rate_by_seller_state"].iloc[0]["seller_state"]
    top_category = eda["late_rate_by_category"].iloc[0]["dominant_category"] if not eda["late_rate_by_category"].empty else "N/A"
    top_driver = importance_df.iloc[0]["feature"] if not importance_df.empty else "N/A"

    rows = [
        {
            "risk_condition": f"Orders shipped to customer state {top_cust_state} (elevated observed late-rate)",
            "managerial_issue": "Regional delivery capacity or last-mile logistics constraint",
            "recommended_intervention": "Regional capacity review with logistics partners; consider adding buffer days to the delivery promise for this region",
            "priority": "High", "responsible_function": "Logistics Coordination",
        },
        {
            "risk_condition": f"Orders dispatched from seller state {top_seller_state} (elevated observed late-rate)",
            "managerial_issue": "Seller-side dispatch delay or carrier pickup inefficiency",
            "recommended_intervention": "Seller performance review and escalation; monitor dispatch SLAs for sellers based in this state",
            "priority": "High", "responsible_function": "Seller Management",
        },
        {
            "risk_condition": f"Orders in product category '{top_category}' (elevated observed late-rate)",
            "managerial_issue": "Category-specific fulfilment complexity (e.g. size, fragility, specialised handling)",
            "recommended_intervention": "Category management review of packaging/handling requirements and carrier suitability",
            "priority": "Medium", "responsible_function": "Category Management",
        },
        {
            "risk_condition": "High predicted risk probability (top ~15% of scored orders)",
            "managerial_issue": "Orders combining several risk drivers simultaneously",
            "recommended_intervention": "Immediate proactive dispatch monitoring and pre-emptive customer communication",
            "priority": "High", "responsible_function": "Fulfilment Operations",
        },
        {
            "risk_condition": f"Orders with an elevated value of the top model driver ('{top_driver}')",
            "managerial_issue": "The single strongest statistical predictor of delay in the current data",
            "recommended_intervention": "Incorporate this variable into a simple daily operational watch-list alongside the model score",
            "priority": "Medium", "responsible_function": "Fulfilment Operations",
        },
        {
            "risk_condition": "Multi-seller orders (seller_count > 1)",
            "managerial_issue": "Coordination overhead across multiple fulfilment parties for a single order",
            "recommended_intervention": "Prioritise single-seller fulfilment where feasible; add coordination checkpoints for multi-seller orders",
            "priority": "Medium", "responsible_function": "Fulfilment Operations",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 16. MASTER RESULT OBJECT + ORCHESTRATOR
# --------------------------------------------------------------------------
@dataclass
class PipelineResults:
    data_dir: str
    cleaning_logs: dict
    integration_log: dict
    df_model: pd.DataFrame
    df_excluded: pd.DataFrame
    target_log: dict
    missing_summary: pd.DataFrame
    missing_treatment_log: dict
    outlier_summary: pd.DataFrame
    outlier_treatment_log: dict
    numeric_features: list
    categorical_features: list
    selection_report: pd.DataFrame
    dim_reduction: dict
    eda: dict
    review_narrative: Optional[pd.DataFrame]
    model_results: dict
    model_meta: dict
    comparison_df: pd.DataFrame
    best_model_name: str
    best_model_justification: str
    feature_importance: pd.DataFrame
    risk_thresholds: dict
    order_prediction_table: pd.DataFrame
    managerial_insights: list
    intervention_framework: pd.DataFrame


@_cache
def run_pipeline(data_dir: str) -> PipelineResults:
    """The single orchestrator. app.py should call ONLY this function (plus
    check_data_availability) and render the returned PipelineResults."""

    integrated, integration_logs = integrate_full_dataset(data_dir)
    cleaning_logs = {k: v for k, v in integration_logs.items() if k != "integration"}
    integration_log = integration_logs["integration"]

    df_model, df_excluded, target_log = compute_target(integrated)

    missing_candidate_cols = CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL
    missing_candidate_cols = [c for c in missing_candidate_cols if c in df_model.columns]
    missing_summary = missing_value_analysis(df_model, missing_candidate_cols)

    numeric_present = [c for c in CANDIDATE_NUMERIC if c in df_model.columns]
    categorical_present = [c for c in CANDIDATE_CATEGORICAL if c in df_model.columns]
    df_model, missing_treatment_log = handle_missing_values(df_model, numeric_present, categorical_present)

    # Engineer features AFTER geo/aggregation but note some engineered
    # features (approval_time_hours, estimated_window_days, freight_ratio,
    # same_state_flag, region) depend on raw timestamp/state columns that
    # must exist prior to imputation of the *aggregate* numeric columns --
    # so we engineer first, then run missing-value handling again for any
    # newly created columns, then outlier-treat.
    df_model = engineer_features(df_model)
    df_model, missing_treatment_log_2 = handle_missing_values(
        df_model, ["approval_time_hours", "estimated_window_days", "freight_ratio"], []
    )
    missing_treatment_log.update(missing_treatment_log_2)

    outlier_cols = [c for c in CANDIDATE_NUMERIC if c in df_model.columns]
    outlier_summary = outlier_analysis(df_model, outlier_cols)
    df_model, outlier_treatment_log = treat_outliers(df_model, outlier_cols, outlier_summary)

    numeric_features, categorical_features, selection_report = select_features(df_model)
    dim_reduction = assess_dimensionality_reduction(df_model, numeric_features)

    eda = eda_summary(df_model)
    _, cleaned_raw_logs = run_all_cleaning(data_dir)
    review_narrative = None
    try:
        raw = load_raw_tables(data_dir)
        reviews_clean, _ = clean_reviews(raw["reviews"])
        review_narrative = secondary_review_narrative(df_model, reviews_clean)
    except Exception:
        review_narrative = None

    training_output = train_and_evaluate_models(df_model, numeric_features, categorical_features)
    model_results, model_meta = training_output["results"], training_output["meta"]

    comparison_df, best_model_name, justification = compare_models(model_results)
    best_pipeline = model_results[best_model_name].pipeline
    feature_importance = compute_feature_importance(best_pipeline, model_meta["X_test"], model_meta["y_test"])

    best_proba = model_results[best_model_name].y_proba
    risk_segments, risk_thresholds = risk_segmentation(best_proba)
    order_prediction_table = build_order_prediction_table(
        df_model, model_meta["idx_test"], model_meta["y_test"],
        model_results[best_model_name].y_pred, best_proba, risk_segments,
    )

    managerial_insights = generate_managerial_insights(eda, feature_importance, comparison_df, best_model_name, risk_thresholds)
    intervention_framework = generate_intervention_framework(eda, feature_importance)

    return PipelineResults(
        data_dir=data_dir,
        cleaning_logs=cleaning_logs,
        integration_log=integration_log,
        df_model=df_model,
        df_excluded=df_excluded,
        target_log=target_log,
        missing_summary=missing_summary,
        missing_treatment_log=missing_treatment_log,
        outlier_summary=outlier_summary,
        outlier_treatment_log=outlier_treatment_log,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        selection_report=selection_report,
        dim_reduction=dim_reduction,
        eda=eda,
        review_narrative=review_narrative,
        model_results=model_results,
        model_meta=model_meta,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_model_justification=justification,
        feature_importance=feature_importance,
        risk_thresholds=risk_thresholds,
        order_prediction_table=order_prediction_table,
        managerial_insights=managerial_insights,
        intervention_framework=intervention_framework,
    )
