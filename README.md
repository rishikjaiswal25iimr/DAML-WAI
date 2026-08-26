# E-Commerce Delivery Risk Intelligence: Predicting Late Orders Using Machine Learning

**Course:** Data Analytics and Machine Learning Techniques (DAML) — Working-with-AI (WAI) Assignment, IIM Ranchi
**Dataset:** Olist Brazilian E-Commerce Public Dataset (public, anonymised, ~100k orders, 2016–2018)

## Purpose

Can Olist flag orders that are at high risk of late delivery early enough for
proactive intervention? This project builds an order-level analytical
dataset from Olist's nine relational CSVs, engineers leakage-free features,
compares **K-Nearest Neighbours** and **Naive Bayes** classifiers, and
translates the output into a risk-scoring and intervention framework
presented through an interactive Streamlit dashboard.

The project treats this strictly as a **public-data academic case analysis**
— it does not imply access to Olist's confidential/internal systems.

## Files

| File | Role |
|---|---|
| `core.py` | Single analytical engine — data loading, cleaning, integration, missing-value/outlier treatment, feature engineering & selection, EDA aggregation, KNN + Naive Bayes modelling, evaluation, risk profiling, managerial insights, and intervention framework. **This is the only place analytical logic lives.** |
| `app.py` | Streamlit dashboard. Imports and calls `core.py`; never re-implements analysis. Renders a 7-tab managerial dashboard with filters, charts, tables and CSV downloads. |
| `requirements.txt` | Minimal, Streamlit-Cloud-compatible dependency list. |

## Required data files (place in the SAME folder as `core.py` / `app.py`)

```
core.py
app.py
requirements.txt
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

No subfolder is required or supported — `core.discover_data_directory()`
automatically looks in the script's own folder and the current working
directory, so the same code runs unmodified on a laptop, in a Colab/Jupyter
shell, or on Streamlit Community Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load runs the full pipeline (integration → cleaning → EDA →
modelling) once and caches it (`st.cache_data`); moving dashboard filters
afterwards does **not** retrain the models — it only re-slices the already
computed dataframes.

## Deploy on Streamlit Community Cloud

1. Push `core.py`, `app.py`, `requirements.txt`, and all nine CSVs to a public
   GitHub repository (flat structure, no subfolders — as above).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at that repo and `app.py` as the entry point.
3. No secrets, environment variables, or local paths are required — the app
   is fully self-discovering.

## Key modelling decisions (documented in code + Tab 7 of the dashboard)

- **Unit of analysis:** one row = one delivered order.
- **Target:** `late_delivery` = 1 if `order_delivered_customer_date >
  order_estimated_delivery_date`, computed only for orders with status
  `delivered` and complete actual/estimated dates. Non-qualifying orders are
  excluded from modelling and the exclusion is explicitly reported, not
  silently dropped.
- **Leakage policy:** `order_delivered_customer_date`,
  `order_delivered_carrier_date`, and all review fields are never used as
  predictors — a model that "knows" the parcel already left the carrier, or
  what the customer wrote afterwards, is not an *early*-warning system.
- **Missing values:** explicit missing-value analysis before treatment;
  median imputation for numeric fields, `"Unknown"` category for categorical
  fields — never blanket zero-filling.
- **Outliers:** IQR-based detection, then winsorized (capped at 3×IQR)
  rather than deleted, since extreme order values/weights are often
  legitimate business events, not data errors.
- **Feature selection:** correlation + mutual information for numeric
  features (with collinearity pruning at |r| > 0.9), mutual information for
  categorical features. A dimensionality-reduction (PCA) check is run but
  only recommended if genuine multicollinearity remains after selection.
- **Models:** KNN (k chosen via a small cross-validated recall search) and
  Gaussian Naive Bayes, both inside `sklearn` `Pipeline`/`ColumnTransformer`
  objects so preprocessing is fit only on the training fold.
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC and confusion
  matrices for both models. The preferred model is selected primarily on
  **Recall for the late-delivery class**, since a missed high-risk order is
  operationally costlier than an unnecessary monitoring flag.
- **Risk segmentation:** data-driven — top ~15% of predicted probability =
  High Risk, next ~25% = Medium Risk, remainder = Low Risk (not an arbitrary
  fixed cut-off).

## Known assumptions

- Payment records are assumed finalised at/near checkout and are therefore
  treated as an early, non-leaky signal.
- Geolocation is resolved at zip-code-prefix granularity (mean lat/lng per
  prefix), so seller–customer distance is approximate, not exact-address.
- Order-level aggregation from item-level records (sum of price, mean of
  weight, dominant seller/category by mode) is a defensible simplification
  of multi-item, multi-seller orders.

## Limitations (also shown in the app's Managerial Action Center tab)

- Historical (2016–2018), public, anonymised data — not live operational
  data, and not necessarily representative of Olist's current logistics
  network or seller base.
- No weather, holiday-calendar, or traffic data is available to explain some
  delays.
- The model provides **probabilistic risk scores that support, not replace,
  managerial judgment**, and predicts association, not proven causation.

## AI-assisted workflow note (for WAI documentation)

This codebase was produced with AI assistance (Claude) as a development and
reasoning aid, working strictly from the dataset's column/schema information
(not from the full raw data) as specified in the accompanying prompt. All
computed statistics, model metrics, and managerial insights shown in the
dashboard are generated at runtime from the actual CSV data — none are
hard-coded or fabricated.
