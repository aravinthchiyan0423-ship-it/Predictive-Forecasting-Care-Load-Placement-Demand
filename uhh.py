# app.py
# -----------------------------------------------------------------------------
# UAC Program — Predictive Capacity Dashboard
#
# IMPORTANT: Run data_prep.py FIRST. This app reads "uac_data_processed.csv",
# which data_prep.py generates from your raw HHS export.
#
# Run with:  streamlit run app.py
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="UAC Care Load Forecast", layout="wide")
st.title("UAC Program \u2014 Predictive Capacity Dashboard")

DATA_PATH = "uac_data_processed.csv"
TARGET_COL = "Children in HHS Care"
FEATURE_COLS = [
    "Children in HHS Care_lag1", "Children in HHS Care_lag7",
    "CareLoad_roll7_mean", "CareLoad_roll7_std",
    "day_of_week", "month", "is_weekend", "Net_Pressure",
]

# ----------------------------------------------------------------------------
# 1. LOAD DATA  (cached so it only re-reads the CSV when the file changes)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.index.name = "Date"
    return data

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find '{DATA_PATH}' in this folder.\n\n"
        f"Run **data_prep.py** first (it cleans the raw CSV and saves "
        f"'{DATA_PATH}'), then restart this app. Or upload the processed "
        f"file below as a one-off workaround."
    )
    uploaded = st.file_uploader("Upload uac_data_processed.csv", type="csv")
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
    df.index.name = "Date"

required_cols = FEATURE_COLS + [TARGET_COL, "Net_Pressure", "Children discharged from HHS Care"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(
        f"The processed CSV is missing required columns: {missing_cols}.\n\n"
        f"Re-run data_prep.py to regenerate '{DATA_PATH}' with all feature columns."
    )
    st.stop()

# Rows with NaN (first ~14 days, due to lag/rolling windows) can't be used
# for model training, but we keep the FULL df for the historical trend charts.
df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

if len(df_model) < 30:
    st.error("Not enough clean rows to train models after removing NaNs. Check data_prep.py output.")
    st.stop()

# ----------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 30, 14)
model_choice = st.sidebar.selectbox("Forecast Model", ["SARIMA", "Random Forest", "Gradient Boosting"])
st.sidebar.caption(f"Data range: {df.index.min().date()} \u2192 {df.index.max().date()} ({len(df)} days)")

# ----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------------
split_point = int(len(df_model) * 0.8)
train = df_model.iloc[:split_point]
test = df_model.iloc[split_point:]

X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

# ----------------------------------------------------------------------------
# 4. TRAIN MODELS  (cached — slider/dropdown changes won't retrain everything)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_models(train_target_series, X_train_df, y_train_series):
    sarima_fit = SARIMAX(
        train_target_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)
    ).fit(disp=False)
    es_fit = ExponentialSmoothing(
        train_target_series, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_df, y_train_series)
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(
        X_train_df, y_train_series
    )
    return sarima_fit, es_fit, rf_model, gb_model

with st.spinner("Training models..."):
    sarima_fit, es_fit, rf, gb = train_models(train[TARGET_COL], X_train, y_train)

# ----------------------------------------------------------------------------
# 5. EVALUATE ON TEST SET
# ----------------------------------------------------------------------------
naive_pred = test[TARGET_COL].shift(1).bfill()
sarima_test_pred = sarima_fit.forecast(steps=len(test))
es_test_pred = es_fit.forecast(len(test))
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}


results_df = pd.DataFrame([
    evaluate(y_test, naive_pred, "Naive"),
    evaluate(y_test, sarima_test_pred, "SARIMA"),
    evaluate(y_test, es_test_pred, "Exponential Smoothing"),
    evaluate(y_test, rf_pred, "Random Forest"),
    evaluate(y_test, gb_pred, "Gradient Boosting"),
]).sort_values("RMSE").reset_index(drop=True)

best_model_row = results_df.iloc[0]
selected_row = results_df[results_df["model"] == model_choice].iloc[0]

# ----------------------------------------------------------------------------
# 6. CAPACITY / RISK THRESHOLDS
# ----------------------------------------------------------------------------
capacity_threshold = df[TARGET_COL].quantile(0.95)
pressure_threshold = df["Net_Pressure"].quantile(0.90)

breach_count = int(np.sum(rf_pred > capacity_threshold))
capacity_breach_probability = (breach_count / len(rf_pred)) * 100

actual_breach_dates = test.index[test[TARGET_COL] > capacity_threshold]
warning_dates = test.index[test["Net_Pressure"] > pressure_threshold]

if len(actual_breach_dates) == 0:
    surge_lead_time = "No breach in test period"
elif len(warning_dates) == 0:
    surge_lead_time = "No early warning signal found"
else:
    first_breach = actual_breach_dates[0]
    prior_warnings = warning_dates[warning_dates < first_breach]
    surge_lead_time = (
        f"{(first_breach - prior_warnings[0]).days} days"
        if len(prior_warnings) > 0
        else "Warning found, but after breach"
    )

# ----------------------------------------------------------------------------
# 7. KPI ROW
# ----------------------------------------------------------------------------
st.subheader("Key Performance Indicators")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Children in HHS Care (Latest)", f"{df[TARGET_COL].iloc[-1]:,.0f}")
k2.metric("Best Model", best_model_row["model"], f"RMSE {best_model_row['RMSE']:.2f}")
k3.metric(f"Forecast Accuracy ({model_choice})", f"{100 - selected_row['MAPE']:.1f}%")
k4.metric("Capacity Threshold (95th pct)", f"{capacity_threshold:,.0f}")
k5.metric("Capacity Breach Probability", f"{capacity_breach_probability:.1f}%")
k6.metric("Surge Lead Time", surge_lead_time)

st.divider()


# ----------------------------------------------------------------------------
# 8. RECURSIVE MULTI-STEP FORECAST FOR TREE MODELS (RF / GB)
#    SARIMA can forecast natively; tree models need lag features that don't
#    exist yet for future dates, so we predict one day at a time and feed
#    each prediction back in as the next day's lag.
# ----------------------------------------------------------------------------
def recursive_tree_forecast(model, history_df, n_steps, feature_cols, target_col, with_ci=False):
    history = history_df[[target_col, "Net_Pressure"]].copy()
    preds, lowers, uppers = [], [], []

    for _ in range(n_steps):
        next_date = history.index[-1] + pd.Timedelta(days=1)
        lag1 = history[target_col].iloc[-1]
        lag7 = history[target_col].iloc[-7] if len(history) >= 7 else history[target_col].iloc[0]
        roll7 = history[target_col].iloc[-7:]
        net_pressure_avg = history["Net_Pressure"].iloc[-7:].mean()

        row = pd.DataFrame([{
            "Children in HHS Care_lag1": lag1,
            "Children in HHS Care_lag7": lag7,
            "CareLoad_roll7_mean": roll7.mean(),
            "CareLoad_roll7_std": roll7.std() if len(roll7) > 1 else 0.0,
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "is_weekend": int(next_date.dayofweek in [5, 6]),
            "Net_Pressure": net_pressure_avg,
        }])[feature_cols]

        pred = model.predict(row)[0]
        preds.append(pred)

        if with_ci and hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(row.values)[0] for est in model.estimators_])
            lowers.append(np.percentile(tree_preds, 2.5))
            uppers.append(np.percentile(tree_preds, 97.5))

        history.loc[next_date] = [pred, net_pressure_avg]

    dates = pd.date_range(start=history_df.index[-1] + pd.Timedelta(days=1), periods=n_steps, freq="D")
    forecast_series = pd.Series(preds, index=dates)
    if with_ci:
        return forecast_series, pd.Series(lowers, index=dates), pd.Series(uppers, index=dates)
    return forecast_series


# ----------------------------------------------------------------------------
# 9. FORECAST CHART (switches based on sidebar model selection)
# ----------------------------------------------------------------------------
st.subheader(f"{horizon}-Day Forecast \u2014 {model_choice}")

if model_choice == "SARIMA":
    forecast_obj = sarima_fit.get_forecast(steps=horizon)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_values = pd.Series(forecast_obj.predicted_mean.values, index=forecast_dates)
    conf_int = forecast_obj.conf_int(alpha=0.05)
    lower_bound = pd.Series(conf_int.iloc[:, 0].values, index=forecast_dates)
    upper_bound = pd.Series(conf_int.iloc[:, 1].values, index=forecast_dates)

elif model_choice == "Random Forest":
    forecast_values, lower_bound, upper_bound = recursive_tree_forecast(
        rf, df_model, horizon, FEATURE_COLS, TARGET_COL, with_ci=True
    )

else:  # Gradient Boosting — no per-tree percentile available, use residual-based band
    forecast_values = recursive_tree_forecast(gb, df_model, horizon, FEATURE_COLS, TARGET_COL, with_ci=False)
    residual_std = float(np.std(y_test.values - gb_pred))
    lower_bound = forecast_values - 1.96 * residual_std
    upper_bound = forecast_values + 1.96 * residual_std

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[TARGET_COL], name="Actual", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, name="Forecast",
                          line=dict(color="orange", dash="dash")))
fig.add_trace(go.Scatter(x=upper_bound.index, y=upper_bound, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=lower_bound.index, y=lower_bound, fill="tonexty",
                          fillcolor="rgba(255,165,0,0.2)", line=dict(width=0), name="95% Confidence Interval"))
fig.update_layout(xaxis_title="Date", yaxis_title="Children in HHS Care", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.divider()

# ----------------------------------------------------------------------------
# 10. MODEL COMPARISON TABLE
# ----------------------------------------------------------------------------
st.subheader("Model Comparison")
st.dataframe(
    results_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"}),
    width="stretch",
)
st.write(f"Selected model: **{model_choice}**  |  MAE: **{selected_row['MAE']:.2f}**  "
         f"|  RMSE: **{selected_row['RMSE']:.2f}**  |  MAPE: **{selected_row['MAPE']:.2f}%**")

st.divider()

# ----------------------------------------------------------------------------
# 11. HISTORICAL TRENDS
# ----------------------------------------------------------------------------
st.subheader("Historical Trends")
col1, col2 = st.columns(2)

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df[TARGET_COL], name="Children in HHS Care",
                               line=dict(color="#1f77b4")))
    fig2.update_layout(title="HHS Care Load Over Time")
    st.plotly_chart(fig2, width="stretch")

with col2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df["Children discharged from HHS Care"], name="Daily Discharges",
                               line=dict(color="#2ca02c")))
    fig3.update_layout(title="Daily Discharges Over Time")
    st.plotly_chart(fig3, width="stretch")

# ----------------------------------------------------------------------------
# 12. DATASET PREVIEW
# ----------------------------------------------------------------------------
st.subheader("Dataset Preview (most recent 20 days)")
st.dataframe(df.tail(20), width="stretch")
