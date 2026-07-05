"""
HDPE Reactor Melt Flow Index (MFI) Prediction
Streamlit application - manual entry and CSV batch prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="HDPE Reactor MFI Prediction", page_icon="🧪", layout="wide")

# ----------------------------------------------------------------------
# Load model, scaler, feature list (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/mfi_model.pkl")
    scaler = joblib.load("model/mfi_scaler.pkl")
    feature_list = joblib.load("model/mfi_feature_list.pkl")
    return model, scaler, feature_list

try:
    model, scaler, FEATURES = load_artifacts()
    MODEL_NAME = type(model).__name__
except Exception as e:
    st.error(f"Could not load the model files. Make sure model/mfi_model.pkl, "
              f"model/mfi_scaler.pkl and model/mfi_feature_list.pkl exist. Error: {e}")
    st.stop()

# The 10 raw process variables the model was trained on (before rolling averages)
RAW_FEATURES = ['R1 Ethylene flow', 'R1 H2/C2', 'R1 C4/C2', 'R1 Temp', 'R1 Pres', 'R1 Level',
                'R1 catalyst Feed rate', 'R1 Co-catalyst feed rate', 'R1 Hexane flow', 'R1 Mother Liquour flow']

# Typical operating ranges, used for slider bounds and input validation
# (from the training data - update these if your process window changes)
FEATURE_RANGES = {
    'R1 Ethylene flow':        (11000.0, 25000.0, 20000.0),
    'R1 H2/C2':                (1.0, 9.0, 4.8),
    'R1 C4/C2':                (0.0, 0.015, 0.004),
    'R1 Temp':                 (80.0, 84.0, 82.7),
    'R1 Pres':                 (6.0, 10.0, 9.0),
    'R1 Level':                (55.0, 70.0, 63.7),
    'R1 catalyst Feed rate':   (30.0, 200.0, 105.0),
    'R1 Co-catalyst feed rate':(9.0, 130.0, 70.0),
    'R1 Hexane flow':          (14000.0, 36000.0, 27500.0),
    'R1 Mother Liquour flow':  (800.0, 18000.0, 7200.0),
}
MFI_RANGE = (40.0, 100.0, 65.5)

# Approximate test-set error of the saved model - shown as a confidence message.
# Update this if the model is retrained (see the comparison notebook's Test MAE for the final model).
MODEL_TEST_MAE = 4.33
MODEL_TEST_R2 = 0.64


def build_feature_row(raw_values: dict, mfi_last_known: float) -> pd.DataFrame:
    """Build one row of the 31 model features from raw inputs.
    No history is available for a single manual entry, so the rolling
    2h/4h averages are set equal to the current reading (assumes the process
    has been steady recently). Use the CSV upload path for real rolling history."""
    row = {}
    for f in RAW_FEATURES:
        row[f] = raw_values[f]
        row[f + "_roll2h"] = raw_values[f]
        row[f + "_roll4h"] = raw_values[f]
    row["MFI_last_known"] = mfi_last_known
    return pd.DataFrame([row])[FEATURES]


def predict(X: pd.DataFrame) -> np.ndarray:
    X_scaled = pd.DataFrame(scaler.transform(X[FEATURES]), columns=FEATURES)
    return model.predict(X_scaled)


def confidence_message(pred_value: float) -> str:
    lo, hi = pred_value - MODEL_TEST_MAE, pred_value + MODEL_TEST_MAE
    return (f"Expected range: **{lo:.1f} - {hi:.1f}** "
            f"(based on the model's historical test error of ±{MODEL_TEST_MAE:.1f} MFI units, "
            f"Test R² = {MODEL_TEST_R2:.2f})")


def mfi_gauge(pred_value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_value,
        number={'suffix': " MFI"},
        gauge={
            'axis': {'range': [30, 110]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [30, 40], 'color': "#f4cccc"},
                {'range': [40, 100], 'color': "#d9ead3"},
                {'range': [100, 110], 'color': "#f4cccc"},
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.8, 'value': pred_value}
        },
        title={'text': "Predicted MFI at 1.2 kg"}
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🧪 HDPE Reactor MFI")
    st.markdown("### Project description")
    st.write(
        "Predicts the reactor Melt Flow Index (MFI at 1.2 kg) from live reactor "
        "process parameters, so operators can adjust conditions before an "
        "off-spec batch happens."
    )
    st.markdown("### Model information")
    st.write(f"**Model:** {MODEL_NAME}")
    st.write(f"**Test MAE:** {MODEL_TEST_MAE:.2f} MFI units")
    st.write(f"**Test R²:** {MODEL_TEST_R2:.2f}")
    st.write(f"**Features used:** {len(FEATURES)}")
    st.caption("Trained on grades with more than 100 lab samples, hourly reactor data, 2023-2026.")
    st.markdown("### Instructions")
    st.write(
        "1. Choose **Manual Entry** for a single prediction, or **CSV Upload** for a batch.\n"
        "2. Manual entry needs the current reactor readings plus the last confirmed lab MFI for this grade.\n"
        "3. CSV upload needs an hourly time series (Timestamp, GRADE, the 10 process columns, and an "
        "optional MFI column) - rolling averages and last-known-MFI are computed automatically."
    )
    st.markdown("---")
    st.caption("Not a substitute for lab testing or operator judgement, especially in the first few "
                "hours after a grade change, where the model has the least information.")

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("HDPE Reactor Melt Flow Index Prediction")

tab_manual, tab_csv = st.tabs(["📝 Manual Entry", "📁 CSV Upload"])

# ---------------- Manual entry ----------------
with tab_manual:
    st.subheader("Enter current reactor parameters")

    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    col1, col2 = st.columns(2)
    raw_values = {}
    feature_items = list(FEATURE_RANGES.items())
    half = len(feature_items) // 2 + 1

    with col1:
        for name, (lo, hi, default) in feature_items[:half]:
            raw_values[name] = st.number_input(
                name, min_value=float(lo), max_value=float(hi), value=float(default),
                key=f"{name}_{st.session_state.reset_counter}"
            )
    with col2:
        for name, (lo, hi, default) in feature_items[half:]:
            raw_values[name] = st.number_input(
                name, min_value=float(lo), max_value=float(hi), value=float(default),
                key=f"{name}_{st.session_state.reset_counter}"
            )
        mfi_last_known = st.number_input(
            "Last known lab MFI for this grade",
            min_value=float(MFI_RANGE[0]), max_value=float(MFI_RANGE[1]), value=float(MFI_RANGE[2]),
            help="The most recent confirmed lab MFI reading for the grade currently running. "
                 "If unknown (e.g. right after a grade change), use the grade's typical average.",
            key=f"mfi_last_known_{st.session_state.reset_counter}"
        )

    col_a, col_b = st.columns([1, 1])
    predict_clicked = col_a.button("🔮 Predict MFI", type="primary")
    reset_clicked = col_b.button("🔄 Reset inputs")

    if reset_clicked:
        st.session_state.reset_counter += 1
        st.rerun()

    if predict_clicked:
        try:
            X_row = build_feature_row(raw_values, mfi_last_known)
            pred = predict(X_row)[0]

            st.markdown("### Prediction result")
            gcol, tcol = st.columns([1, 1])
            with gcol:
                st.plotly_chart(mfi_gauge(pred), use_container_width=True)
            with tcol:
                st.metric("Predicted MFI at 1.2 kg", f"{pred:.2f}")
                st.info(confidence_message(pred))
                st.write(f"**Model used:** {MODEL_NAME}")
                st.write(f"**Prediction timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            st.markdown("### Entered values")
            summary = raw_values.copy()
            summary["Last known lab MFI"] = mfi_last_known
            st.dataframe(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]), use_container_width=True)

            result_row = pd.DataFrame([{**raw_values, "Last known lab MFI": mfi_last_known,
                                         "Predicted MFI": pred,
                                         "Timestamp": datetime.datetime.now()}])
            st.download_button(
                "⬇️ Download this prediction as CSV",
                result_row.to_csv(index=False).encode("utf-8"),
                file_name="mfi_prediction.csv", mime="text/csv"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- CSV upload ----------------
with tab_csv:
    st.subheader("Upload an hourly reactor data CSV")
    st.write(
        "Required columns: `Timestamp`, `GRADE`, and the 10 raw process columns "
        "(see the sample CSV). An `MFI` column is optional - if present, it's used "
        "to compute the last-known-MFI feature and rolling averages automatically."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
            missing_cols = [c for c in RAW_FEATURES + ["Timestamp", "GRADE"] if c not in raw_df.columns]
            if missing_cols:
                st.error(f"Missing required column(s): {missing_cols}")
            else:
                raw_df = raw_df.sort_values("Timestamp").reset_index(drop=True)

                for c in RAW_FEATURES:
                    raw_df[f"{c}_roll2h"] = raw_df[c].rolling(window=2, min_periods=1).mean()
                    raw_df[f"{c}_roll4h"] = raw_df[c].rolling(window=4, min_periods=1).mean()

                if "MFI" in raw_df.columns:
                    raw_df["MFI_lag1"] = raw_df["MFI"].shift(1)
                    grade_avg = raw_df.groupby("GRADE")["MFI"].transform("mean")
                    same_grade = raw_df["GRADE"] == raw_df["GRADE"].shift(1)
                    raw_df["MFI_last_known"] = np.where(same_grade, raw_df["MFI_lag1"], grade_avg)
                    raw_df["MFI_last_known"] = raw_df["MFI_last_known"].fillna(MFI_RANGE[2])
                else:
                    st.warning("No `MFI` column found - using the default typical MFI "
                               f"({MFI_RANGE[2]}) as the last-known-MFI value for every row. "
                               "Predictions will be less accurate than with real history.")
                    raw_df["MFI_last_known"] = MFI_RANGE[2]

                X_batch = raw_df[FEATURES]
                raw_df["Predicted MFI"] = predict(X_batch)
                raw_df["Prediction timestamp"] = datetime.datetime.now()

                st.success(f"Predicted MFI for {len(raw_df)} rows.")
                st.dataframe(raw_df[["Timestamp", "GRADE", "Predicted MFI"]], use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw_df["Timestamp"], y=raw_df["Predicted MFI"], name="Predicted MFI"))
                if "MFI" in raw_df.columns:
                    fig.add_trace(go.Scatter(x=raw_df["Timestamp"], y=raw_df["MFI"], name="Actual MFI"))
                fig.update_layout(title="Predicted MFI over time", height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download predictions as CSV",
                    raw_df.to_csv(index=False).encode("utf-8"),
                    file_name="mfi_batch_predictions.csv", mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not process the file: {e}")

st.markdown("---")
st.caption("HDPE Reactor MFI Prediction | Internal process monitoring tool | Not a substitute for lab testing.")
