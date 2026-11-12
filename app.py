"""
Smart Crop Yield Prediction – Streamlit App
-------------------------------------------
Keep this file in the same folder as:
    - dtr.pkl
    - preprocessor.pkl

Install dependencies:
    pip install streamlit numpy pandas scikit-learn plotly

Run:
    streamlit run app.py
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Smart Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 85% 4%, rgba(95, 170, 111, .13), transparent 25%),
            linear-gradient(180deg, #f5faf5 0%, #f7f9f6 48%, #f3f7f2 100%);
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
    .hero {
        background: linear-gradient(115deg, #123d2b 0%, #1d6844 58%, #398653 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 16px 38px rgba(20, 75, 47, .16);
        margin-bottom: 24px;
    }
    .hero h1 {
        color: white; font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12;
        margin: 8px 0 10px; font-weight: 800;
    }
    .hero p { color: #e2f3e7; font-size: 1.02rem; margin: 0; max-width: 720px; }
    .eyebrow {
        color: #c4ebce; text-transform: uppercase; letter-spacing: 2px;
        font-size: .72rem; font-weight: 700;
    }
    .section-heading {
        color: #183d2c; font-family: 'Manrope', sans-serif;
        font-size: 1.25rem; font-weight: 800; margin: 8px 0 4px;
    }
    .muted { color: #65776a; font-size: .92rem; }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #e0eae0; border-radius: 20px;
        padding: 22px 22px 12px;
        box-shadow: 0 8px 26px rgba(27, 66, 40, .045);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(100deg, #17623d, #2c8a53);
        color: white; border: 0; border-radius: 12px;
        padding: .72rem 1.2rem; font-weight: 700;
        min-height: 48px; transition: .2s ease;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #124d31; color: white; border: 0;
        transform: translateY(-1px);
    }
    .result-card {
        background: linear-gradient(135deg, #ffffff, #f0f8f0);
        border: 1px solid #dcebdd; border-radius: 20px;
        padding: 24px; box-shadow: 0 10px 28px rgba(27, 66, 40, .06);
    }
    .result-label { color: #587361; font-size: .85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .result-number { color: #145b37; font-family: 'Manrope', sans-serif; font-size: clamp(2.1rem, 5vw, 3.3rem); font-weight: 800; line-height: 1.2; margin: 8px 0; }
    .result-unit { color: #6a7c70; font-size: .9rem; }
    .info-card {
        background: #ffffff; border: 1px solid #e3ebe2; border-radius: 16px;
        padding: 16px 18px; height: 100%;
    }
    .info-card b { color: #204b32; }
    .footer { color: #829087; font-size: .8rem; text-align: center; padding-top: 26px; }
    label { color: #294a35 !important; font-weight: 600 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Model loading -----------------------------
@st.cache_resource
def load_models():
    """Load the trained regression model and preprocessing pipeline once."""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "dtr.pkl"
    preprocessor_path = base_dir / "preprocessor.pkl"

    missing = [p.name for p in (model_path, preprocessor_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + ". Place both .pkl files in the same folder as app.py."
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)
    with preprocessor_path.open("rb") as file:
        processor = pickle.load(file)

    return model, processor


# ----------------------------- Header -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI • Agriculture • Decision Support</div>
        <h1>🌾 Smart Crop Yield Predictor</h1>
        <p>Explore how climate, agricultural inputs and crop selection relate to
        estimated crop productivity. Enter the field conditions and generate
        a prediction using your trained machine-learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 🌱 About this project")
    st.write(
        "This application uses a trained Decision Tree Regressor and its "
        "preprocessing pipeline to estimate crop productivity."
    )
    st.divider()
    st.markdown("### Model inputs")
    st.caption("Year • Rainfall • Pesticides • Temperature • Area • Crop")
    st.divider()
    st.markdown("### Important")
    st.caption(
        "Predictions are model estimates, not guaranteed field outcomes. "
        "Use reliable, locally relevant input data."
    )

# ----------------------------- Input form -----------------------------
st.markdown('<div class="section-heading">📋 Enter agricultural details</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Fill in all six fields, then select Predict yield.</div>', unsafe_allow_html=True)
st.write("")

with st.form("crop_prediction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input(
            "Year",
            min_value=1900,
            max_value=2100,
            value=2024,
            step=1,
            help="The year for which crop productivity is being estimated.",
        )
        rainfall = st.number_input(
            "Average rainfall (mm/year)",
            min_value=0.0,
            max_value=20000.0,
            value=1000.0,
            step=10.0,
            help="Annual average rainfall in millimetres.",
        )

    with col2:
        pesticides = st.number_input(
            "Pesticides usage (tonnes)",
            min_value=0.0,
            max_value=100000000.0,
            value=1000.0,
            step=10.0,
            help="Pesticide usage in tonnes, using the same unit and scale as the training data.",
        )
        temperature = st.number_input(
            "Average temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
        )

    with col3:
        area = st.text_input(
            "Country / Area",
            value="India",
            placeholder="Example: India",
            help="Enter an area value exactly as represented in your training dataset.",
        )
        item = st.text_input(
            "Crop / Item",
            value="Rice",
            placeholder="Example: Rice",
            help="Enter the crop name exactly as represented in your training dataset.",
        )

    submitted = st.form_submit_button("🌿 Predict crop yield", use_container_width=True)

# ----------------------------- Prediction -----------------------------
if submitted:
    if not area.strip() or not item.strip():
        st.error("Please enter both the Country / Area and Crop / Item.")
    else:
        try:
            with st.spinner("Analysing agricultural conditions..."):
                model, preprocessor = load_models()

                # Keep the same feature order and object dtype as the reference Flask app.
                features = np.array(
                    [[year, rainfall, pesticides, temperature, area.strip(), item.strip()]],
                    dtype=object,
                )
                transformed_features = preprocessor.transform(features)
                prediction = float(np.asarray(model.predict(transformed_features)).reshape(-1)[0])

            if not np.isfinite(prediction):
                st.error("The model returned an invalid prediction. Please check the input values.")
            else:
                st.session_state["last_prediction"] = prediction
                st.session_state["last_inputs"] = {
                    "Year": int(year),
                    "Average rainfall": f"{rainfall:,.1f} mm/year",
                    "Pesticides usage": f"{pesticides:,.1f} tonnes",
                    "Average temperature": f"{temperature:.1f} °C",
                    "Area": area.strip(),
                    "Crop": item.strip(),
                }
                st.success("Prediction generated successfully!")

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "Prediction could not be generated. Check that the crop and area names "
                "match the values used while training, and that the model files are compatible."
            )
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")

# ----------------------------- Result display -----------------------------
if "last_prediction" in st.session_state:
    prediction = st.session_state["last_prediction"]
    st.write("")
    st.markdown('<div class="section-heading">📈 Prediction result</div>', unsafe_allow_html=True)
    st.write("")

    result_col, details_col = st.columns([1.05, 1], gap="large")

    with result_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated crop productivity</div>
                <div class="result-number">{prediction:,.2f}</div>
                <div class="result-unit">hg/ha • hectograms per hectare</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # A visual display of the predicted numeric value; no quality category is inferred.
        gauge_max = max(abs(prediction) * 1.25, 1.0)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"valueformat": ",.2f", "font": {"size": 30, "color": "#145b37"}},
                title={"text": "Model output (hg/ha)", "font": {"size": 14, "color": "#587361"}},
                gauge={
                    "axis": {"range": [min(0, prediction), gauge_max]},
                    "bar": {"color": "#25804c"},
                    "bgcolor": "#edf5ed",
                    "borderwidth": 0,
                    "steps": [{"range": [min(0, prediction), gauge_max], "color": "#e6f0e6"}],
                },
            )
        )
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with details_col:
        st.markdown("#### Input summary")
        st.markdown('<div class="muted">Conditions used for this prediction</div>', unsafe_allow_html=True)
        for key, value in st.session_state["last_inputs"].items():
            st.markdown(
                f'<div class="info-card" style="margin-bottom:9px;padding:11px 15px;">'
                f'<span class="muted">{key}</span><br><b>{value}</b></div>',
                unsafe_allow_html=True,
            )

        report = pd.DataFrame(
            [{"Input": key, "Value": value} for key, value in st.session_state["last_inputs"].items()]
            + [{"Input": "Predicted productivity", "Value": f"{prediction:,.2f} hg/ha"}]
        )
        st.download_button(
            "⬇️ Download prediction report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="crop_yield_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.info(
        "Interpretation: this is the output of your trained model for the values entered. "
        "It is not a guaranteed harvest quantity or a recommendation to apply a specific input."
    )

# ----------------------------- Footer -----------------------------
st.markdown(
    """
    <div class="footer">
        Smart Crop Yield Prediction System • Machine Learning Project<br>
        Built with Streamlit, NumPy, Pandas and Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
