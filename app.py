import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        background-color: #e63946;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        padding: 0.6em;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 5px 0;
    }
    .metric-label { color: #6c757d; font-size: 14px; margin-bottom: 4px; }
    .metric-value { color: #212529; font-size: 26px; font-weight: 700; }
    .predict-box {
        background: linear-gradient(135deg, #e63946 0%, #c1121f 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .predict-title { font-size: 18px; opacity: 0.9; margin-bottom: 8px; }
    .predict-price { font-size: 48px; font-weight: 800; letter-spacing: -1px; }
    .predict-sub   { font-size: 14px; opacity: 0.8; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD PKL FILES (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("locality_mean.pkl", "rb") as f:
        locality_mean = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, locality_mean, feature_cols

model, scaler, locality_mean, feature_cols = load_model()


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏠  House Price Predictor")
st.markdown("**Delhi NCR** | R² = 0.86")
st.divider()


# INPUT FORM
# ─────────────────────────────────────────
st.subheader("🔍 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    locality    = st.selectbox("📍 Locality", sorted(locality_mean.keys()))
    area        = st.number_input("📐 Area (sq ft)", min_value=100, max_value=10000, value=1200, step=50)
    bhk         = st.slider("🛏️ BHK", min_value=1, max_value=6, value=2)
    bathroom    = st.slider("🚿 Bathrooms", min_value=1, max_value=6, value=2)

with col2:
    parking     = st.slider("🚗 Parking", min_value=0, max_value=5, value=1)
    furnishing  = st.selectbox("🛋️ Furnishing", ["Semi-Furnished", "Furnished", "Unfurnished"])
    status      = st.selectbox("🏗️ Status", ["Ready_to_move", "Almost_ready"])
    transaction = st.selectbox("💼 Transaction", ["Resale", "New_Property"])
    prop_type   = st.selectbox("🏢 Type", ["Apartment", "Builder_Floor"])

st.markdown("")

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
if st.button("🔮 Predict Price"):

    # FIX 1 — division safety (bhk=0 se crash nahi hoga)
    apb = area / bhk if bhk != 0 else 0

    # FIX 2 — locality fallback: median is safer than mean
    loc_enc = locality_mean.get(locality, np.median(list(locality_mean.values())))

    row = {
        'Area':                     area,
        'BHK':                      bhk,
        'Bathroom':                 bathroom,
        'Parking':                  parking,
        'log_locality':             np.log1p(loc_enc),
        'log_area':                 np.log1p(area),
        'area_per_bhk':             apb,
        'log_area_per_bhk':         np.log1p(apb),
        # FIX 3 — bathroom/bhk safety
        'bath_per_bhk':             bathroom / bhk if bhk != 0 else 0,
        'bhk_x_bath':               bhk * bathroom,
        'bhk_x_area':               bhk * area,
        'park_x_bhk':               parking * bhk,
        'Transaction_Resale':       1 if transaction == 'Resale' else 0,
        'Type_Builder_Floor':       1 if prop_type == 'Builder_Floor' else 0,
        'Furnishing_Furnished':     1 if furnishing == 'Furnished' else 0,
        'Furnishing_Unfurnished':   1 if furnishing == 'Unfurnished' else 0,
        'Status_Ready_to_move':     1 if status == 'Ready_to_move' else 0,
    }

    input_df = pd.DataFrame([row])

    # FIX 4 — reindex is cleaner than manual loop
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    input_scaled    = scaler.transform(input_df)
    pred_log        = model.predict(input_scaled)[0]
    predicted_price = np.expm1(pred_log)

    def fmt(v):
        return f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.1f} L"

    st.markdown(f"""
    <div class="predict-box">
        <div class="predict-title">Estimated Property Price</div>
        <div class="predict-price">{fmt(predicted_price)}</div>
        <div class="predict-sub">({locality} · {bhk} BHK · {area} sq ft)</div>
    </div>
    """, unsafe_allow_html=True)

    lower = predicted_price * 0.90
    upper = predicted_price * 1.10
    st.info(f"📉 Likely range: **{fmt(lower)}** — **{fmt(upper)}** (±10%)")