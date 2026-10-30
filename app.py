"""Streamlit UI for the Delhi NCR house price predictor.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from housing import format_inr, predict_price

HERE = Path(__file__).parent
st.set_page_config(page_title="Delhi NCR House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(HERE / "model.joblib")
    meta = json.loads((HERE / "meta.json").read_text())
    return model, meta


if not (HERE / "model.joblib").exists():
    st.error("model.joblib not found. Run `python train.py` first, then reload this page.")
    st.stop()

model, meta = load_artifacts()
cats = meta["categories"]

st.title("🏠 Delhi NCR House Price Predictor")
st.caption("XGBoost model trained on ~7.7k property listings across Delhi, Gurgaon, Noida, "
           "Greater Noida, Ghaziabad and Faridabad.")

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Property details")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City / region", meta["cities"], index=meta["cities"].index("Gurgaon") if "Gurgaon" in meta["cities"] else 0)

    locs = meta["localities"].get(city, [])
    loc_names = [l["name"] for l in locs] + ["Other / not listed"]
    loc_choice = c2.selectbox("Locality", loc_names)
    if loc_choice == "Other / not listed":
        centre = meta["city_centre"][city]
        lat0, lon0, ref_psf = centre["lat"], centre["lon"], centre["median_price_sqft"]
        locality_for_model = loc_choice
    else:
        l = next(x for x in locs if x["name"] == loc_choice)
        lat0, lon0, ref_psf = l["lat"], l["lon"], l["median_price_sqft"]
        locality_for_model = loc_choice

    c3, c4 = st.columns(2)
    btype = c3.selectbox("Property type", cats["type_of_building"])
    area = c4.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1200, step=50)

    c5, c6, c7 = st.columns(3)
    beds = c5.number_input("Bedrooms", 1, 10, 3)
    baths = c6.number_input("Bathrooms", 1, 10, 2)
    balc = c7.selectbox("Balconies", [None, 1, 2, 3, 4, 5], format_func=lambda v: "Not specified" if v is None else str(v))

    c8, c9, c10 = st.columns(3)
    parking = c8.selectbox("Parking spots", [None, 1, 2, 3, 4, 5, 6], format_func=lambda v: "Not specified" if v is None else str(v))
    lift = c9.selectbox("Lifts", [None, 0, 1, 2, 3, 4], format_func=lambda v: "Not specified" if v is None else str(v))
    furnished = c10.selectbox("Furnishing", cats["Furnished_status"])

    c11, c12 = st.columns(2)
    status = c11.selectbox("Possession status", cats["Status"])
    age = c12.selectbox("New or resale", cats["neworold"])

    with st.expander("Advanced: fine-tune location"):
        st.caption("Defaults to the typical coordinates of the chosen locality.")
        a, b = st.columns(2)
        lat = a.number_input("Latitude", value=float(lat0), format="%.5f", step=0.001)
        lon = b.number_input("Longitude", value=float(lon0), format="%.5f", step=0.001)

    go = st.button("Predict price", type="primary", use_container_width=True)

with right:
    st.subheader("Estimate")
    if go:
        price, low, high = predict_price(model, meta, {
            "area": area, "latitude": lat, "longitude": lon, "Bedrooms": beds, "Bathrooms": baths,
            "Balcony": np.nan if balc is None else balc,
            "parking": np.nan if parking is None else parking,
            "Lift": np.nan if lift is None else lift,
            "city": city, "Status": status, "neworold": age,
            "Furnished_status": furnished, "type_of_building": btype, "locality": locality_for_model,
        })
        st.metric("Estimated price", format_inr(price))
        st.write(f"**Likely range (80%):** {format_inr(low)} – {format_inr(high)}")
        psf = price / area
        st.write(f"**Price per sq ft:** ₹ {psf:,.0f}")
        st.write(f"**Typical in {loc_choice if loc_choice != 'Other / not listed' else city}:** ₹ {ref_psf:,.0f} / sq ft")
        diff = (psf / ref_psf - 1) * 100
        st.caption(f"This estimate is {abs(diff):.0f}% {'above' if diff >= 0 else 'below'} the typical listing price per sq ft in this area.")
    else:
        st.info("Fill in the details and click **Predict price**.")

    with st.expander("Model performance (held-out test set)"):
        m = meta["metrics"]
        st.write(f"R²: **{m['r2']:.3f}**  ·  Median error: **{format_inr(m['median_abs_error'])}**  ·  "
                 f"Mean abs. % error: **{m['mape_pct']:.1f}%**")
        st.caption(f"Trained on {m['n_train']:,} listings, validated on {m['n_val']:,}, tested on {m['n_test']:,}.")
        imp = pd.DataFrame(meta["importances"], columns=["feature", "importance"]).set_index("feature")
        st.bar_chart(imp)

st.caption("Estimates are statistical predictions from listing data, not valuations or financial advice.")
