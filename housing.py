"""Shared feature engineering for training (train.py) and the Streamlit app (app.py)."""
import numpy as np
import pandas as pd

NUMERIC = ["area", "latitude", "longitude", "Bedrooms", "Bathrooms",
           "Balcony", "parking", "Lift", "area_per_bed"]
ONEHOT = ["city", "Status", "neworold", "Furnished_status", "type_of_building"]
TARGET_ENCODED = ["locality"]
FEATURES = NUMERIC + ONEHOT + TARGET_ENCODED


def parse_address(addr):
    """'Sector 79, Gurgaon, Delhi NCR' -> ('Sector 79', 'Gurgaon')."""
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    if len(parts) >= 3:
        locality, city = parts[-3], parts[-2]
    elif len(parts) == 2:
        locality, city = parts[0], "Other"
    else:
        locality, city = (parts[0] if parts else "Unknown"), "Other"
    city = city.split(" - ")[0].strip()          # 'New Delhi - West' -> 'New Delhi'
    return locality, city


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw dataset rows into model features.

    Price_sqft is deliberately NOT used: it is price / area, i.e. it leaks the target.
    Missing values in numeric columns are left as NaN (XGBoost handles them natively);
    missing categoricals become 'Unknown'.
    """
    df = pd.DataFrame(index=raw.index)
    addr = raw["Address"].map(parse_address)
    df["locality"] = addr.str[0]
    df["city"] = addr.str[1]

    for c in ["area", "latitude", "longitude", "Bedrooms", "Bathrooms", "Balcony", "parking", "Lift"]:
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    df.loc[df["parking"] > 10, "parking"] = np.nan   # values like 702 are data-entry errors
    df["area_per_bed"] = df["area"] / df["Bedrooms"].clip(lower=1)

    for c in ["Status", "neworold", "Furnished_status", "type_of_building"]:
        df[c] = raw[c].fillna("Unknown").astype(str)
    return df[FEATURES]


def predict_price(model, meta: dict, inputs: dict):
    """Return (price, low, high) in rupees for one property described by `inputs`."""
    row = pd.DataFrame([{
        "area": inputs["area"],
        "latitude": inputs["latitude"],
        "longitude": inputs["longitude"],
        "Bedrooms": inputs["Bedrooms"],
        "Bathrooms": inputs["Bathrooms"],
        "Balcony": inputs.get("Balcony", np.nan),
        "parking": inputs.get("parking", np.nan),
        "Lift": inputs.get("Lift", np.nan),
        "area_per_bed": inputs["area"] / max(inputs["Bedrooms"], 1),
        "city": inputs["city"],
        "Status": inputs["Status"],
        "neworold": inputs["neworold"],
        "Furnished_status": inputs["Furnished_status"],
        "type_of_building": inputs["type_of_building"],
        "locality": inputs["locality"],
    }])[FEATURES]
    log_pred = float(model.predict(row)[0])
    price = float(np.expm1(log_pred))
    low = price * float(np.exp(meta["metrics"]["resid_q10"]))
    high = price * float(np.exp(meta["metrics"]["resid_q90"]))
    return price, low, high


def format_inr(x: float) -> str:
    if x >= 1e7:
        return f"₹ {x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹ {x / 1e5:.1f} Lakh"
    return f"₹ {x:,.0f}"
