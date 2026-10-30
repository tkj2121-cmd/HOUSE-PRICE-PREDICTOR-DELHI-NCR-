"""Train an XGBoost house-price model on Delhi_v2.csv.

Usage:  python train.py [path/to/Delhi_v2.csv] [--out DIR]
Writes: model.joblib (fitted pipeline) and meta.json (UI options + metrics) into DIR.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from xgboost import XGBRegressor

from housing import FEATURES, NUMERIC, ONEHOT, TARGET_ENCODED, build_frame

HERE = Path(__file__).parent
MIN_CITY_COUNT = 30      # cities with fewer listings are grouped as "Other"
MIN_LOCALITY_COUNT = 3   # localities offered in the UI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(HERE / "Delhi_v2.csv"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    X = build_frame(raw)
    counts = X["city"].value_counts()
    X.loc[X["city"].map(counts) < MIN_CITY_COUNT, "city"] = "Other"
    y = np.log1p(raw["price"])          # model log-price: prices are heavily right-skewed

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15, random_state=42)

    pre = ColumnTransformer(
        [("num", "passthrough", NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT),
         # locality has ~1,500 values: target-encode it (cross-fitted, so no leakage)
         ("loc", TargetEncoder(target_type="continuous", random_state=42), TARGET_ENCODED)],
        sparse_threshold=0.0,           # keep dense: XGBoost treats sparse zeros as missing
        verbose_feature_names_out=False,
    )
    Xtr = pre.fit_transform(X_train, y_train)
    Xva = pre.transform(X_val)

    model = XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", early_stopping_rounds=100, random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    pipe = Pipeline([("pre", pre), ("xgb", model)])

    # ---- evaluate on the untouched test split ----
    pred_log = pipe.predict(X_test)
    pred, actual = np.expm1(pred_log), np.expm1(y_test)
    resid = np.asarray(y_test) - pred_log
    metrics = {
        "r2": float(r2_score(actual, pred)),
        "r2_log": float(r2_score(y_test, pred_log)),
        "mae": float(mean_absolute_error(actual, pred)),
        "median_abs_error": float(median_absolute_error(actual, pred)),
        "mape_pct": float(np.mean(np.abs(pred - actual) / actual) * 100),
        "resid_q10": float(np.quantile(resid, 0.10)),
        "resid_q90": float(np.quantile(resid, 0.90)),
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
    }
    print({k: round(v, 3) for k, v in metrics.items()})

    # ---- metadata for the Streamlit UI ----
    full = X.assign(price=raw["price"].values, price_sqft=(raw["price"] / raw["area"]).values)
    localities = {}
    for city, g in full.groupby("city"):
        rows = []
        for name, h in g.groupby("locality"):
            if len(h) >= MIN_LOCALITY_COUNT:
                rows.append({"name": name, "n": int(len(h)),
                             "lat": float(h["latitude"].median()), "lon": float(h["longitude"].median()),
                             "median_price_sqft": float(h["price_sqft"].median()),
                             "median_price": float(h["price"].median())})
        rows.sort(key=lambda r: -r["n"])
        localities[city] = rows
    cities = sorted(c for c in full["city"].unique() if c != "Other") + (["Other"] if "Other" in set(full["city"]) else [])
    city_centre = {c: {"lat": float(g["latitude"].median()), "lon": float(g["longitude"].median()),
                       "median_price_sqft": float(g["price_sqft"].median())}
                   for c, g in full.groupby("city")}

    names = pre.get_feature_names_out()
    imp = sorted(zip(names, model.feature_importances_.tolist()), key=lambda t: -t[1])[:12]

    meta = {
        "cities": cities,
        "localities": localities,
        "city_centre": city_centre,
        "categories": {c: sorted(X[c].unique().tolist()) for c in ["Status", "neworold", "Furnished_status", "type_of_building"]},
        "area_range": [int(raw["area"].min()), int(min(raw["area"].max(), 6000))],
        "metrics": metrics,
        "importances": imp,
    }
    joblib.dump(pipe, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out / 'model.joblib'} and {out / 'meta.json'}")


if __name__ == "__main__":
    main()
