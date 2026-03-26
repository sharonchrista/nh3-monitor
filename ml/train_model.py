import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

print("Loading synthetic data...")
df = pd.read_csv("../data/processed/synthetic_data.csv")

# ── Stage 1: Baseline model (days + density only) ─────────────
print("\n[Stage 1] Training baseline model (days + density)...")
BASE_FEATURES = ["days", "density", "protein", "wex", "system", "species"]
SENSOR_FEATURES = ["temp", "ph", "turb", "cons", "vib"]
ALL_FEATURES = BASE_FEATURES + SENSOR_FEATURES

X = df[ALL_FEATURES]
y = df["tan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline uses only farm management variables
base_scaler = StandardScaler()
X_base_train = base_scaler.fit_transform(X_train[BASE_FEATURES])
X_base_test  = base_scaler.transform(X_test[BASE_FEATURES])

base_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)
base_model.fit(X_base_train, y_train)

# Compute residuals — what baseline gets wrong
y_base_pred_train = base_model.predict(X_base_train)
y_base_pred_test  = base_model.predict(X_base_test)
residuals_train   = y_train.values - y_base_pred_train
residuals_test    = y_test.values  - y_base_pred_test

print(f"  Baseline R²:   {r2_score(y_test, y_base_pred_test):.4f}")
print(f"  Residual std:  {residuals_train.std():.4f} mg/L")

# ── Stage 2: Sensor residual model ────────────────────────────
print("\n[Stage 2] Training sensor residual model...")
sensor_scaler = StandardScaler()
X_sensor_train = sensor_scaler.fit_transform(X_train[SENSOR_FEATURES])
X_sensor_test  = sensor_scaler.transform(X_test[SENSOR_FEATURES])

sensor_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.04,
    max_depth=4, min_samples_leaf=8,
    subsample=0.85, random_state=42,
    verbose=0
)
sensor_model.fit(X_sensor_train, residuals_train)

# ── Combined prediction ────────────────────────────────────────
y_sensor_residual = sensor_model.predict(X_sensor_test)
y_combined        = y_base_pred_test + y_sensor_residual
y_combined        = np.clip(y_combined, 0.05, 15.0)

r2_combined  = r2_score(y_test, y_combined)
rmse_combined = np.sqrt(mean_squared_error(y_test, y_combined))
mae_combined  = mean_absolute_error(y_test, y_combined)

print(f"\nCombined model performance:")
print(f"  R²  : {r2_combined:.4f}")
print(f"  RMSE: {rmse_combined:.4f} mg/L")
print(f"  MAE : {mae_combined:.4f} mg/L")

# ── Sensor feature importance ──────────────────────────────────
print(f"\nSensor feature importance (what sensors explain beyond baseline):")
for feat, imp in sorted(
    zip(SENSOR_FEATURES, sensor_model.feature_importances_),
    key=lambda x: -x[1]
):
    bar = "█" * int(imp * 60)
    print(f"  {feat:<12} {bar} {imp:.3f}")

print(f"\nBaseline feature importance (farm management):")
for feat, imp in sorted(
    zip(BASE_FEATURES, base_model.feature_importances_),
    key=lambda x: -x[1]
):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<12} {bar} {imp:.3f}")

# ── Save all models ────────────────────────────────────────────
out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

joblib.dump(base_model,     out_dir / "base_model.pkl")
joblib.dump(base_scaler,    out_dir / "base_scaler.pkl")
joblib.dump(sensor_model,   out_dir / "sensor_model.pkl")
joblib.dump(sensor_scaler,  out_dir / "sensor_scaler.pkl")
joblib.dump(BASE_FEATURES,  out_dir / "base_features.pkl")
joblib.dump(SENSOR_FEATURES,out_dir / "sensor_features.pkl")

print(f"\nSaved to ml/models/:")
print(f"  base_model.pkl    ← farm management baseline")
print(f"  base_scaler.pkl")
print(f"  sensor_model.pkl  ← sensor residual correction")
print(f"  sensor_scaler.pkl")
print(f"  base_features.pkl")
print(f"  sensor_features.pkl")
print(f"\nR² = {r2_combined:.4f} — two-stage model ready for Flask")