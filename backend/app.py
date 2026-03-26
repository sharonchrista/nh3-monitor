from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import csv
import joblib
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATA_FILE = os.path.join("..", "data", "raw", "readings.csv")
MODEL_DIR = Path("../ml/models")

# ── Load models ───────────────────────────────────────────────
print("Loading ML models...")
try:
    base_model      = joblib.load(MODEL_DIR / "base_model.pkl")
    base_scaler     = joblib.load(MODEL_DIR / "base_scaler.pkl")
    sensor_model    = joblib.load(MODEL_DIR / "sensor_model.pkl")
    sensor_scaler   = joblib.load(MODEL_DIR / "sensor_scaler.pkl")
    BASE_FEATURES   = joblib.load(MODEL_DIR / "base_features.pkl")
    SENSOR_FEATURES = joblib.load(MODEL_DIR / "sensor_features.pkl")
    ML_READY = True
    print(f"  Base model    : loaded ({len(BASE_FEATURES)} features)")
    print(f"  Sensor model  : loaded ({len(SENSOR_FEATURES)} features)")
    print(f"  Mode          : two-stage ML prediction")
except Exception as e:
    ML_READY = False
    print(f"  ML models not found — falling back to rule engine ({e})")

# ── Emerson free NH3 formula (Emerson 1975) ───────────────────
def emerson_nh3_fraction(pH, temp_c):
    T   = temp_c + 273.15
    pKa = 0.09018 + 2729.92 / T
    Ka  = 10 ** (-pKa)
    H   = 10 ** (-pH)
    return Ka / (Ka + H)

# ── Rule-based fallback (used if models not loaded) ───────────
def rule_based_tan(d):
    temp    = float(d.get("temp",    28))
    pH      = float(d.get("ph",     7.5))
    turb    = float(d.get("turb",    25))
    cons    = float(d.get("cons",    80))
    vib     = float(d.get("vib",     70))
    density = float(d.get("density", 30))
    days    = float(d.get("days",    45))
    protein = float(d.get("protein",  1))
    wex     = float(d.get("wex",      3))
    system  = float(d.get("system",   0))

    protein_frac  = [0.28, 0.35, 0.43][int(min(protein, 2))]
    body_weight   = max(0.5, 25 * (1 - np.exp(-0.04 * (days - 30))))
    feed_kg_m2    = density * body_weight / 1000 * 0.03
    n_input       = feed_kg_m2 * protein_frac * 0.16 * 1000
    tan_prod      = n_input * 0.60

    do_proxy    = max(0.1, 1 - (turb/200)*0.5 + (cons/100)*0.5)
    temp_factor = 0.3 if temp>35 else 0.7 if temp>30 else 1.0 if temp>20 else 0.5
    pH_factor   = 0.2 if pH<6.5 else 0.7 if pH>8.5 else 1.0
    sys_factor  = [0.3, 0.6, 0.9][int(min(system, 2))]
    nitrif      = 0.25 * do_proxy * temp_factor * pH_factor * sys_factor
    exch        = 0.30 if wex<=0 else 0.15 if wex<=2 else 0.05 if wex<=5 else 0.01

    tan  = max(0, tan_prod * (1 - nitrif - exch)) * 7 * (1 - np.exp(-days/7))
    tan += 0.6 if cons<50 else 0.25 if cons<75 else 0
    tan += 0.3 if vib<30  else 0.10 if vib<55  else 0
    return max(0.05, round(float(tan), 3))

# ── Two-stage ML prediction ───────────────────────────────────
def ml_predict_tan(d):
    species = {"shrimp":0,"tilapia":1,"catfish":2,"carp":3,"milkfish":4}
    sp = species.get(d.get("species","shrimp"), 0)

    base_input = [[
        float(d.get("days",    45)),
        float(d.get("density", 30)),
        float(d.get("protein",  1)),
        float(d.get("wex",      3)),
        float(d.get("system",   0)),
        float(sp)
    ]]
    sensor_input = [[
        float(d.get("temp",  28)),
        float(d.get("ph",   7.5)),
        float(d.get("turb",  25)),
        float(d.get("cons",  80)),
        float(d.get("vib",   70)),
    ]]

    base_scaled   = base_scaler.transform(base_input)
    sensor_scaled = sensor_scaler.transform(sensor_input)

    tan_base     = base_model.predict(base_scaled)[0]
    tan_residual = sensor_model.predict(sensor_scaled)[0]
    tan          = np.clip(tan_base + tan_residual, 0.05, 15.0)
    return round(float(tan), 3), round(float(tan_base), 3), round(float(tan_residual), 3)

# ── Estimate TAN (ML or fallback) ────────────────────────────
def estimate_tan(d):
    if ML_READY:
        tan, _, _ = ml_predict_tan(d)
        return tan
    return rule_based_tan(d)

# ── Surge forecast ────────────────────────────────────────────
def surge_forecast(data, thresh_tan, hours=12):
    forecasts = []
    sim = dict(data)
    for h in range(1, hours + 1):
        sim["days"] = float(data.get("days", 45)) + h / 24
        sim["wex"]  = float(data.get("wex",   3)) + h / 24
        if forecasts and forecasts[-1] > thresh_tan * 0.8:
            sim["cons"] = max(20, float(data.get("cons", 80)) - h * 2)
            sim["vib"]  = max(10, float(data.get("vib",  70)) - h * 3)
        forecasts.append(round(estimate_tan(sim), 3))
    breach = next((i for i, v in enumerate(forecasts) if v >= thresh_tan), -1)
    return {
        "forecasts":       forecasts,
        "hours_to_breach": breach + 1 if breach >= 0 else None,
        "peak":            round(max(forecasts), 3)
    }

# ── Species thresholds ────────────────────────────────────────
THRESHOLDS = {
    "shrimp":   {"tan": 1.0, "free": 0.05, "label": "vannamei"},
    "tilapia":  {"tan": 2.0, "free": 0.10, "label": "Tilapia"},
    "catfish":  {"tan": 3.0, "free": 0.12, "label": "Catfish"},
    "carp":     {"tan": 2.5, "free": 0.10, "label": "Carp"},
    "milkfish": {"tan": 1.5, "free": 0.08, "label": "Milkfish"},
}

# ── ROUTES ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        "status":    "NH3 Monitor API running",
        "version":   "2.0",
        "ml_active": ML_READY,
        "model":     "two-stage gradient boosting" if ML_READY else "rule engine fallback",
        "routes":    ["/predict", "/log", "/data", "/sensor", "/health", "/model_info"]
    })

@app.route("/model_info", methods=["GET"])
def model_info():
    if not ML_READY:
        return jsonify({"ml_active": False, "mode": "rule engine fallback"})
    return jsonify({
        "ml_active":       True,
        "mode":            "two-stage gradient boosting",
        "base_features":   BASE_FEATURES,
        "sensor_features": SENSOR_FEATURES,
        "base_importance": dict(zip(
            BASE_FEATURES,
            [round(float(x),4) for x in base_model.feature_importances_]
        )),
        "sensor_importance": dict(zip(
            SENSOR_FEATURES,
            [round(float(x),4) for x in sensor_model.feature_importances_]
        )),
    })

@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json()
    species = data.get("species", "shrimp")
    thresh  = THRESHOLDS.get(species, THRESHOLDS["shrimp"])
    pH      = float(data.get("ph",   7.5))
    temp    = float(data.get("temp", 28))

    if ML_READY:
        tan, tan_base, tan_sensor = ml_predict_tan(data)
    else:
        tan = rule_based_tan(data)
        tan_base = tan
        tan_sensor = 0.0

    frac     = emerson_nh3_fraction(pH, temp)
    free_nh3 = round(tan * frac, 4)
    forecast = surge_forecast(data, thresh["tan"])

    status = ("critical" if tan > thresh["tan"] else
              "warning"  if tan > thresh["tan"] * 0.75 else
              "safe")

    return jsonify({
        "tan":               tan,
        "tan_baseline":      tan_base,
        "tan_sensor_delta":  tan_sensor,
        "tan_ci_low":        round(max(0, tan * 0.87), 3),
        "tan_ci_high":       round(tan * 1.13, 3),
        "free_nh3":          free_nh3,
        "nh3_fraction_pct":  round(frac * 100, 3),
        "status":            status,
        "threshold_tan":     thresh["tan"],
        "threshold_free":    thresh["free"],
        "species_label":     thresh["label"],
        "forecast":          forecast["forecasts"],
        "hours_to_breach":   forecast["hours_to_breach"],
        "forecast_peak":     forecast["peak"],
        "ml_active":         ML_READY,
        "timestamp":         datetime.now().isoformat()
    })

@app.route("/log", methods=["POST"])
def log_reading():
    data = request.get_json()
    os.makedirs(os.path.dirname(os.path.abspath(DATA_FILE)), exist_ok=True)
    file_exists = os.path.exists(DATA_FILE)
    fields = ["timestamp","temp","ph","turb","cons","vib",
              "density","days","protein","wex","system",
              "predicted_tan","actual_tan","species"]
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":     datetime.now().isoformat(),
            "temp":          data.get("temp",""),
            "ph":            data.get("ph",""),
            "turb":          data.get("turb",""),
            "cons":          data.get("cons",""),
            "vib":           data.get("vib",""),
            "density":       data.get("density",""),
            "days":          data.get("days",""),
            "protein":       data.get("protein",""),
            "wex":           data.get("wex",""),
            "system":        data.get("system",""),
            "predicted_tan": data.get("predicted_tan",""),
            "actual_tan":    data.get("actual_tan",""),
            "species":       data.get("species","shrimp"),
        })
    return jsonify({"status": "logged", "file": DATA_FILE})

@app.route("/data", methods=["GET"])
def get_data():
    if not os.path.exists(DATA_FILE):
        return jsonify({"readings": [], "count": 0})
    rows = []
    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify({"readings": rows[-50:], "count": len(rows)})

@app.route("/sensor", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    print(f"[IoT] Sensor received: {data}")
    return jsonify({"status": "received", "data": data})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "ml_active": ML_READY,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    print("\n  NH3 Monitor API v2.0 — two-stage ML")
    print("  Running at:  http://127.0.0.1:5000")
    print("  Model info:  GET  http://127.0.0.1:5000/model_info")
    print("  Predict:     POST http://127.0.0.1:5000/predict")
    print("  Log reading: POST http://127.0.0.1:5000/log")
    print("  IoT sensor:  POST http://127.0.0.1:5000/sensor")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=5000)