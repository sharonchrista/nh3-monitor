# NH₃ Monitor — AI-Powered Ammonia Prediction System for Aquaculture

An end-to-end ammonia monitoring and prediction system designed for 
low-cost Indian aquaculture farms. Predicts total ammonia nitrogen (TAN) 
and free NH₃ levels in real time using a two-stage machine learning model 
trained on physics-informed synthetic data — no sensors required to start.

---

## The Problem

Ammonia is the leading cause of fish and shrimp mortality in aquaculture.
It is expensive and slow to measure directly — lab tests cost time and money,
and by the time results come back, the damage is done. Continuous sensor 
systems exist but are priced out of reach for small Indian farms.

This system predicts ammonia from variables farmers already know — feeding 
behaviour, water appearance, stocking density, production stage — and from 
low-cost sensors costing under ₹3,800 total.

---

## What It Does

- Predicts TAN (mg/L) and toxic free NH₃ (mg/L) in real time
- Generates a 12-hour surge forecast — warns before crisis hits
- Shows which factors are driving ammonia risk
- Logs actual test kit readings to build a training dataset
- Upgrades automatically as real data accumulates
- Connects to IoT sensors (ESP32) when hardware is ready

---

## System Architecture
```
Sensor inputs (IoT / manual)
        ↓
Flask API (Python)
        ↓
Two-stage ML model
  ├── Stage 1: Baseline (farm management variables)
  │     days, density, protein, water exchange, system type
  └── Stage 2: Sensor correction (real-time signals)
        temperature, pH, turbidity, feed consumption, vibration
        ↓
Emerson formula → free NH₃ fraction
        ↓
Live dashboard + 12h surge forecast
```

---

## ML Model — Physics-Informed Synthetic Training

Since real labelled ammonia data is scarce, the model is trained on 
5,000 synthetic pond scenarios generated from established aquaculture 
science equations:

- **Nitrogen budget** — Beveridge (1996): feed × protein → TAN production
- **Nitrification kinetics** — Wheaton (1991): temperature + DO → TAN removal  
- **Emerson formula** (1975): pH + temperature → free NH₃ fraction

**Two-stage gradient boosting (scikit-learn):**

| Stage | Input features | Purpose |
|---|---|---|
| Baseline model | days, density, protein, water exchange, system | Biomass-driven TAN |
| Sensor model | temperature, pH, turbidity, consumption %, vibration | Residual correction |

**Performance on held-out synthetic test set:**
- R² = 0.9753
- RMSE = 0.81 mg/L
- MAE = 0.58 mg/L

When real sensor + test kit data is collected (100+ readings), 
the model retrains on actual data for farm-specific accuracy.

---

## Sensor Stack — Under ₹3,800 Total

| Sensor | Measures | Cost |
|---|---|---|
| DS18B20 waterproof probe | Water temperature | ₹100–180 |
| DFRobot analog pH sensor | pH (optional) | ₹800–1,200 |
| TSW-20 turbidity sensor | Turbidity / organic load | ₹300–500 |
| HX711 load cell 5kg | Feed consumption % | ₹350–600 |
| SW-420 vibration × 3 | Feeding activity / behavior | ₹50–100 |
| BME280 | Air temperature / humidity | ₹150–250 |
| Float switch | Water exchange events | ₹50–100 |
| ESP32 microcontroller | WiFi + runs all sensors | ₹250–400 |

**No camera. No cloud subscription. Works offline.**

---

## Feeding Behaviour Without a Camera

Two techniques replace camera-based feeding detection:

**Load cell under feed tray** — weighs feed before and after each 
session. Consumption below 70% flags stress automatically.

**SW-420 vibration sensors (×3)** — clip to pond frame. Active 
feeding creates surface agitation pulses. Quiet surface during 
feeding = appetite suppressed. Combined with load cell for 
dual confirmation.

---

## Species Support

| Species | Safe TAN limit | Safe free NH₃ |
|---|---|---|
| White leg shrimp (vannamei) | 1.0 mg/L | 0.05 mg/L |
| Tilapia | 2.0 mg/L | 0.10 mg/L |
| Catfish | 3.0 mg/L | 0.12 mg/L |
| Common carp | 2.5 mg/L | 0.10 mg/L |
| Milkfish | 1.5 mg/L | 0.08 mg/L |

---

## Project Structure
```
nh3_monitor/
├── backend/
│   └── app.py              # Flask API v2.0 — predict, log, sensor, health
├── frontend/
│   └── index.html          # Live dashboard — connects to Flask API
├── ml/
│   ├── generate_data.py    # Physics-informed synthetic data generator
│   ├── train_model.py      # Two-stage gradient boosting trainer
│   └── models/             # Saved .pkl model files
├── data/
│   ├── processed/          # synthetic_data.csv (5,000 rows)
│   └── raw/                # readings.csv — grows as you log real data
├── iot_simulator/
│   └── simulate.py         # Mock ESP32 — 4 pond scenarios
├── requirements.txt
└── HOWTORUN.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard (HTML) |
| POST | `/predict` | Predict TAN + NH₃ + 12h forecast |
| POST | `/log` | Log actual test kit reading |
| GET | `/data` | Last 50 logged readings |
| POST | `/sensor` | Receive live ESP32 sensor data |
| GET | `/model_info` | Feature importances, model metadata |
| GET | `/health` | API health check |

**Example predict request:**
```json
POST /predict
{
  "species": "shrimp",
  "temp": 30,
  "ph": 7.8,
  "turb": 45,
  "cons": 72,
  "vib": 55,
  "density": 30,
  "days": 45,
  "protein": 1,
  "wex": 3,
  "system": 0
}
```

**Example response:**
```json
{
  "tan": 2.466,
  "tan_baseline": 3.212,
  "tan_sensor_delta": -0.745,
  "tan_ci_low": 2.145,
  "tan_ci_high": 2.787,
  "free_nh3": 0.1105,
  "nh3_fraction_pct": 4.821,
  "status": "critical",
  "hours_to_breach": 1,
  "forecast_peak": 3.216,
  "ml_active": true
}
```

---

## How to Run Locally

**Requirements:** Python 3.x, Git
```bash
# Clone
git clone https://github.com/sharonchrista/nh3-monitor.git
cd nh3-monitor

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data + train model
cd ml
python generate_data.py
python train_model.py

# Start Flask API
cd ../backend
python app.py

# Open dashboard
# Browser → http://127.0.0.1:5000
```

**Run IoT simulator** (optional, second terminal):
```bash
cd iot_simulator
python simulate.py
```

---

## IoT Integration — ESP32

When hardware is ready, the ESP32 posts sensor readings directly 
to the API. No dashboard changes needed:
```cpp
// Arduino / ESP32 code (simplified)
HTTPClient http;
http.begin("http://YOUR_LAPTOP_IP:5000/sensor");
http.addHeader("Content-Type", "application/json");
String payload = "{\"temp\":" + String(temp) + 
                 ",\"ph\":"   + String(ph)   + 
                 ",\"turb\":" + String(turb) + "}";
http.POST(payload);
```

---

## Roadmap

- [x] Rule-based risk engine (Phase 1)
- [x] Physics-informed synthetic data generation
- [x] Two-stage gradient boosting ML model (Phase 2)
- [x] Live Flask API with all endpoints
- [x] Real-time dashboard with 12h surge forecast
- [x] IoT simulator (4 pond scenarios)
- [ ] ESP32 hardware integration (sensor team)
- [ ] Retrain on real farm data (100+ readings)
- [ ] LSTM time-series model for longer forecast horizon (Phase 3)
- [ ] Ollama local LLM for plain-language farm recommendations
- [ ] Mobile-friendly dashboard
- [ ] WhatsApp / SMS alert integration for India

---

## Scientific References

- Emerson, K. et al. (1975). Aqueous ammonia equilibrium calculations.
  *Journal of the Fisheries Research Board of Canada*, 32(12), 2379–2383.
- Beveridge, M.C.M. (1996). *Cage Aquaculture* (2nd ed.). Fishing News Books.
- Wheaton, F.W. (1991). *Recirculating Aquaculture Systems*. 
  Northeastern Regional Aquaculture Center.
- Boyd, C.E. (1998). *Water Quality for Pond Aquaculture*. 
  Auburn University.

---

## Author

**Sharon Christa**  
AI/ML researcher — Aquaculture water quality monitoring  
GitHub: [@sharonchrista](https://github.com/sharonchrista)  
Email: sharonchrista.ai@gmail.com

---

*Built for low-cost Indian aquaculture farms — 
cost edge through AI, not expensive hardware.*