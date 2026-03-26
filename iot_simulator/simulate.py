import requests
import time
import random
import math
import json
from datetime import datetime

API = "http://127.0.0.1:5000"

# ── Pond scenario profiles ────────────────────────────────────
SCENARIOS = {
    "normal": {
        "desc": "Healthy pond — stable conditions",
        "temp": (26, 29), "ph": (7.2, 7.8),
        "turb": (10, 40),  "cons": (80, 95),
        "vib":  (70, 110), "wex": (1, 4)
    },
    "heat_stress": {
        "desc": "Hot weather — temperature rising",
        "temp": (32, 37), "ph": (7.8, 8.4),
        "turb": (30, 80),  "cons": (50, 75),
        "vib":  (35, 65),  "wex": (4, 8)
    },
    "algae_crash": {
        "desc": "Phytoplankton crash — oxygen depletion",
        "temp": (28, 32), "ph": (8.5, 9.1),
        "turb": (80, 180), "cons": (20, 50),
        "vib":  (10, 35),  "wex": (5, 12)
    },
    "critical": {
        "desc": "Emergency — ammonia surge in progress",
        "temp": (33, 38), "ph": (8.2, 8.9),
        "turb": (120, 220),"cons": (10, 35),
        "vib":  (5, 20),   "wex": (8, 15)
    }
}

def rnd(lo, hi):
    return round(random.uniform(lo, hi), 2)

def simulate_pond(scenario_name, species, density, days, protein, system):
    sc = SCENARIOS[scenario_name]
    return {
        "species": species,
        "temp":    rnd(*sc["temp"]),
        "ph":      rnd(*sc["ph"]),
        "turb":    rnd(*sc["turb"]),
        "cons":    rnd(*sc["cons"]),
        "vib":     round(rnd(*sc["vib"])),
        "density": density,
        "days":    round(days, 1),
        "protein": protein,
        "wex":     rnd(*sc["wex"]),
        "system":  system
    }

def print_result(payload, result, scenario):
    status_colors = {
        "safe":     "\033[92m",  # green
        "warning":  "\033[93m",  # yellow
        "critical": "\033[91m",  # red
    }
    reset = "\033[0m"
    col   = status_colors.get(result.get("status",""), "")

    print(f"\n{'─'*60}")
    print(f"  {datetime.now().strftime('%H:%M:%S')} | Scenario: {scenario.upper()}")
    print(f"  {SCENARIOS[scenario]['desc']}")
    print(f"{'─'*60}")
    print(f"  Inputs  → temp:{payload['temp']}°C  pH:{payload['ph']}  "
          f"turb:{payload['turb']}NTU  cons:{payload['cons']}%  vib:{payload['vib']}")
    print(f"  Baseline→ TAN {result.get('tan_baseline','—')} mg/L "
          f"(farm management signal)")
    print(f"  Sensor  → delta {result.get('tan_sensor_delta','—'):+.3f} mg/L "
          f"(sensor correction)")
    print(f"  {col}PREDICTED TAN : {result['tan']} mg/L  "
          f"[{result['tan_ci_low']}–{result['tan_ci_high']}]{reset}")
    print(f"  Free NH₃      : {result['free_nh3']} mg/L  "
          f"({result['nh3_fraction_pct']}% of TAN)")
    print(f"  {col}Status        : {result['status'].upper()}{reset}")
    breach = result.get('hours_to_breach')
    print(f"  Surge forecast: {'NOW — breach active' if result['tan'] >= result['threshold_tan'] else (str(breach)+'h to breach' if breach else 'stable 24h+')}")
    print(f"  Forecast peak : {result['forecast_peak']} mg/L  "
          f"(threshold: {result['threshold_tan']} mg/L for {result['species_label']})")
    print(f"  ML active     : {result.get('ml_active', False)}")

def check_api():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        d = r.json()
        print(f"  API: {d['status']} | ML: {d['ml_active']}")
        return True
    except:
        print(f"  ERROR: Cannot reach {API}")
        print(f"  Make sure Flask is running: python backend/app.py")
        return False

# ── Main simulation loop ──────────────────────────────────────
print("\n" + "="*60)
print("  NH3 MONITOR — IoT Simulator")
print("  Simulates ESP32 sending sensor data to Flask API")
print("="*60)

print("\nChecking API connection...")
if not check_api():
    exit(1)

print("\nConfiguration:")
species  = "shrimp"
density  = 30
days     = 45.0
protein  = 1
system   = 0

print(f"  Species: {species} | Density: {density}/m² | "
      f"Day: {days} | System: earthen pond")
print(f"\nStarting simulation — Ctrl+C to stop\n")

scenario_sequence = [
    ("normal",      8),
    ("heat_stress", 5),
    ("algae_crash", 5),
    ("critical",    4),
]

interval = 4  # seconds between readings
reading_count = 0

try:
    while True:
        for scenario_name, count in scenario_sequence:
            for i in range(count):
                days += interval / 86400  # increment by real elapsed time
                payload = simulate_pond(
                    scenario_name, species, density,
                    days, protein, system
                )
                try:
                    r = requests.post(
                        f"{API}/predict",
                        json=payload,
                        timeout=5
                    )
                    result = r.json()
                    reading_count += 1
                    print_result(payload, result, scenario_name)
                    print(f"  Reading #{reading_count} | "
                          f"Next in {interval}s...")
                except requests.exceptions.ConnectionError:
                    print(f"\n  [ERROR] Flask API not responding. "
                          f"Is it still running?")
                except Exception as e:
                    print(f"\n  [ERROR] {e}")

                time.sleep(interval)

except KeyboardInterrupt:
    print(f"\n\n  Simulation stopped after {reading_count} readings.")
    print(f"  Data logged at: data/raw/readings.csv")
    print(f"  GitHub: https://github.com/sharonchrista/nh3-monitor")