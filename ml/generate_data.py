import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 5000

def rnd(lo, hi, n=None):
    return np.random.uniform(lo, hi, N if n is None else n)

def rnd_int(lo, hi):
    return np.random.randint(lo, hi+1, N)

def calc_tan(temp, ph, turb, cons, vib, density,
             days, protein, wex, system, species):

    # ── 1. Base nitrogen load from biomass (days + density)
    protein_frac = np.choose(protein.astype(int), [0.28, 0.35, 0.43])
    body_weight  = np.maximum(0.5, 25 * (1 - np.exp(-0.04 * (days - 30))))
    feed_kg_m2   = density * body_weight / 1000 * 0.03
    n_input      = feed_kg_m2 * protein_frac * 0.16 * 1000
    tan_base     = n_input * 0.60

    # ── 2. Temperature effect — strong nonlinear impact
    # High temp sharply reduces nitrification AND increases NH3 toxicity
    temp_nitrif  = np.where(temp > 35, 0.20,
                   np.where(temp > 32, 0.45,
                   np.where(temp > 28, 0.75,
                   np.where(temp > 22, 1.00, 0.55))))
    # Temperature also directly adds TAN via increased metabolism
    temp_excretion = np.where(temp > 33, 0.80,
                     np.where(temp > 30, 0.40,
                     np.where(temp > 27, 0.15, 0.0)))

    # ── 3. pH effect on nitrification
    ph_nitrif = np.where(ph < 6.5, 0.20,
                np.where(ph < 7.0, 0.55,
                np.where(ph <= 8.0, 1.00,
                np.where(ph <= 8.5, 0.80, 0.50))))

    # ── 4. Dissolved oxygen proxy (turbidity + consumption combined)
    # High turbidity = high organic load = low DO = poor nitrification
    turb_norm    = np.clip(turb / 220, 0, 1)
    cons_norm    = cons / 100
    do_proxy     = np.clip(cons_norm * 0.6 - turb_norm * 0.4 + 0.5, 0.05, 1.0)
    do_nitrif    = np.where(do_proxy < 0.2, 0.10,
                   np.where(do_proxy < 0.4, 0.40,
                   np.where(do_proxy < 0.6, 0.70, 1.00)))

    # ── 5. Nitrification removal rate (combined sensor factors)
    sys_factor   = np.choose(system.astype(int), [0.30, 0.65, 0.92])
    nitrif_rate  = 0.30 * temp_nitrif * ph_nitrif * do_nitrif * sys_factor

    # ── 6. Water exchange dilution
    exch_rate    = np.where(wex <= 0,  0.35,
                   np.where(wex <= 2,  0.18,
                   np.where(wex <= 5,  0.06,
                   np.where(wex <= 10, 0.02, 0.005))))

    # ── 7. Accumulation over cycle
    net_daily    = np.maximum(0, tan_base * (1 - nitrif_rate - exch_rate))
    tan          = net_daily * 7 * (1 - np.exp(-days / 7))

    # ── 8. Direct additive effects from sensor signals
    # Temperature directly boosts TAN beyond nitrification effect
    tan += temp_excretion * (density / 30)

    # High turbidity adds organic-decay TAN directly
    turb_boost   = np.where(turb > 150, 0.80,
                   np.where(turb > 100, 0.45,
                   np.where(turb > 60,  0.20,
                   np.where(turb > 30,  0.08, 0.0))))
    tan += turb_boost

    # Poor feeding response = ammonia already elevated (behavioral signal)
    cons_boost   = np.where(cons < 30,  1.20,
                   np.where(cons < 50,  0.70,
                   np.where(cons < 70,  0.30,
                   np.where(cons < 85,  0.10, 0.0))))
    tan += cons_boost

    # Low vibration = poor feeding activity = stress = elevated TAN
    vib_boost    = np.where(vib < 20,  0.60,
                   np.where(vib < 40,  0.30,
                   np.where(vib < 60,  0.12, 0.0)))
    tan += vib_boost

    # pH extremes cause direct stress and TAN accumulation
    ph_stress    = np.where(ph < 6.5,  0.50,
                   np.where(ph > 8.8,  0.35,
                   np.where(ph > 8.3,  0.15, 0.0)))
    tan += ph_stress

    # ── 9. Realistic noise ±12%
    noise = 1 + np.random.uniform(-0.12, 0.12, N)
    tan   = np.clip(tan * noise, 0.05, 15.0)
    return np.round(tan, 3)

print(f"Generating {N} synthetic pond scenarios...")
print("Using improved sensor-weighted nitrogen model...")

df = pd.DataFrame({
    "temp":    rnd(20, 38),
    "ph":      rnd(6.5, 9.2),
    "turb":    rnd(2, 220),
    "cons":    rnd(10, 100),
    "vib":     rnd(0, 120),
    "density": rnd(5, 150),
    "days":    rnd(1, 120),
    "protein": rnd_int(0, 2).astype(float),
    "wex":     rnd(0, 15),
    "system":  rnd_int(0, 2).astype(float),
    "species": rnd_int(0, 4).astype(float),
})

df["tan"] = calc_tan(
    df.temp, df.ph, df.turb, df.cons, df.vib,
    df.density, df.days, df.protein, df.wex,
    df.system, df.species
)

out = Path("../data/processed/synthetic_data.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"\nSaved {N} rows → {out}")
print(f"\nTAN distribution:")
print(df["tan"].describe().round(3))
print(f"\nCorrelation with TAN (should be spread across sensors):")
print(df.corr()["tan"].drop("tan").sort_values(ascending=False).round(3))