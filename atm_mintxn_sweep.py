"""
MIN_TXN Sistematik Tarama — 3'ten 30'a
Z-score bir kez hesaplanır (maskeleme yok); her eşik için maskeleme uygulanır.
Baseline sabit tutulmuş yaklaşım: stability elbow tespiti için yeterli doğruluk.
Final pipeline (atm_twin_v3.py) baseline'ı yalnızca geçerli günlerden hesaplar.
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "spar_nord_data"
OUT_DIR  = BASE_DIR
WINDOW_D = 28
Z_THR    = 2.5
MIN_STD  = 0.01

# ═══════════════════════════════════════════════════════════════════════════════
# 1. VERİ + FEATURE ENGINEERING (bir kez)
# ═══════════════════════════════════════════════════════════════════════════════
print("Veri yükleniyor...")
p1 = pd.read_csv(DATA_DIR / "atm_data_part1.csv", low_memory=False)
p2 = pd.read_csv(DATA_DIR / "atm_data_part2.csv", low_memory=False)
df = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2) + "-" +
    df["day"].astype(str).str.zfill(2),
    errors="coerce"
)
df = df.dropna(subset=["date"])

df["is_peak"]       = df["hour"].between(12, 14).astype(int)
df["is_night"]      = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
df["is_morning"]    = df["hour"].between(7, 11).astype(int)
df["is_mastercard"] = df["card_type"].str.contains("MasterCard|Mastercard", na=False).astype(int)

def hour_entropy(hours):
    counts = np.zeros(24)
    for h in hours:
        if pd.notna(h) and 0 <= int(h) < 24:
            counts[int(h)] += 1
    total = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))

print("Hour entropy hesaplanıyor...")
entropy_map = (
    df.groupby(["atm_id","date"])["hour"]
    .apply(hour_entropy).reset_index()
    .rename(columns={"hour":"hour_entropy"})
)

daily = (
    df.groupby(["atm_id","date"])
    .agg(txn_count=("atm_id","count"),
         peak_count=("is_peak","sum"),
         night_count=("is_night","sum"),
         morning_count=("is_morning","sum"),
         mastercard_count=("is_mastercard","sum"))
    .reset_index()
    .merge(entropy_map, on=["atm_id","date"], how="left")
)
daily["peak_rate"]       = (daily["peak_count"]       / daily["txn_count"]).clip(0,1)
daily["night_rate"]      = (daily["night_count"]      / daily["txn_count"]).clip(0,1)
daily["morning_rate"]    = (daily["morning_count"]    / daily["txn_count"]).clip(0,1)
daily["mastercard_rate"] = (daily["mastercard_count"] / daily["txn_count"]).clip(0,1)
daily["calendar_context"] = np.where(daily["date"].dt.day >= 28, "month_end", "")
daily = daily.sort_values(["atm_id","date"]).reset_index(drop=True)

feature_cols = ["txn_count","peak_rate","night_rate","morning_rate","mastercard_rate","hour_entropy"]
print(f"Günlük tablo: {len(daily):,} ATM-gün")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING Z-SCORE — TÜM GÜNLER (maskeleme yok, bir kez)
# ═══════════════════════════════════════════════════════════════════════════════
print("Rolling z-score hesaplanıyor (bir kez, tüm günler)...")

def rzs_all(group, col):
    rm   = group[col].shift(1).rolling(WINDOW_D, min_periods=7).mean()
    rs   = group[col].shift(1).rolling(WINDOW_D, min_periods=7).std()
    safe = rs.where(rs >= MIN_STD, other=np.nan)
    z    = (group[col] - rm) / safe
    return z.fillna(0)

for f in feature_cols:
    res = daily.groupby("atm_id", group_keys=False).apply(lambda g: rzs_all(g, f))
    daily[f"z_{f}"] = res.values

z_cols = [f"z_{f}" for f in feature_cols]
daily["raw_max_z"]  = daily[z_cols].abs().max(axis=1, skipna=True)
daily["raw_n_flag"] = (daily[z_cols].abs() > Z_THR).sum(axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SWEEP: MIN_TXN = 3..30
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_ATM  = 15
TARGET_DATE = pd.Timestamp("2017-08-31")
SWEEP_VALS  = list(range(3, 31))

rows = []
print("Sweep başlıyor (3–30)...")

for mt in SWEEP_VALS:
    mask_valid   = daily["txn_count"] >= mt
    valid        = daily[mask_valid]
    excluded_n   = (~mask_valid).sum()
    excl_pct     = excluded_n / len(daily) * 100

    # Anomali: valid günlerde |z| > Z_THR olan feature var mı?
    anom_mask    = (valid["raw_n_flag"] >= 1)
    z_anom       = anom_mask.sum()
    z_anom_pct   = anom_mask.mean() * 100

    # Top-10 max_z (valid günler)
    top10_maxz   = valid["raw_max_z"].nlargest(10)
    top10_txn    = valid.loc[top10_maxz.index, "txn_count"]
    median_top10 = top10_maxz.median()
    p95_maxz     = valid["raw_max_z"].quantile(0.95)
    max_maxz     = valid["raw_max_z"].max()

    # Top-10 txn_count dağılımı
    top10_txn_median = top10_txn.median()
    top10_txn_min    = top10_txn.min()

    # ATM 15 / 31 Ağustos
    r15 = daily[(daily["atm_id"]==TARGET_ATM) & (daily["date"]==TARGET_DATE)]
    if not r15.empty:
        tc15   = int(r15["txn_count"].values[0])
        z15    = float(r15["z_txn_count"].values[0])
        anom15 = (tc15 >= mt) and (abs(z15) > Z_THR)
    else:
        tc15, z15, anom15 = 0, 0.0, False

    rows.append({
        "min_txn":          mt,
        "valid_days":       len(valid),
        "excluded_days":    excluded_n,
        "excl_pct":         excl_pct,
        "z_anom":           z_anom,
        "z_anom_pct":       z_anom_pct,
        "median_top10_maxz":median_top10,
        "p95_maxz":         p95_maxz,
        "max_maxz":         max_maxz,
        "top10_txn_median": top10_txn_median,
        "top10_txn_min":    top10_txn_min,
        "atm15_anom":       anom15,
        "atm15_z":          z15,
    })

sweep = pd.DataFrame(rows)
print("Sweep tamamlandı.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. KONSOL TABLOSU
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("MIN_TXN SWEEP SONUÇLARI (3–30)")
print("=" * 85)
print(f"{'MIN':>4} {'Valid':>7} {'Excl%':>6} {'Zanom%':>7} {'Med-top10-z':>12} "
      f"{'p95-z':>7} {'max-z':>7} {'top10-txn-med':>14} {'ATM15':>6}")
print("-" * 85)
for _, r in sweep.iterrows():
    atm_mark = "EVET" if r["atm15_anom"] else "HAYIR"
    print(f"{int(r['min_txn']):>4} {int(r['valid_days']):>7,} {r['excl_pct']:>6.1f}% "
          f"{r['z_anom_pct']:>7.1f}% {r['median_top10_maxz']:>12.2f} "
          f"{r['p95_maxz']:>7.2f} {r['max_maxz']:>7.2f} "
          f"{r['top10_txn_median']:>14.0f} {atm_mark:>6}")

# Stability elbow: top10 median max_z'nin değişim hızı
sweep["delta_med"] = sweep["median_top10_maxz"].diff().abs()
sweep["delta_pct_z"] = sweep["z_anom_pct"].diff().abs()

print("\n--- Stability: median top-10 max_z değişim hızı ---")
print(f"{'MIN':>4} {'Med-top10-z':>12} {'|Δ|':>8}")
print("-" * 28)
for _, r in sweep.iterrows():
    d = r["delta_med"]
    marker = " ◄ hızlı düşüş" if d > 2.0 else ("" if pd.isna(d) else "")
    print(f"{int(r['min_txn']):>4} {r['median_top10_maxz']:>12.2f} {d:>8.2f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. TOP-10 AYRINTISI — seçili eşikler için
# ═══════════════════════════════════════════════════════════════════════════════
check_pts = [5, 8, 10, 12, 15, 20]
print("\n--- Top-10 txn_count dağılımı seçili eşiklerde ---")
for mt in check_pts:
    valid = daily[daily["txn_count"] >= mt]
    top10 = valid.nlargest(10, "raw_max_z")[["atm_id","date","txn_count","raw_max_z"]].copy()
    top10["date"] = top10["date"].dt.date
    med_txn = top10["txn_count"].median()
    min_txn = top10["txn_count"].min()
    print(f"\n  MIN={mt}: top-10 içinde txn_count min={min_txn}, median={med_txn:.0f}")
    print(top10.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRAFİKLER
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 13))
fig.suptitle("MIN_TXN Sistematik Tarama (3–30)", fontsize=12, fontweight="bold")

# Grafik 1: anomali oranı vs MIN_TXN
ax = axes[0]
ax.plot(sweep["min_txn"], sweep["z_anom_pct"], "o-", color="#1E3A5F", lw=2, ms=5)
ax.set_ylabel("Z-Score Anomali Oranı (%)", fontsize=9)
ax.set_title("Anomali Oranı — MIN_TXN'e karşı", fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_xticks(SWEEP_VALS)
ax.tick_params(axis="x", labelsize=7)

# Grafik 2: dışlanan veri % vs MIN_TXN
ax = axes[1]
ax.fill_between(sweep["min_txn"], sweep["excl_pct"], alpha=0.3, color="#C0392B")
ax.plot(sweep["min_txn"], sweep["excl_pct"], "o-", color="#C0392B", lw=2, ms=5)
ax.set_ylabel("Dışlanan ATM-Gün (%)", fontsize=9)
ax.set_title("Veri Kaybı — MIN_TXN'e karşı", fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_xticks(SWEEP_VALS)
ax.tick_params(axis="x", labelsize=7)

# Grafik 3: median top-10 max_z vs MIN_TXN — stability elbow
ax = axes[2]
ax.plot(sweep["min_txn"], sweep["median_top10_maxz"],
        "o-", color="#1A6B6B", lw=2, ms=5, label="Median top-10 max_z")
ax.plot(sweep["min_txn"], sweep["p95_maxz"],
        "s--", color="#7B3F00", lw=1.5, ms=4, alpha=0.7, label="p95 max_z")

# Elbow noktasını işaretle (delta_med < 1.0'a yerleştiği ilk nokta)
stable = sweep[sweep["delta_med"] < 1.0].head(1)
if not stable.empty:
    elbow_x = int(stable["min_txn"].values[0])
    elbow_y = float(stable["median_top10_maxz"].values[0])
    ax.axvline(elbow_x, color="darkorange", lw=1.5, ls="--", alpha=0.8,
               label=f"Stability elbow: MIN={elbow_x}")
    ax.annotate(f"elbow\nMIN={elbow_x}", (elbow_x, elbow_y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="darkorange", fontweight="bold")

ax.set_xlabel("MIN_TXN", fontsize=9)
ax.set_ylabel("Max_Z değeri", fontsize=9)
ax.set_title("Top-10 Median Max-Z — MIN_TXN'e karşı (Stability Elbow)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
ax.set_xticks(SWEEP_VALS)
ax.tick_params(axis="x", labelsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "13_mintxn_sweep.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nGrafik: 13_mintxn_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. STABILITY ELBOW ANALİZİ + ÖNERİ
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STABİLİTY ELBOW ANALİZİ")
print("=" * 65)

# Delta serisini yazdır ve elbow noktasını tespit et
deltas = sweep[["min_txn","median_top10_maxz","delta_med","excl_pct","z_anom_pct"]].copy()

# Stability elbow: median_top10_maxz büyük düşüşlerden sonra |Δ| < 1.0'a girdiği ilk nokta.
# valid = daily[txn_count >= mt] filtresinden sonra top-10'un min txn_count'u
# her zaman >= mt olur (tautoloji) — bu bağımsız bir kriter değildir.
crit_a = (
    sweep[sweep["delta_med"].fillna(999) < 1.0].iloc[0]["min_txn"]
    if not sweep[sweep["delta_med"].fillna(999) < 1.0].empty else None
)

print(f"\n  Stability elbow (|Δ| < 1.0 ilk nokta)  : MIN_TXN = {crit_a}")
print(f"  Stability elbow (Grafik)                : MIN_TXN = {elbow_x if not stable.empty else 'N/A'}")

# Öneri: elbow'un bir sonraki adımı — stabilizasyon başladıktan hemen sonra,
# veri kaybı hâlâ düşükken.
recommended = (int(crit_a) + 1) if crit_a is not None else 8
r_row = sweep[sweep["min_txn"] == recommended].iloc[0]
print(f"""
Öneri: MIN_TXN = {int(recommended)}

  Sayısal gerekçe:
    Stability elbow     : MIN_TXN = {int(crit_a) if crit_a else '?'} (|Δ median top-10 max_z| < 1.0 ilk kez)
    Veri kaybı (excl%)  : {r_row['excl_pct']:.1f}%  (kabul edilebilir)
    Z-score anomali %   : {r_row['z_anom_pct']:.1f}%
    Median top-10 max_z : {r_row['median_top10_maxz']:.2f}
    ATM 15 / 31 Ağu     : {'EVET' if r_row['atm15_anom'] else 'HAYIR'}  (referans örnek stabil)

  MIN_TXN < {int(recommended)}: median top-10 max_z henüz stabilize olmamış;
    düşük txn günlerinden kaynaklanan yapay uç z-değerleri top-10'u baskılıyor.
  MIN_TXN > {int(recommended)}: ek veri kaybı artar, marginal kazanım azalır.
""")

sweep.to_csv(OUT_DIR / "mintxn_sweep_results.csv", index=False)
print("Sweep tablosu: mintxn_sweep_results.csv")
