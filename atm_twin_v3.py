import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "spar_nord_data"
OUT_DIR  = BASE_DIR

WINDOW_D = 28
WINDOW_W = 8
Z_THR    = 2.5
MIN_STD  = 0.01
CHOSEN_MIN_TXN = 8
TARGET_ATM  = 15
TARGET_DATE = pd.Timestamp("2017-08-31")

# ── 1. VERİ YÜKLE + TEMİZLE + FEATURE ENGINEERING ──────────────────────────
print("Veri yükleniyor...")
p1 = pd.read_csv(DATA_DIR / "atm_data_part1.csv", low_memory=False)
p2 = pd.read_csv(DATA_DIR / "atm_data_part2.csv", low_memory=False)
df_raw = pd.concat([p1, p2], ignore_index=True)
# Duplicate: 33 orijinal sütunun tamamı birebir eşit olan satırları kaldır.
# is_dup gibi türetilmiş kolon varken drop_duplicates() çağırmak partial cleaning yapar;
# bu nedenle drop_duplicates() ham df_raw üzerinde, ek kolon eklenmeden çağrılır.
df     = df_raw.drop_duplicates().copy()

df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" +
    df["month"].astype(str).str.zfill(2) + "-" +
    df["day"].astype(str).str.zfill(2),
    errors="coerce"
)
df = df.dropna(subset=["date"])
print(f"  Temizlenmiş: {len(df):,} satır | {df['atm_id'].nunique()} ATM")

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

daily["weekday"]  = daily["date"].dt.dayofweek
daily["month"]    = daily["date"].dt.month
daily["week"]     = daily["date"].dt.isocalendar().week.astype(int)
daily["year_num"] = daily["date"].dt.year
# month_end_window: gün >= 28 koşulunu ifade eder; gerçek ay sonu değil pencere yaklaşımı.
# Anomali kararını değiştirmez; operasyonel değerlendirmede bağlam olarak sunulur.
daily["calendar_context"] = np.where(daily["date"].dt.day >= 28, "month_end_window", "")

feature_cols = ["txn_count","peak_rate","night_rate","morning_rate","mastercard_rate","hour_entropy"]
daily = daily.sort_values(["atm_id","date"])
print(f"Günlük tablo: {len(daily):,} ATM-gün")

# ── 2. ROLLING Z-SCORE ──────────────────────────────────────────────────────
def compute_zscores(df_in, min_txn):
    d = df_in.copy()
    # txn_count < min_txn olan günler profil dışı: az örnek nedeniyle oran feature'larının
    # güvenilirliği düşer ve uç değer üretme riski artar.
    d["low_txn"] = (d["txn_count"] < min_txn).astype(int)

    def rzs(group, col):
        # z = (güncel - baseline_ort) / baseline_std  —  her ATM kendi geçmişiyle kıyaslanır.
        # Baseline yalnızca geçerli günlerden (txn_count >= min_txn) hesaplanır:
        # düşük txn günleri NaN olarak maskelenir, rolling hesabı NaN'ları atlar.
        col_valid = group[col].where(group["txn_count"] >= min_txn, other=np.nan)
        rm   = col_valid.shift(1).rolling(WINDOW_D, min_periods=7).mean()
        rs   = col_valid.shift(1).rolling(WINDOW_D, min_periods=7).std()
        # rolling_std < MIN_STD ise z'yi 0 yap: çok düşük varyansta matematiksel şişme önlenir.
        safe = rs.where(rs >= MIN_STD, other=np.nan)
        z    = (group[col] - rm) / safe
        z    = z.fillna(0)
        z[group["txn_count"] < min_txn] = np.nan  # profil dışı günler değerlendirme dışı
        return z

    for f in feature_cols:
        res = d.groupby("atm_id", group_keys=False).apply(lambda g: rzs(g, f))
        d[f"z_{f}"] = res.values

    z_cols = [f"z_{f}" for f in feature_cols]
    def nflag(row):
        return sum(1 for f in feature_cols
                   if pd.notna(row[f"z_{f}"]) and abs(row[f"z_{f}"]) > Z_THR)
    d["n_flagged"]      = d.apply(nflag, axis=1)
    d["zscore_anomaly"] = ((d["n_flagged"] >= 1) & (d["low_txn"] == 0)).astype(int)
    d["max_z"]          = d[[f"z_{f}" for f in feature_cols]].abs().max(axis=1, skipna=True)
    return d, z_cols

# ── 3. MIN_TXN KARŞILAŞTIRMASI (5 / 10 / 20) ───────────────────────────────
print("\nMIN_TXN karşılaştırması hesaplanıyor (3 eşik)...")
results = {}
for mt in [5, 10, 20]:
    d, zc = compute_zscores(daily, mt)
    valid     = d[d["low_txn"] == 0]
    excluded  = d[d["low_txn"] == 1]
    valid_nona = valid.dropna(subset=zc)
    scaler = StandardScaler()
    X = scaler.fit_transform(valid_nona[zc])
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(X)
    valid_nona = valid_nona.copy()
    valid_nona["if_anomaly"] = (clf.predict(X) == -1).astype(int)
    valid_nona["if_score"]   = -clf.score_samples(X)
    both = ((valid_nona["zscore_anomaly"] == 1) & (valid_nona["if_anomaly"] == 1)).sum()
    top5 = (valid_nona.nlargest(5, "if_score")
            [["atm_id","date","txn_count","max_z","if_score"]].copy())
    top5["date"] = top5["date"].dt.date
    r15 = d[(d["atm_id"] == TARGET_ATM) & (d["date"] == TARGET_DATE)]
    results[mt] = {
        "valid": len(valid_nona),
        "excluded": len(excluded),
        "excl_pct": len(excluded) / len(d) * 100,
        "z_anom": valid_nona["zscore_anomaly"].sum(),
        "z_pct":  valid_nona["zscore_anomaly"].mean() * 100,
        "both": both,
        "both_pct": both / len(valid_nona) * 100,
        "top5": top5,
        "atm15_z":    float(r15["z_txn_count"].values[0]) if not r15.empty else float("nan"),
        "atm15_tc":   int(r15["txn_count"].values[0])     if not r15.empty else 0,
        "atm15_anom": int(r15["zscore_anomaly"].values[0]) if not r15.empty else 0,
    }
    print(f"  MIN_TXN={mt} tamamlandı")

print("\n" + "=" * 65)
print("1. MIN_TXN KARŞILAŞTIRMASI")
print("=" * 65)
print(f"\n{'Metrik':<35} {'MIN=5':>10} {'MIN=10':>10} {'MIN=20':>10}")
print("-" * 68)
rows = [
    ("Değerlendirilebilir ATM-gün",  "valid",     "d",  ),
    ("Profil dışı ATM-gün",          "excluded",  "d",  ),
    ("Profil dışı %",                "excl_pct",  "f1", ),
    ("Z-score anomali",              "z_anom",    "d",  ),
    ("Z-score anomali %",            "z_pct",     "f1", ),
    ("IF+Z ortak (güçlü sinyal)",    "both",      "d",  ),
    ("IF+Z ortak %",                 "both_pct",  "f2", ),
    ("ATM15/31Ağu txn_count",        "atm15_tc",  "d",  ),
    ("ATM15/31Ağu z_txn",            "atm15_z",   "f2", ),
    ("ATM15/31Ağu anomali?",         "atm15_anom","b",  ),
]
for label, key, fmt in rows:
    vals = []
    for mt in [5, 10, 20]:
        v = results[mt][key]
        if   fmt == "d":  vals.append(f"{int(v):,}")
        elif fmt == "f1": vals.append(f"%{v:.1f}")
        elif fmt == "f2": vals.append(f"{v:.2f}")
        elif fmt == "b":  vals.append("EVET" if v else "HAYIR")
    print(f"{label:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

for mt in [5, 10, 20]:
    print(f"\n--- Top 5 yüksek IF skoru (MIN={mt}) ---")
    print(results[mt]["top5"].to_string(index=False))

# ── 4. FİNAL PIPELINE (MIN_TXN=8) ───────────────────────────────────────────
daily_v3, z_cols = compute_zscores(daily, CHOSEN_MIN_TXN)
valid_v3   = daily_v3[daily_v3["low_txn"] == 0].copy()
valid_nona = valid_v3.dropna(subset=z_cols).copy()
scaler = StandardScaler()
X = scaler.fit_transform(valid_nona[z_cols])
# Isolation Forest: 6 feature'ı birlikte değerlendiren destekleyici sinyal.
# contamination=0.05 eğitim etiketi değil; threshold oluşturulurken ~%5 outlier
# hedefleyen ayardır. Çıkan anomali oranı bu parametrenin yansımasıdır.
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(X)
valid_nona["if_score"]   = -clf.score_samples(X)
valid_nona["if_anomaly"] = (clf.predict(X) == -1).astype(int)
daily_v3 = daily_v3.merge(
    valid_nona[["atm_id","date","if_score","if_anomaly"]],
    on=["atm_id","date"], how="left"
)

def get_triggered(row):
    if row["low_txn"] == 1:
        return f"profil_disi (txn<{CHOSEN_MIN_TXN})"
    pairs = [(f, abs(row[f"z_{f}"])) for f in feature_cols
             if pd.notna(row[f"z_{f}"]) and abs(row[f"z_{f}"]) > Z_THR]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return ", ".join([f"{n}({v:.1f}σ)" for n,v in pairs]) if pairs else "—"

daily_v3["triggered_features"] = daily_v3.apply(get_triggered, axis=1)

# ── 5. ATM 15 / 31 AĞUSTOS ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"2. ATM {TARGET_ATM} / 31 Ağustos 2017 (MIN_TXN={CHOSEN_MIN_TXN})")
print("=" * 65)

atm15  = daily_v3[daily_v3["atm_id"] == TARGET_ATM].sort_values("date").copy()
row31  = atm15[atm15["date"] == TARGET_DATE]

if not row31.empty:
    r   = row31.iloc[0]
    idx = atm15.index.get_loc(row31.index[0])
    win = atm15.iloc[max(0, idx - WINDOW_D): idx]
    win_e = win[win["low_txn"] == 0]

    print(f"\nTemizlenmiş ham satır: {int(r['txn_count'])}")
    print(f"{'Feature':<20} {'28g-Ort':>9} {'28g-Std':>9} {'Bant [±2σ]':>18} {'Güncel':>9} {'Z':>7} {'Sapma%':>8} {'Durum':>9}")
    print("-" * 96)
    for f in feature_cols:
        mu   = win_e[f].mean()
        sig  = win_e[f].std() if win_e[f].std() > 0 else 0.0
        curr = r[f]
        z_v  = r[f"z_{f}"]
        if pd.isna(z_v): z_v = 0.0
        hi   = mu + 2*sig
        lo   = max(0, mu - 2*sig)
        pct  = (curr - mu) / mu * 100 if mu > 0 else 0.0
        if   abs(z_v) >= 3.5:  durum = "KRİTİK"
        elif abs(z_v) >= Z_THR: durum = "Dikkat"
        else:                   durum = "Normal"
        print(f"{f:<20} {mu:>9.3f} {sig:>9.3f} [{lo:.2f} — {hi:.2f}]{'':<2} {curr:>9.3f} {z_v:>7.2f} {pct:>8.1f}% {durum:>9}")

    if_s = r.get("if_score", float("nan"))
    if_a = int(r.get("if_anomaly", 0)) if pd.notna(r.get("if_anomaly")) else 0
    print(f"\nIF Anomaly Score     : {if_s:.3f}")
    print(f"IF Anomali           : {'EVET' if if_a else 'HAYIR'}")
    print(f"Z-Score Anomali      : {'EVET' if r['zscore_anomaly'] else 'HAYIR'}")
    print(f"Tetikleyen           : {r['triggered_features']}")
    ctx = r["calendar_context"]
    print(f"calendar_context     : {ctx if ctx else '—'}")

# ── 6. HAFTALIK BEHAVIORAL PROFILE ──────────────────────────────────────────
print("\n" + "=" * 65)
print("3. HAFTALIK BEHAVIORAL PROFILE (ATM kendi geçmiş haftalarıyla)")
print("=" * 65)

weekly_base = (
    daily_v3[daily_v3["low_txn"] == 0]
    .groupby(["atm_id","year_num","week"])
    .agg(
        week_start       = ("date",           "min"),
        week_txn_total   = ("txn_count",      "sum"),
        week_txn_days    = ("txn_count",      "count"),
        week_peak_rate   = ("peak_rate",      "mean"),
        week_night_rate  = ("night_rate",     "mean"),
        week_morning_rate= ("morning_rate",   "mean"),
        week_mc_rate     = ("mastercard_rate","mean"),
        week_entropy     = ("hour_entropy",   "mean"),
    )
    .reset_index()
    .sort_values(["atm_id","week_start"])
)

weekly_fcols = ["week_txn_total","week_peak_rate","week_night_rate",
                "week_morning_rate","week_mc_rate","week_entropy"]

print(f"Haftalık tablo: {len(weekly_base):,} ATM-hafta | {weekly_base['atm_id'].nunique()} ATM")

def weekly_rzs(group, col):
    rm   = group[col].shift(1).rolling(WINDOW_W, min_periods=4).mean()
    rs   = group[col].shift(1).rolling(WINDOW_W, min_periods=4).std()
    safe = rs.where(rs >= MIN_STD, other=np.nan)
    z    = (group[col] - rm) / safe
    return z.fillna(0)

weekly_base = weekly_base.sort_values(["atm_id","week_start"])
for f in weekly_fcols:
    res = weekly_base.groupby("atm_id", group_keys=False).apply(lambda g: weekly_rzs(g, f))
    weekly_base[f"wz_{f}"] = res.values

wz_cols = [f"wz_{f}" for f in weekly_fcols]
weekly_base["w_n_flagged"]      = (weekly_base[wz_cols].abs() > Z_THR).sum(axis=1)
weekly_base["w_zscore_anomaly"] = (weekly_base["w_n_flagged"] >= 1).astype(int)
weekly_base["w_max_z"]          = weekly_base[wz_cols].abs().max(axis=1, skipna=True)

w_anom  = weekly_base["w_zscore_anomaly"].sum()
w_total = len(weekly_base)
print(f"Haftalık z-score anomali: {w_anom:,} / {w_total:,} (%{w_anom/w_total*100:.1f})")

atm15_w = weekly_base[weekly_base["atm_id"] == TARGET_ATM].sort_values("week_start")
print(f"\nATM 15 haftalık özet (son 10 hafta):")
cols_show = ["week_start","week_txn_total","week_txn_days","w_max_z","w_zscore_anomaly"]
print(atm15_w[cols_show].tail(10).to_string(index=False))

print(f"\nHaftalık en yüksek z-score'lu 5 ATM-hafta:")
top5_w = weekly_base.nlargest(5, "w_max_z")[
    ["atm_id","week_start","week_txn_total","w_max_z","w_n_flagged"]
].copy()
top5_w["week_start"] = top5_w["week_start"].dt.date
print(top5_w.to_string(index=False))

atm_weekly_anom = (
    weekly_base.groupby("atm_id")
    .agg(toplam_hafta=("week_start","count"),
         anomali_hafta=("w_zscore_anomaly","sum"))
    .reset_index()
)
atm_weekly_anom["pct"] = (atm_weekly_anom["anomali_hafta"] / atm_weekly_anom["toplam_hafta"] * 100).round(1)
print(f"\nHaftalık en çok anomali olan üst 10 ATM:")
print(atm_weekly_anom.sort_values("anomali_hafta", ascending=False).head(10).to_string(index=False))

# ── 7. AYLIK ÖZET (descriptive) ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("4. AYLIK ÖZET (2017 — yıllar arası doğrulama yapılamaz)")
print("=" * 65)

monthly = (
    daily_v3[daily_v3["low_txn"] == 0]
    .groupby(["atm_id","month"])
    .agg(
        month_txn_days = ("txn_count",      "count"),
        month_avg_txn  = ("txn_count",      "mean"),
        month_anomali  = ("zscore_anomaly", "sum"),
    )
    .reset_index()
)
monthly["anomali_pct"] = (monthly["month_anomali"] / monthly["month_txn_days"] * 100).round(1)

fleet_monthly = (
    monthly.groupby("month")
    .agg(ort_txn=("month_avg_txn","mean"), ort_anomali_pct=("anomali_pct","mean"))
    .reset_index()
)
month_names = {1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",
               7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
fleet_monthly["ay"] = fleet_monthly["month"].map(month_names)

print("Filo geneli aylık ortalama:")
print(f"{'Ay':<6} {'Ort txn/gün':>12} {'Ort anomali%':>14}")
print("-" * 35)
for _, row in fleet_monthly.iterrows():
    print(f"{row['ay']:<6} {row['ort_txn']:>12.1f} {row['ort_anomali_pct']:>14.1f}%")

atm15_m = monthly[monthly["atm_id"] == TARGET_ATM].copy()
atm15_m["ay"] = atm15_m["month"].map(month_names)
print(f"\nATM 15 aylık:")
print(atm15_m[["ay","month_txn_days","month_avg_txn","month_anomali","anomali_pct"]].to_string(index=False))

# ── 8. GRAFİKLER ─────────────────────────────────────────────────────────────
print("\nGrafikler üretiliyor...")

atm15_plot = daily_v3[(daily_v3["atm_id"]==TARGET_ATM) & (daily_v3["low_txn"]==0)].sort_values("date")
both_dates = valid_nona.loc[
    (valid_nona["atm_id"]==TARGET_ATM) &
    (valid_nona["zscore_anomaly"]==1) & (valid_nona["if_anomaly"]==1), "date"
].values

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for ax, (feat, col, lbl) in zip(axes, [
    ("txn_count",   "#1E3A5F", "Günlük İşlem Sayısı"),
    ("hour_entropy","#1A6B6B", "Saatlik Entropi"),
]):
    sub = atm15_plot.dropna(subset=[feat])
    # shift(1).rolling() — model baseline ile tutarlı; güncel gün kendi baseline'ına katılmaz
    rm  = sub[feat].shift(1).rolling(WINDOW_D, min_periods=7).mean()
    rs  = sub[feat].shift(1).rolling(WINDOW_D, min_periods=7).std().fillna(0)
    ax.fill_between(sub["date"], (rm-2*rs).clip(lower=0), rm+2*rs, alpha=0.15, color=col, label="±2σ bant")
    ax.plot(sub["date"], sub[feat], color=col, lw=1.1, label=lbl)
    ax.plot(sub["date"], rm, color=col, lw=1.5, ls="--", alpha=0.5, label="28g ort.")
    bth = sub[sub["date"].isin(both_dates)]
    ax.scatter(bth["date"], bth[feat], color="red", s=55, zorder=6, label="Anomali (Z+IF)")
    mend = sub[sub["calendar_context"] == "month_end_window"]
    for d_ in mend["date"]:
        ax.axvline(d_, color="gray", lw=0.5, alpha=0.3)
    if TARGET_DATE in sub["date"].values:
        val = sub.loc[sub["date"]==TARGET_DATE, feat].values[0]
        ax.axvline(TARGET_DATE, color="darkorange", lw=1.4, ls=":", alpha=0.85)
        ax.annotate("31 Ağu", (TARGET_DATE, val), textcoords="offset points",
                    xytext=(5,5), fontsize=8, color="darkorange", fontweight="bold")
    ax.set_ylabel(lbl, fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

axes[0].set_title(f"ATM {TARGET_ATM} — Günlük Profil (MIN_TXN={CHOSEN_MIN_TXN})\n"
                  "Gri dikey: month_end_window bağlamı | Turuncu: 31 Ağustos",
                  fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "10_v3_atm15_daily.png", dpi=120, bbox_inches="tight")
plt.close()
print("  10_v3_atm15_daily.png")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
atm15_wk = weekly_base[weekly_base["atm_id"]==TARGET_ATM].sort_values("week_start")
ax = axes[0]
ax.bar(atm15_wk["week_start"], atm15_wk["week_txn_total"], color="#1E3A5F", alpha=0.7, width=5)
wm = atm15_wk["week_txn_total"].shift(1).rolling(WINDOW_W, min_periods=4).mean()
ws = atm15_wk["week_txn_total"].shift(1).rolling(WINDOW_W, min_periods=4).std().fillna(0)
ax.plot(atm15_wk["week_start"], wm, color="orange", lw=1.5, label="8h ort.")
ax.fill_between(atm15_wk["week_start"], (wm-2*ws).clip(lower=0), wm+2*ws,
                alpha=0.15, color="orange", label="±2σ bant")
ax.set_ylabel("Haftalık Toplam İşlem", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_title(f"ATM {TARGET_ATM} — Haftalık Profil", fontsize=10, fontweight="bold")
ax = axes[1]
colors_w = ["#C0392B" if v > 0 else "#2980B9" for v in atm15_wk["wz_week_txn_total"]]
ax.bar(atm15_wk["week_start"], atm15_wk["wz_week_txn_total"].fillna(0),
       color=colors_w, alpha=0.75, width=5)
ax.axhline( Z_THR, color="red", lw=0.9, ls="--", alpha=0.7)
ax.axhline(-Z_THR, color="red", lw=0.9, ls="--", alpha=0.7)
ax.axhline(0, color="black", lw=0.7, alpha=0.4)
ax.set_ylabel("z_week_txn_total", fontsize=9)
ax.set_ylim(-5, 5)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "11_v3_atm15_weekly.png", dpi=120, bbox_inches="tight")
plt.close()
print("  11_v3_atm15_weekly.png")

atm_sum = (
    valid_nona.groupby("atm_id")
    .agg(toplam=("date","count"), anom=("zscore_anomaly","sum"))
    .reset_index()
)
atm_sum["pct"] = (atm_sum["anom"] / atm_sum["toplam"] * 100).round(1)
ps = atm_sum.sort_values("pct", ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
bar_c = ["#C0392B" if p > 20 else "#1E3A5F" for p in ps["pct"]]
ax.bar(ps["atm_id"].astype(str), ps["pct"], color=bar_c, alpha=0.8, width=0.75)
ax.axhline(ps["pct"].mean(), color="orange", lw=1.5, ls="--",
           label=f"Ort: %{ps['pct'].mean():.1f}")
ax.set_xlabel("ATM ID", fontsize=9)
ax.set_ylabel("Anomali Gün %", fontsize=9)
ax.set_title(f"Tüm Filo — Günlük Anomali Gün Oranı (MIN_TXN={CHOSEN_MIN_TXN})",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.tick_params(axis="x", labelsize=6, rotation=90)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "12_v3_fleet_anomaly.png", dpi=120, bbox_inches="tight")
plt.close()
print("  12_v3_fleet_anomaly.png")

atm_sum.to_csv(OUT_DIR / "atm_v3_summary.csv", index=False)
print("\n" + "="*65)
print(f"V3 TAMAMLANDI (MIN_TXN={CHOSEN_MIN_TXN})")
print("="*65)
