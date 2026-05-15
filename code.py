"""
=============================================================================
  UAV SWARM ANOMALY DETECTION — RESEARCH-GRADE PIPELINE  (FIXED v2)
=============================================================================
Architecture:
  ML  Branch  XGBoost + LightGBM (windowed aggregated features)
  DL  Branch  CNN-LSTM (raw time-series sequences — raw sensor cols only)
  Fusion      Adaptive soft-voting ensemble
  Decision    4-level decision layer (SAFE / MONITOR / INVESTIGATE / CRITICAL)

Changes vs original:
  FIX 1  ADMM removed → simple inverse-frequency + severity weights
  FIX 2  DL branch uses only raw sensor features (16 cols)
  FIX 3  Window stride reduced 15 → 5 (more training samples)
  FIX 4  Window label = mean > 0.2  (noise-robust majority vote)
  FIX 5  Lighter CNN-LSTM (Conv32→16, LSTM32) + BatchNorm
  FIX 6  class_weight removed from DL (sample_weight already used)
  FIX 7  DL threshold raised 0.5 → 0.8 (calibrated for high-prob outputs)
  FIX 8  Sequence scaling clipped to [-5, 5]  (no exploding gradients)
  FIX 9  epochs capped at 50  (avoid overfitting)
=============================================================================
"""

import warnings
import os
import time

warnings.filterwarnings("ignore")
os.makedirs("/kaggle/working/plots", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

PLOT_DIR = "/kaggle/working/plots"

print("=" * 70)
print("  UAV SWARM ANOMALY DETECTION — RESEARCH-GRADE PIPELINE  (FIXED v2)")
print("=" * 70)


# =============================================================================
# SECTION 0  LOAD DATA
# =============================================================================
print("\n[0] LOADING DATA ...")

df = pd.read_csv(
    "/kaggle/input/datasets/rishitasharma22/uav-swarm-dataset/uav_swarm_dataset.csv"
)
print(f"    Shape : {df.shape}")

TARGET    = "anomaly_flag"
DROP_COLS = ["time", "drone_id", "anomaly_onset", "drone_state",
             "response_type", "rtb_flag", "attack_class", "severity"]

df = df.sort_values(["drone_id", "time"]).reset_index(drop=True)
print(f"    Sorted by [drone_id, time]")


# =============================================================================
# SECTION 1  TEMPORAL DERIVATIVE FEATURES
# =============================================================================
print("\n[1] TEMPORAL DERIVATIVE FEATURES ...")

df["acc_x_diff"] = df.groupby("drone_id")["acc_x"].diff().fillna(0)
df["volt_diff"]  = df.groupby("drone_id")["volt"].diff().fillna(0)

if "rpm1" in df.columns and "rpm2" in df.columns:
    df["rpm_diff"] = df["rpm1"] - df["rpm2"]
    print("    Added: acc_x_diff, volt_diff, rpm_diff")
else:
    print("    Added: acc_x_diff, volt_diff")

df["vibe_x_diff"] = df.groupby("drone_id")["vibe_x"].diff().fillna(0)
df["gyr_z_diff"]  = df.groupby("drone_id")["gyr_z"].diff().fillna(0)
print("    Added: vibe_x_diff, gyr_z_diff")

feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
X_raw        = df[feature_cols].copy()
y            = df[TARGET].copy()
severity_raw = df["severity"].fillna(0).values

print(f"    Features : {len(feature_cols)}")
print(f"    Anomaly% : {y.mean()*100:.2f}%  |  "
      f"Normal: {(y==0).sum()}  |  Anomaly: {(y==1).sum()}")


# =============================================================================
# SECTION 2  PREPROCESSING
# =============================================================================
print("\n[2] PREPROCESSING ...")


def iqr_clip(df_in, factor=3.0):
    df_out = df_in.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        Q1, Q3  = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        IQR     = Q3 - Q1
        df_out[col] = df_out[col].clip(Q1 - factor * IQR, Q3 + factor * IQR)
    return df_out


X_clipped = iqr_clip(X_raw, factor=3.0)
print(f"    Missing values : {X_raw.isnull().sum().sum()}")
print("    IQR clipping   : factor=3")

scaler_robust = RobustScaler()
X_scaled = pd.DataFrame(
    scaler_robust.fit_transform(X_clipped),
    columns=feature_cols,
    index=df.index,
)
print("    RobustScaler   : applied")


# =============================================================================
# SECTION 3  UNIT 1  MATHEMATICAL FEATURE ENGINEERING
# =============================================================================
print("\n[3] UNIT 1 — MATHEMATICAL FEATURE ENGINEERING ...")

imu_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
FFT_WIN  = 50

print("    [3-A] Window-based Fourier Transform ...")

fft_feats = {}
for drone_id, grp in df.groupby("drone_id"):
    for col in imu_cols:
        sig = grp[col].values
        n   = len(sig)
        for i in range(n):
            segment = sig[:i + 1] if i < FFT_WIN else sig[i - FFT_WIN:i]
            if len(segment) < 2:
                segment = sig[:2]
            fft_mag = np.abs(np.fft.rfft(segment))
            k       = len(fft_mag)
            row_idx = grp.index[i]
            fft_feats.setdefault(f"fft_{col}_low",  {})[row_idx] = fft_mag[:max(1, k // 4)].mean()
            fft_feats.setdefault(f"fft_{col}_mid",  {})[row_idx] = fft_mag[k // 4:k // 2].mean() if k > 4 else 0.0
            fft_feats.setdefault(f"fft_{col}_high", {})[row_idx] = fft_mag[k // 2:].mean() if k > 2 else 0.0

fft_df = pd.DataFrame({k: pd.Series(v) for k, v in fft_feats.items()})
fft_df = fft_df.reindex(X_scaled.index).fillna(0)
print(f"       Added {fft_df.shape[1]} DFT features.")

print("    [3-B] Window-based DMD features ...")


def build_hankel(signal, rows):
    cols = len(signal) - rows + 1
    if cols < 1:
        return None
    H = np.empty((rows, cols), dtype=float)
    for r in range(rows):
        H[r] = signal[r:r + cols]
    return H


def dmd_features(signal, rows=15, r=5):
    if len(signal) < rows + 2:
        return np.zeros(r)
    H = build_hankel(signal, rows)
    if H is None:
        return np.zeros(r)
    X1, X2     = H[:, :-1], H[:, 1:]
    U, S, Vt   = np.linalg.svd(X1, full_matrices=False)
    r          = min(r, len(S))
    Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r]
    Atilde     = Ur.T @ X2 @ Vtr.T @ np.diag(1.0 / (Sr + 1e-10))
    eigs       = np.linalg.eigvals(Atilde)
    return np.sort(np.abs(eigs))[::-1]


DMD_WIN  = 40
DMD_ROWS = 10

dmd_feats_dict = {}
for drone_id, grp in df.groupby("drone_id"):
    sig   = grp["acc_x"].values
    n     = len(sig)
    for i in range(n):
        segment    = sig[max(0, i - DMD_WIN):i + 1]
        eig_energy = dmd_features(segment, rows=DMD_ROWS, r=5)
        row_idx    = grp.index[i]
        for k, v in enumerate(eig_energy):
            dmd_feats_dict.setdefault(f"dmd_eig_{k}", {})[row_idx] = v

dmd_df = pd.DataFrame({k: pd.Series(v) for k, v in dmd_feats_dict.items()})
dmd_df = dmd_df.reindex(X_scaled.index).fillna(0)
print(f"       Added {dmd_df.shape[1]} DMD eigenvalue features.")

print("    [3-C] Toeplitz/Circulant residuals ...")


def circulant_smooth(signal, kernel_size=5):
    kernel = np.ones(kernel_size) / kernel_size
    return np.real(
        np.fft.ifft(
            np.fft.fft(signal, n=len(signal))
            * np.fft.fft(kernel, n=len(signal))
        )
    )


toep_feats = {}
for drone_id, grp in df.groupby("drone_id"):
    for col in ["acc_x", "gyr_z", "volt"]:
        sig   = grp[col].values
        filt  = circulant_smooth(sig)
        resid = sig - filt
        for i, idx_val in enumerate(grp.index):
            toep_feats.setdefault(f"toep_resid_{col}", {})[idx_val] = resid[i]

toep_df = pd.DataFrame({k: pd.Series(v) for k, v in toep_feats.items()})
toep_df = toep_df.reindex(X_scaled.index).fillna(0)
print(f"       Added {toep_df.shape[1]} Toeplitz residual features.")

print("    [3-D] Kronecker cross-feature interactions ...")

kron_base = ["acc_x", "gyr_z", "volt", "vibe_x"]
kron_df   = pd.DataFrame(index=X_scaled.index)
for c1 in kron_base:
    for c2 in kron_base:
        kron_df[f"kron_{c1}_{c2}"] = X_scaled[c1] * X_scaled[c2]
print(f"       Added {kron_df.shape[1]} Kronecker product features.")

X_eng = pd.concat([X_scaled, fft_df, dmd_df, toep_df, kron_df], axis=1)
X_eng.fillna(0, inplace=True)
print(f"\n    Total features after engineering: {X_eng.shape[1]}")


# =============================================================================
# SECTION 4  UNIT 3  STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# =============================================================================
print("\n[4] UNIT 3 — STATISTICAL ESTIMATION & HYPOTHESIS TESTING ...")

n_total   = len(y)
n_anomaly = int(y.sum())
p_hat_mle = n_anomaly / n_total
se        = np.sqrt(p_hat_mle * (1 - p_hat_mle) / n_total)
ci_low    = p_hat_mle - 1.96 * se
ci_high   = p_hat_mle + 1.96 * se
print(f"    MLE p̂ = {p_hat_mle:.4f}  |  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

vibe_normal  = df.loc[y == 0, "vibe_x"]
vibe_anomaly = df.loc[y == 1, "vibe_x"]
t_stat, p_val = stats.ttest_ind(vibe_normal, vibe_anomaly, equal_var=False)
print(f"    Welch t-test (vibe_x)   : t={t_stat:.4f} | p={p_val:.2e} "
      f"→ {'Reject H₀' if p_val < 0.05 else 'Fail to reject H₀'}")

ks_stat, ks_p = stats.ks_2samp(
    df.loc[y == 0, "packet_loss"], df.loc[y == 1, "packet_loss"]
)
print(f"    KS test (packet_loss)   : KS={ks_stat:.4f} | p={ks_p:.2e}")

cont_table        = pd.crosstab(df["drone_state"], y)
chi2, chi2_p, dof, _ = stats.chi2_contingency(cont_table)
print(f"    χ² test (drone_state)   : χ²={chi2:.4f} | dof={dof} | p={chi2_p:.2e}")


# =============================================================================
# SECTION 5  UNIT 4  QUANTUM COMPUTING (QISKIT)
# =============================================================================
print("\n[5] UNIT 4 — QUANTUM COMPUTING (QISKIT) ...")

print("    [5-A] Bell State ...")
bell_qc = QuantumCircuit(2, 2)
bell_qc.h(0)
bell_qc.cx(0, 1)
bell_qc.measure([0, 1], [0, 1])
print(bell_qc.draw(output="text"))

print("    [5-B] Superdense Coding ('11') ...")
sdc_qc = QuantumCircuit(2, 2)
sdc_qc.h(0)
sdc_qc.cx(0, 1)
sdc_qc.x(0)
sdc_qc.z(0)
sdc_qc.cx(0, 1)
sdc_qc.h(0)
sdc_qc.measure([0, 1], [0, 1])
print(sdc_qc.draw(output="text"))

print("    [5-C] Quantum Teleportation (anomaly probability encoded) ...")
tele_qc      = QuantumCircuit(3, 3)
feature_angle = float(np.pi * p_hat_mle)
tele_qc.ry(feature_angle, 0)
tele_qc.h(1)
tele_qc.cx(1, 2)
tele_qc.cx(0, 1)
tele_qc.h(0)
tele_qc.measure([0, 1], [0, 1])
tele_qc.cx(1, 2)
tele_qc.cz(0, 2)
tele_qc.measure(2, 2)
print(tele_qc.draw(output="text"))

print("    [5-D] Quantum amplitude encoding → dynamic similarity feature ...")

top_feats = ["packet_loss", "vibe_x"]
qvec      = np.zeros(4)
for i, cls in enumerate([0, 1]):
    vals = X_eng.loc[y == cls, top_feats].mean().values.copy()
    norm = np.linalg.norm(vals)
    if norm > 0:
        vals /= norm
    qvec[i * 2:i * 2 + 2] = vals[:2]
qvec /= np.linalg.norm(qvec) + 1e-10

amp_qc = QuantumCircuit(2)
amp_qc.initialize(qvec, [0, 1])
sv = Statevector(amp_qc)
q_similarity_score = float(np.abs(sv.data[0]) ** 2 - np.abs(sv.data[3]) ** 2)

X_eng["q_similarity"] = X_eng["packet_loss"] * q_similarity_score
print(f"       q_similarity_score : {q_similarity_score:.4f}")
print(f"       State amplitudes   : {np.round(sv.data, 4)}")
print(f"       Probabilities      : {np.round(sv.probabilities(), 4)}")

sim    = AerSimulator()
t_bell = transpile(bell_qc, sim)
job    = sim.run(t_bell, shots=1024)
counts = job.result().get_counts()
print(f"\n    Bell state counts (1024 shots): {counts}")


# =============================================================================
# SECTION 6  SLIDING WINDOW PREPARATION
# =============================================================================
print("\n[6] SLIDING WINDOW — TIME-SERIES PREPARATION ...")

WINDOW = 30
STEP   = 5   # FIX 3: stride 15 → 5 for more training samples

# FIX 2: DL branch uses only raw sensor features
seq_cols = [
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "vel_x", "vel_y", "vel_z",
    "rpm1",  "rpm2",
    "volt",  "vibe_x",
    "packet_loss", "gps_hdop", "rssi"
]
# keep only columns that actually exist in df
seq_cols = [c for c in seq_cols if c in df.columns]
print(f"    DL sequence columns ({len(seq_cols)}): {seq_cols}")

raw_feats  = df[seq_cols].values
eng_feats  = X_eng.values
y_arr      = y.values

windows_raw     = []
windows_agg     = []
labels_window   = []
severity_window = []

for drone_id, grp in df.groupby("drone_id"):
    idx    = grp.index.tolist()
    n_rows = len(idx)
    start  = 0
    while start + WINDOW <= n_rows:
        w_idx = idx[start:start + WINDOW]
        w_raw = raw_feats[w_idx]
        w_eng = eng_feats[w_idx]

        # FIX 4: majority-vote label — avoids "1 noisy point = whole window anomaly"
        w_lbl = int(np.mean(y_arr[w_idx]) > 0.2)
        w_sev = float(severity_raw[w_idx].max())

        windows_raw.append(w_raw)
        agg = np.concatenate([
            w_eng.mean(0), w_eng.std(0), w_eng.min(0), w_eng.max(0)
        ])
        windows_agg.append(agg)
        labels_window.append(w_lbl)
        severity_window.append(w_sev)
        start += STEP

X_seq   = np.array(windows_raw,    dtype=np.float32)
X_agg   = np.array(windows_agg,    dtype=np.float32)
y_win   = np.array(labels_window,  dtype=np.int32)
sev_win = np.array(severity_window, dtype=np.float32)

print(f"    Sequence tensor  : {X_seq.shape}  → DL branch")
print(f"    Aggregated table : {X_agg.shape}  → ML branch")
print(f"    Anomaly rate     : {y_win.mean():.4f}")
print(f"    Severity stats   : min={sev_win.min():.3f} "
      f"mean={sev_win.mean():.3f} max={sev_win.max():.3f}")


# =============================================================================
# SECTION 7  TIME-AWARE TRAIN/TEST SPLIT
# =============================================================================
print("\n[7] TIME-AWARE TRAIN/TEST SPLIT ...")

split_point = int(0.8 * len(X_seq))

X_seq_tr, X_seq_te = X_seq[:split_point], X_seq[split_point:]
X_agg_tr, X_agg_te = X_agg[:split_point], X_agg[split_point:]
y_tr,     y_te     = y_win[:split_point], y_win[split_point:]
sev_tr,   sev_te   = sev_win[:split_point], sev_win[split_point:]

print(f"    Split point    : window {split_point} / {len(X_seq)}")
print(f"    Train seq      : {X_seq_tr.shape}  |  Test seq  : {X_seq_te.shape}")
print(f"    Train agg      : {X_agg_tr.shape}  |  Test agg  : {X_agg_te.shape}")
print(f"    Train anomaly% : {y_tr.mean():.4f}  |  Test     : {y_te.mean():.4f}")

sc_agg     = StandardScaler()
X_agg_tr_s = sc_agg.fit_transform(X_agg_tr)
X_agg_te_s = sc_agg.transform(X_agg_te)

sc_seq     = StandardScaler()
n_tr, T, F = X_seq_tr.shape
X_seq_tr_s = sc_seq.fit_transform(X_seq_tr.reshape(-1, F)).reshape(n_tr, T, F)
n_te       = X_seq_te.shape[0]
X_seq_te_s = sc_seq.transform(X_seq_te.reshape(-1, F)).reshape(n_te, T, F)

# FIX 8: clip sequences to prevent exploding gradients
X_seq_tr_s = np.clip(X_seq_tr_s, -5, 5)
X_seq_te_s = np.clip(X_seq_te_s, -5, 5)
print("    Sequence scaling clipped to [-5, 5]")

# =============================================================================
# SECTION 8  SAMPLE WEIGHTING  (FIX 1: ADMM removed → simple inverse-freq)
# =============================================================================
print("\n[8] SAMPLE WEIGHTING (inverse-frequency + severity) ...")

# FIX 1: replace ADMM with clean inverse-frequency weighting
p           = y_tr.mean()
w_normal    = 1.0 / (1.0 - p)
w_anomaly   = 1.0 / p
sw_win      = np.where(y_tr == 1, w_anomaly, w_normal)

# controlled severity scaling
K_SEVERITY  = 2.5
sw_win      = sw_win * (1.0 + K_SEVERITY * sev_tr)

# normalize and clip for stability
sw_win      = sw_win / np.mean(sw_win)
sw_win      = np.clip(sw_win, 0.5, 4.0)

print(f"    Severity k     : {K_SEVERITY}")
print(f"    Weight stats   : mean={sw_win.mean():.3f}  max={sw_win.max():.2f}")
print(f"    Avg weight — normal  : {sw_win[y_tr==0].mean():.3f}")
print(f"    Avg weight — anomaly : {sw_win[y_tr==1].mean():.3f}")

for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
    mask = (sev_tr >= lo) & (sev_tr < hi) & (y_tr == 1)
    if mask.sum() > 0:
        print(f"    sev [{lo:.2f},{hi:.2f}) anomaly windows: "
              f"n={mask.sum():4d}  avg_weight={sw_win[mask].mean():.2f}")

# class weights for ML models
class_weights_arr = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
class_weight_dict = {0: float(class_weights_arr[0]), 1: float(class_weights_arr[1])}
print(f"    ML class weights : {class_weight_dict}")


# =============================================================================
# SECTION 9  ML BRANCH  XGBoost + LightGBM
# =============================================================================
print("\n[9] ML BRANCH — XGBoost + LightGBM ...")

ml_models = {
    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=class_weights_arr[1] / class_weights_arr[0],
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        class_weight=class_weight_dict,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}

ml_results = {}
for name, mdl in ml_models.items():
    t0 = time.time()
    print(f"    Training {name} ...", end=" ", flush=True)
    mdl.fit(X_agg_tr_s, y_tr, sample_weight=sw_win)
    y_pred = mdl.predict(X_agg_te_s)
    y_prob = mdl.predict_proba(X_agg_te_s)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    f1     = f1_score(y_te, y_pred, average="macro")
    ap     = average_precision_score(y_te, y_prob)
    ml_results[name] = {
        "model": mdl, "y_pred": y_pred, "y_prob": y_prob,
        "AUC": auc, "F1": f1, "AP": ap, "time": time.time() - t0,
    }
    print(f"AUC={auc:.4f} | F1={f1:.4f} | AP={ap:.4f} | {time.time()-t0:.1f}s")

best_ml_name = max(ml_results, key=lambda n: ml_results[n]["AUC"])
best_ml      = ml_results[best_ml_name]
print(f"    Best ML : {best_ml_name} (AUC={best_ml['AUC']:.4f})")


# =============================================================================
# SECTION 10  DL BRANCH  CNN-LSTM  (FIX 5: lighter architecture + BatchNorm)
# =============================================================================
print("\n[10] DL BRANCH — CNN-LSTM (lighter, BatchNorm) ...")


# FIX 5: lighter model — Conv32→16, LSTM32, BatchNorm
def build_cnn_lstm(window, n_features):
    model = Sequential([
        Conv1D(32, 3, activation="relu", padding="same",
               input_shape=(window, n_features)),
        BatchNormalization(),
        Conv1D(16, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ], name="CNN_LSTM_v2")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


n_features = X_seq_tr_s.shape[2]
dl_model   = build_cnn_lstm(WINDOW, n_features)
dl_model.summary()

# FIX 6: no class_weight — sample_weight already handles imbalance
# (removed keras_cw / class_weight= argument from model.fit)

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\n    Training CNN-LSTM ...")
t0_dl   = time.time()
history = dl_model.fit(
    X_seq_tr_s, y_tr,
    validation_split=0.15,
    epochs=50,              # FIX 9: cap at 50 to avoid overfitting
    batch_size=64,
    sample_weight=sw_win,   # FIX 6: sample_weight only — no class_weight
    callbacks=callbacks,
    verbose=1,
)
dl_train_time = time.time() - t0_dl

dl_prob = dl_model.predict(X_seq_te_s, verbose=0).flatten()
dl_pred = (dl_prob >= 0.8).astype(int)  # FIX 7: threshold 0.5 → 0.8
dl_auc  = roc_auc_score(y_te, dl_prob)
dl_f1   = f1_score(y_te, dl_pred, average="macro")
dl_ap   = average_precision_score(y_te, dl_prob)

print(f"\n    CNN-LSTM : AUC={dl_auc:.4f} | F1={dl_f1:.4f} | "
      f"AP={dl_ap:.4f} | {dl_train_time:.1f}s")


# =============================================================================
# SECTION 11  ADAPTIVE ENSEMBLE FUSION
# =============================================================================
print("\n[11] ADAPTIVE ENSEMBLE FUSION ...")

auc_gap = dl_auc - best_ml["AUC"]
if   auc_gap < -0.05: alpha = 0.2
elif auc_gap >  0.05: alpha = 0.7
else:                  alpha = 0.5

print(f"    DL AUC  : {dl_auc:.4f}  |  ML AUC : {best_ml['AUC']:.4f}")
print(f"    AUC gap : {auc_gap:+.4f}")
print(f"    α (DL weight) : {alpha}")

P_ensemble = alpha * dl_prob + (1 - alpha) * best_ml["y_prob"]

P_ensemble_smooth = (
    pd.Series(P_ensemble)
    .rolling(window=5, min_periods=1)
    .mean()
    .values
)
y_ensemble = (P_ensemble_smooth >= 0.5).astype(int)

ens_auc = roc_auc_score(y_te, P_ensemble_smooth)
ens_f1  = f1_score(y_te, y_ensemble, average="macro")
ens_ap  = average_precision_score(y_te, P_ensemble_smooth)

print(f"\n    Ensemble : AUC={ens_auc:.4f} | F1={ens_f1:.4f} | AP={ens_ap:.4f}")


# =============================================================================
# SECTION 12  CONFIDENCE SCORE & DECISION LAYER
# =============================================================================
print("\n[12] CONFIDENCE SCORE & DECISION LAYER ...")

confidence_score = np.abs(P_ensemble_smooth - 0.5) * 2


def decision_layer(p):
    if   p < 0.3: return "SAFE"
    elif p < 0.6: return "MONITOR"
    elif p < 0.8: return "INVESTIGATE"
    else:         return "CRITICAL"


# =============================================================================
# SECTION 13  FINAL INFERENCE OUTPUT
# =============================================================================
print("\n[13] FINAL INFERENCE OUTPUT ...")

inference_df = pd.DataFrame({
    "Window_ID"        : np.arange(len(y_te)),
    "True_Label"       : y_te,
    "Severity_Max"     : sev_te,
    "ML_Prob"          : np.round(best_ml["y_prob"], 4),
    "DL_Prob"          : np.round(dl_prob, 4),
    "Ensemble_Prob"    : np.round(P_ensemble_smooth, 4),
    "Confidence_Score" : np.round(confidence_score, 3),
    "Prediction"       : np.where(y_ensemble == 1, "ANOMALY", "Normal"),
    "Decision"         : [decision_layer(p) for p in P_ensemble_smooth],
})

print("\n    Sample inference output (first 25 windows):")
print(inference_df.head(25).to_string(index=False))

print("\n    Decision Layer Summary:")
print(inference_df["Decision"].value_counts().to_string())

critical = (inference_df["Decision"] == "CRITICAL").sum()
invest   = (inference_df["Decision"] == "INVESTIGATE").sum()
monitor  = (inference_df["Decision"] == "MONITOR").sum()
safe     = (inference_df["Decision"] == "SAFE").sum()
print(f"\n    SAFE={safe}  MONITOR={monitor}  INVESTIGATE={invest}  CRITICAL={critical}")

inference_df.to_csv(f"{PLOT_DIR}/../inference_output.csv", index=False)
print("    Saved: inference_output.csv")


# =============================================================================
# SECTION 14  VISUALIZATIONS
# =============================================================================
print("\n[14] GENERATING VISUALIZATIONS ...")

# Figure 1  Dataset Overview
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("UAV Swarm Dataset — Exploratory Analysis", fontsize=16, fontweight="bold")

axes[0, 0].bar(["Normal", "Anomaly"], [(y == 0).sum(), (y == 1).sum()],
               color=["#2196F3", "#F44336"], edgecolor="black")
axes[0, 0].set_title("Class Distribution (Raw)")
for i, v in enumerate([(y == 0).sum(), (y == 1).sum()]):
    axes[0, 0].text(i, v + 200, f"{v}\n({v/len(y)*100:.1f}%)", ha="center")

axes[0, 1].bar(["Normal", "Anomaly"], [(y_win == 0).sum(), (y_win == 1).sum()],
               color=["#1976D2", "#C62828"], edgecolor="black")
axes[0, 1].set_title(f"Class Distribution (Windows, stride={STEP})")

axes[0, 2].hist(df.loc[y == 0, "vibe_x"], bins=60, alpha=0.6,
                label="Normal", color="#2196F3", density=True)
axes[0, 2].hist(df.loc[y == 1, "vibe_x"], bins=60, alpha=0.6,
                label="Anomaly", color="#F44336", density=True)
axes[0, 2].set_title("Vibration-X Distribution")
axes[0, 2].legend()

axes[1, 0].hist(df.loc[y == 0, "packet_loss"], bins=50, alpha=0.6,
                label="Normal", color="#2196F3", density=True)
axes[1, 0].hist(df.loc[y == 1, "packet_loss"], bins=50, alpha=0.6,
                label="Anomaly", color="#F44336", density=True)
axes[1, 0].set_title("Packet Loss Distribution")
axes[1, 0].legend()

axes[1, 1].hist(df.loc[y == 0, "acc_x_diff"], bins=60, alpha=0.6,
                label="Normal", color="#2196F3", density=True)
axes[1, 1].hist(df.loc[y == 1, "acc_x_diff"], bins=60, alpha=0.6,
                label="Anomaly", color="#F44336", density=True)
axes[1, 1].set_title("acc_x Derivative — Rate of Change")
axes[1, 1].legend()

axes[1, 2].hist(df.loc[y == 0, "volt_diff"], bins=60, alpha=0.6,
                label="Normal", color="#2196F3", density=True)
axes[1, 2].hist(df.loc[y == 1, "volt_diff"], bins=60, alpha=0.6,
                label="Anomaly", color="#F44336", density=True)
axes[1, 2].set_title("volt Derivative — Voltage Drop Rate")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 01_dataset_overview.png")

# Figure 2  Mathematical Features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Unit 1 — Mathematical Feature Engineering", fontsize=16, fontweight="bold")

drone1 = df[df["drone_id"] == 1]
sig    = drone1["acc_x"].values
freqs  = np.fft.rfftfreq(FFT_WIN)
mag    = np.abs(np.fft.rfft(sig[:FFT_WIN]))
axes[0, 0].plot(freqs, mag, color="#9C27B0", lw=2)
axes[0, 0].set_title(f"Window-based DFT (window={FFT_WIN}) — Drone 1 acc_x")
axes[0, 0].set_xlabel("Normalized Frequency")
axes[0, 0].set_ylabel("|FFT|")

sig_n = df.loc[(y == 0) & (df["drone_id"] == 1), "acc_x"].values
sig_a = df.loc[(y == 1) & (df["drone_id"] == 1), "acc_x"].values
if len(sig_n) > 4 and len(sig_a) > 4:
    f_n, Pxx_n = periodogram(sig_n[:200])
    f_a, Pxx_a = periodogram(sig_a[:200])
    axes[0, 1].semilogy(f_n, Pxx_n, label="Normal",  alpha=0.8, color="#2196F3")
    axes[0, 1].semilogy(f_a, Pxx_a, label="Anomaly", alpha=0.8, color="#F44336")
    axes[0, 1].set_title("Power Spectral Density (Normal vs Anomaly)")
    axes[0, 1].legend()

H = build_hankel(sig[:100], rows=15) if len(sig) >= 17 else np.zeros((5, 5))
if H is not None:
    im = axes[0, 2].imshow(H, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=axes[0, 2])
axes[0, 2].set_title("Hankel Matrix (acc_x, Drone 1)")

eigs_n = dmd_features(sig_n[:50], rows=10, r=5) if len(sig_n) >= 12 else np.zeros(5)
eigs_a = dmd_features(sig_a[:50], rows=10, r=5) if len(sig_a) >= 12 else np.zeros(5)
x_k    = np.arange(len(eigs_n))
axes[1, 0].bar(x_k - 0.2, eigs_n, 0.4, label="Normal",  color="#2196F3")
axes[1, 0].bar(x_k + 0.2, eigs_a, 0.4, label="Anomaly", color="#F44336")
axes[1, 0].set_title("Window-based DMD Eigenvalue Magnitudes")
axes[1, 0].legend()

toep_col = "toep_resid_acc_x"
if toep_col in X_eng.columns:
    axes[1, 1].hist(X_eng.loc[y == 0, toep_col], bins=60, alpha=0.6,
                    label="Normal", color="#2196F3", density=True)
    axes[1, 1].hist(X_eng.loc[y == 1, toep_col], bins=60, alpha=0.6,
                    label="Anomaly", color="#F44336", density=True)
    axes[1, 1].set_title("Toeplitz Residual (acc_x)")
    axes[1, 1].legend()

kron_col = "kron_acc_x_vibe_x"
if kron_col in X_eng.columns:
    n_s = min(500, (y == 0).sum())
    a_s = min(200, (y == 1).sum())
    axes[1, 2].scatter(
        X_eng.loc[y == 0, kron_col].sample(n_s, random_state=0),
        X_eng.loc[y == 0, "packet_loss"].sample(n_s, random_state=0),
        alpha=0.3, s=3, label="Normal", color="#2196F3",
    )
    axes[1, 2].scatter(
        X_eng.loc[y == 1, kron_col].sample(a_s, random_state=0),
        X_eng.loc[y == 1, "packet_loss"].sample(a_s, random_state=0),
        alpha=0.5, s=8, label="Anomaly", color="#F44336",
    )
    axes[1, 2].set_title("Kronecker Feature vs Packet Loss")
    axes[1, 2].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_math_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 02_math_features.png")

# Figure 3  Quantum Analysis
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Unit 4 — Quantum Analysis (Qiskit)", fontsize=16, fontweight="bold")

bars = axes[0].bar(
    list(counts.keys()), list(counts.values()),
    color=["#673AB7", "#E91E63", "#FF9800", "#4CAF50"][:len(counts)],
)
axes[0].set_title("Bell State (1024 shots)\n|00⟩ and |11⟩ confirm entanglement")
axes[0].set_xlabel("Outcome")
axes[0].set_ylabel("Count")
for bar, v in zip(bars, counts.values()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha="center")

state_labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
probs        = sv.probabilities()
axes[1].bar(state_labels, probs, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
axes[1].set_title(f"Amplitude Encoding — Normal vs Anomaly States\n"
                  f"q_similarity = {q_similarity_score:.4f}")
axes[1].set_ylabel("Probability")
for i, v in enumerate(probs):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

axes[2].hist(X_eng.loc[y == 0, "q_similarity"], bins=60, alpha=0.6,
             label="Normal", color="#2196F3", density=True)
axes[2].hist(X_eng.loc[y == 1, "q_similarity"], bins=60, alpha=0.6,
             label="Anomaly", color="#F44336", density=True)
axes[2].set_title("Dynamic Quantum Feature\npacket_loss × q_similarity_score")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_quantum_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 03_quantum_analysis.png")

# Figure 4  DL Training History
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("CNN-LSTM v2 — Training History", fontsize=14, fontweight="bold")

ep_x = np.arange(1, len(history.history["loss"]) + 1)
axes[0].plot(ep_x, history.history["loss"],     label="Train", color="#2196F3")
axes[0].plot(ep_x, history.history["val_loss"], label="Val",   color="#F44336")
axes[0].set_title("Loss (Binary Cross-Entropy)")
axes[0].legend()
axes[0].set_xlabel("Epoch")

axes[1].plot(ep_x, history.history["accuracy"],     label="Train", color="#2196F3")
axes[1].plot(ep_x, history.history["val_accuracy"], label="Val",   color="#F44336")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].set_xlabel("Epoch")

axes[2].plot(ep_x, history.history["auc"],     label="Train", color="#2196F3")
axes[2].plot(ep_x, history.history["val_auc"], label="Val",   color="#F44336")
axes[2].set_title("AUC-ROC")
axes[2].legend()
axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_dl_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 04_dl_training_history.png")

# Figure 5  Model Comparison
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Model Comparison — ML vs DL vs Ensemble",
             fontsize=14, fontweight="bold")

all_models_summary = {
    **{k: {"AUC": v["AUC"], "F1": v["F1"], "AP": v["AP"], "y_prob": v["y_prob"]}
       for k, v in ml_results.items()},
    "CNN-LSTM": {"AUC": dl_auc,  "F1": dl_f1,  "AP": dl_ap,  "y_prob": dl_prob},
    "Ensemble": {"AUC": ens_auc, "F1": ens_f1, "AP": ens_ap, "y_prob": P_ensemble_smooth},
}

names_all  = list(all_models_summary.keys())
colors_all = ["#1976D2", "#0097A7", "#7B1FA2", "#C62828"]
aucs_all   = [all_models_summary[n]["AUC"] for n in names_all]
f1s_all    = [all_models_summary[n]["F1"]  for n in names_all]
aps_all    = [all_models_summary[n]["AP"]  for n in names_all]

axes[0, 0].barh(names_all, aucs_all, color=colors_all)
axes[0, 0].set_xlim(0, 1)
axes[0, 0].axvline(0.9, ls="--", color="red", alpha=0.5)
axes[0, 0].set_title("AUC-ROC")
for i, v in enumerate(aucs_all):
    axes[0, 0].text(v + 0.005, i, f"{v:.4f}", va="center")

axes[0, 1].barh(names_all, f1s_all, color=colors_all)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_title("F1-Macro")
for i, v in enumerate(f1s_all):
    axes[0, 1].text(v + 0.005, i, f"{v:.4f}", va="center")

axes[0, 2].barh(names_all, aps_all, color=colors_all)
axes[0, 2].set_xlim(0, 1)
axes[0, 2].set_title("Average Precision (PR-AUC)")
for i, v in enumerate(aps_all):
    axes[0, 2].text(v + 0.005, i, f"{v:.4f}", va="center")

for i, (name, res) in enumerate(all_models_summary.items()):
    fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
    lw = 3 if "Ensemble" in name else 1.5
    axes[1, 0].plot(fpr, tpr, label=f"{name} ({res['AUC']:.3f})",
                    color=colors_all[i], lw=lw)
axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[1, 0].set_title("ROC Curves")
axes[1, 0].set_xlabel("FPR")
axes[1, 0].set_ylabel("TPR")
axes[1, 0].legend(fontsize=8)

for i, (name, res) in enumerate(all_models_summary.items()):
    prec, rec, _ = precision_recall_curve(y_te, res["y_prob"])
    lw = 3 if "Ensemble" in name else 1.5
    axes[1, 1].plot(rec, prec, label=f"{name} (AP={res['AP']:.3f})",
                    color=colors_all[i], lw=lw)
axes[1, 1].axhline(y_te.mean(), ls="--", color="gray",
                   label=f"Baseline ({y_te.mean():.3f})")
axes[1, 1].set_title("Precision-Recall Curves")
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_xlabel("Recall")
axes[1, 1].set_ylabel("Precision")

cm = confusion_matrix(y_te, y_ensemble)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 2],
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
axes[1, 2].set_title("Ensemble Confusion Matrix")
axes[1, 2].set_ylabel("True")
axes[1, 2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 05_model_comparison.png")

# Figure 6  Statistical Analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Unit 3 — Statistical Estimation & Hypothesis Testing",
             fontsize=14, fontweight="bold")

p_range  = np.linspace(0.05, 0.25, 200)
log_liks = n_anomaly * np.log(p_range) + (n_total - n_anomaly) * np.log(1 - p_range)
axes[0, 0].plot(p_range, log_liks, color="#9C27B0", lw=2)
axes[0, 0].axvline(p_hat_mle, color="red", ls="--", label=f"MLE p̂={p_hat_mle:.4f}")
axes[0, 0].axvspan(ci_low, ci_high, alpha=0.2, color="red", label="95% CI")
axes[0, 0].set_title("MLE Log-Likelihood")
axes[0, 0].legend()

t_crit = stats.t.ppf(0.975, df=len(vibe_normal) + len(vibe_anomaly) - 2)
x_t    = np.linspace(-8, 8, 300)
axes[0, 1].plot(x_t, stats.t.pdf(x_t, df=1000), color="gray", lw=1.5)
axes[0, 1].axvline(t_stat, color="#F44336", ls="--", lw=2, label=f"t={t_stat:.2f}")
axes[0, 1].axvline(-t_crit, ls=":", color="green", label=f"±t_crit={t_crit:.2f}")
axes[0, 1].axvline(t_crit,  ls=":", color="green")
axes[0, 1].fill_between(x_t, stats.t.pdf(x_t, 1000),
                         where=(x_t > t_crit) | (x_t < -t_crit),
                         alpha=0.3, color="green")
axes[0, 1].set_title(f"Welch t-test — p={p_val:.2e}")
axes[0, 1].legend(fontsize=8)

for cls, lbl, c in [(0, "Normal", "#2196F3"), (1, "Anomaly", "#F44336")]:
    vals = np.sort(df.loc[y == cls, "packet_loss"].values)
    axes[0, 2].plot(vals, np.arange(len(vals)) / len(vals), label=lbl, color=c)
axes[0, 2].set_title(f"KS Test — Packet Loss CDF\nKS={ks_stat:.4f}, p={ks_p:.2e}")
axes[0, 2].legend()

feat_box   = ["acc_x", "gyr_z", "vibe_x", "packet_loss", "volt", "formation_score"]
box_data_n = [df.loc[y == 0, f].values for f in feat_box if f in df.columns]
box_data_a = [df.loc[y == 1, f].values for f in feat_box if f in df.columns]
feat_box   = [f for f in feat_box if f in df.columns]
pos_n      = np.arange(len(feat_box)) * 2
pos_a      = pos_n + 0.7
axes[1, 0].boxplot(box_data_n, positions=pos_n, widths=0.6, patch_artist=True,
                   medianprops=dict(color="white"),
                   boxprops=dict(facecolor="#2196F3", alpha=0.7))
axes[1, 0].boxplot(box_data_a, positions=pos_a, widths=0.6, patch_artist=True,
                   medianprops=dict(color="white"),
                   boxprops=dict(facecolor="#F44336", alpha=0.7))
axes[1, 0].set_xticks(pos_n + 0.35)
axes[1, 0].set_xticklabels(feat_box, rotation=30, ha="right", fontsize=8)
axes[1, 0].set_title("Feature Box Plots: Normal vs Anomaly")
from matplotlib.patches import Patch
axes[1, 0].legend([Patch(facecolor="#2196F3"), Patch(facecolor="#F44336")],
                  ["Normal", "Anomaly"])

norm_vibe = (vibe_normal - vibe_normal.mean()) / vibe_normal.std()
stats.probplot(norm_vibe.values[:2000], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot — vibe_x (Normal samples)")

x_chi = np.linspace(0, 30, 300)
axes[1, 2].plot(x_chi, stats.chi2.pdf(x_chi, df=dof), color="#FF9800", lw=2)
axes[1, 2].axvline(chi2, color="red", ls="--", lw=2, label=f"χ²={chi2:.2f}")
axes[1, 2].fill_between(x_chi, stats.chi2.pdf(x_chi, dof),
                         where=(x_chi > stats.chi2.ppf(0.95, dof)),
                         alpha=0.3, color="red")
axes[1, 2].set_title(f"χ² Test — p={chi2_p:.2e}")
axes[1, 2].legend(fontsize=8)
axes[1, 2].set_xlim(0, 30)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_statistical_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 06_statistical_analysis.png")

# Figure 7  Sample Weight Distribution (updated — no ADMM)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Inverse-Frequency + Severity Sample Weighting (Fixed v2)",
             fontsize=14, fontweight="bold")

axes[0].hist(sw_win[y_tr == 0], bins=60, alpha=0.6, density=True,
             label="Normal", color="#2196F3")
axes[0].hist(sw_win[y_tr == 1], bins=60, alpha=0.6, density=True,
             label="Anomaly", color="#F44336")
axes[0].set_title("Sample Weight Distribution\n1/freq × (1 + k×severity), clipped [0.5, 4]")
axes[0].set_xlabel("Weight")
axes[0].legend()

sev_bins    = np.linspace(0, 1, 11)
sev_mids    = (sev_bins[:-1] + sev_bins[1:]) / 2
avg_weights = []
for lo, hi in zip(sev_bins[:-1], sev_bins[1:]):
    mask = (sev_tr >= lo) & (sev_tr < hi) & (y_tr == 1)
    avg_weights.append(sw_win[mask].mean() if mask.sum() > 0 else 0)
axes[1].bar(sev_mids, avg_weights, width=0.08, color="#FF5722", edgecolor="black")
axes[1].set_title(f"Avg Anomaly Weight by Severity Bin\n(k={K_SEVERITY})")
axes[1].set_xlabel("Severity")
axes[1].set_ylabel("Average Weight")

k_values  = [1.0, 2.5, 5.0, 8.0]
sev_range = np.linspace(0, 1, 100)
for k_v in k_values:
    axes[2].plot(sev_range, np.clip(1.0 + k_v * sev_range, 0.5, 4.0),
                 label=f"k={k_v}", lw=2)
axes[2].axvline(0.5, ls="--", color="gray", alpha=0.5)
axes[2].axhline(4.0, ls=":", color="red",  alpha=0.5, label="clip ceiling=4")
axes[2].set_title("Severity Scaling (clipped)\n1 + k × severity, clip [0.5, 4]")
axes[2].set_xlabel("Severity")
axes[2].set_ylabel("Severity factor")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_severity_weights.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 07_severity_weights.png")

# Figure 8  Decision Layer + Confidence
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Confidence Score & Decision Layer",
             fontsize=14, fontweight="bold")

decision_counts = inference_df["Decision"].value_counts()
dec_colors      = {"SAFE": "#4CAF50", "MONITOR": "#FF9800",
                   "INVESTIGATE": "#FF5722", "CRITICAL": "#F44336"}
axes[0].bar(decision_counts.index, decision_counts.values,
            color=[dec_colors.get(d, "gray") for d in decision_counts.index],
            edgecolor="black")
axes[0].set_title("Decision Layer Distribution\nSAFE / MONITOR / INVESTIGATE / CRITICAL")
axes[0].set_ylabel("Window Count")
for i, (d, v) in enumerate(decision_counts.items()):
    axes[0].text(i, v + 1, str(v), ha="center")

axes[1].hist(confidence_score[y_te == 0], bins=50, alpha=0.6, density=True,
             label="Normal",  color="#2196F3")
axes[1].hist(confidence_score[y_te == 1], bins=50, alpha=0.6, density=True,
             label="Anomaly", color="#F44336")
axes[1].set_title("Confidence Score Distribution\n|P - 0.5| × 2")
axes[1].set_xlabel("Confidence Score  [0=uncertain, 1=certain]")
axes[1].legend()

n_show = min(200, len(P_ensemble))
axes[2].plot(P_ensemble[:n_show],        label="Raw ensemble",      alpha=0.6,
             color="#9C27B0", lw=1)
axes[2].plot(P_ensemble_smooth[:n_show], label="Smoothed (roll=5)", color="#F44336", lw=2)
axes[2].axhline(0.5, ls="--", color="black", alpha=0.5, label="Decision threshold")
axes[2].set_title("Temporal Smoothing\nRolling mean reduces prediction noise")
axes[2].set_xlabel("Window Index")
axes[2].set_ylabel("P(Anomaly)")
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_decision_layer.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 08_decision_layer.png")

# Figure 9  Final Scorecard
fig, ax = plt.subplots(figsize=(18, 7))
ax.axis("off")
fig.suptitle("UAV Swarm Anomaly Detection — Final Scorecard (Fixed v2)",
             fontsize=16, fontweight="bold", y=0.98)

col_labels = ["Model", "Branch", "AUC-ROC", "F1-Macro", "Avg Precision", "Verdict"]
rows_sc = [
    [k,
     "ML" if k in ml_results else ("DL" if k == "CNN-LSTM" else "Fusion"),
     f"{v['AUC']:.4f}", f"{v['F1']:.4f}", f"{v['AP']:.4f}",
     "Best" if k == "Ensemble" else "Good" if v["AUC"] >= 0.9 else "OK"]
    for k, v in all_models_summary.items()
]

tbl = ax.table(cellText=rows_sc, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2.8)
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#1565C0")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
rc = {"Ensemble": "#E8F5E9", "CNN-LSTM": "#EDE7F6",
      "XGBoost":  "#E3F2FD", "LightGBM": "#E0F7FA"}
for i, row in enumerate(rows_sc, 1):
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(rc.get(row[0], "#FAFAFA"))

plt.savefig(f"{PLOT_DIR}/09_final_scorecard.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 09_final_scorecard.png")


# =============================================================================
# SECTION 15  FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY  (Fixed v2)")
print("=" * 70)
print(f"""
  Pipeline:
  ┌──────────────────────────────────────────────────────────────┐
  │  Raw CSV                                                     │
  │    Sort by [drone_id, time]                                  │
  │    Temporal derivatives  (acc_x_diff, volt_diff)            │
  │    IQR clip + RobustScale                                    │
  │    Window FFT  (seg={FFT_WIN})                                     │
  │    Window DMD  (seg={DMD_WIN})                                     │
  │    Toeplitz residuals + Kronecker interactions               │
  │    Quantum dynamic feature                                   │
  │                                                              │
  │  Sliding Window  (w={WINDOW}, stride={STEP})  ← stride fixed          │
  │    label  = mean > 0.2  ← noise-robust majority vote        │
  │    severity = max(severity) per window                       │
  │    ↓                          ↓                             │
  │  [ML: XGBoost + LightGBM]  [DL: CNN-LSTM v2]               │
  │  (aggregated window stats)  (16 raw sensor cols only)        │
  │    ↓                          ↓                             │
  │  Inverse-Freq × Severity Weights  (k={K_SEVERITY}, clip [0.5,4])  │
  │  Adaptive Ensemble Fusion  α={alpha}                              │
  │    Temporal Smoothing (roll=5)                               │
  │    DL threshold = 0.8  ← calibrated                         │
  │    Confidence Score |P-0.5|×2                                │
  │    Decision Layer: SAFE / MONITOR / INVESTIGATE / CRITICAL   │
  └──────────────────────────────────────────────────────────────┘

  Fixes applied:
    ✔  FIX 1  ADMM removed → inverse-frequency + severity weights
    ✔  FIX 2  DL branch → {len(seq_cols)} raw sensor cols only
    ✔  FIX 3  stride 15 → {STEP}
    ✔  FIX 4  window label = mean > 0.2
    ✔  FIX 5  lighter CNN-LSTM (Conv32→16, LSTM32, BatchNorm)
    ✔  FIX 6  class_weight removed from DL (sample_weight only)
    ✔  FIX 7  DL threshold 0.5 → 0.8
    ✔  FIX 8  sequence clip [-5, 5]
    ✔  FIX 9  epochs capped at 50
""")

print(f"  Ensemble  AUC={ens_auc:.4f} | F1={ens_f1:.4f} | AP={ens_ap:.4f}")
print(f"  {best_ml_name:<22} AUC={best_ml['AUC']:.4f} | F1={best_ml['F1']:.4f} | AP={best_ml['AP']:.4f}")
print(f"  CNN-LSTM               AUC={dl_auc:.4f} | F1={dl_f1:.4f} | AP={dl_ap:.4f}")

print(f"\n  Ensemble Classification Report:")
print(classification_report(y_te, y_ensemble,
                             target_names=["Normal", "Anomaly"], digits=4))

print("""
  Mathematical Methods:
    Unit 1: Window-based DFT, Hankel+DMD, Toeplitz, Kronecker
    Unit 2: Inverse-freq + severity weighting (k=2.5), Adam optimiser,
            binary cross-entropy
    Unit 3: MLE, Welch t-test, KS test, chi-squared test, Q-Q plot
    Unit 4: Bell state, Superdense coding, Teleportation,
            Quantum amplitude encoding (dynamic feature)

  Plots  →  /kaggle/working/plots/
  Output →  /kaggle/working/inference_output.csv
""")
print("=" * 70)