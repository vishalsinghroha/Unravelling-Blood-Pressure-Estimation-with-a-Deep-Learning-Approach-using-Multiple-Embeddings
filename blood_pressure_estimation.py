#!/usr/bin/env python3
"""
Blood Pressure Estimation with Deep Learning — Public Release Version
(ECG+PPG => PAT => Multi-embedding: Euclidean/Manhattan; AttentiveConvRegNet)

Repository-ready script:
- Loads PAT sequences and BP labels
- Builds Euclidean/Manhattan similarity matrices
- Trains AttentiveConvRegNet (CNN+CBAM) and baselines
- Evaluates with bootstrap CIs, Bland–Altman, error histograms, failure cases
- Optional “transformed features” pipeline from a pre-trained model

DATASET (PTT-PPG):
  PhysioNet: https://www.physionet.org/content/pulse-transit-time-ppg/1.0.0/

"""

import os
import sys
import time
import builtins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, ttest_rel, wilcoxon, shapiro, t

# ---------- Matplotlib defaults ----------
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.dpi'] = 1200

# ---------- Paths (repo-friendly) ----------
DATA_DIR = os.environ.get("PTT_PPG_DATA_DIR", "./data")
SAVE_DIR = os.environ.get("PTT_PPG_RESULTS_DIR", "./results")
os.makedirs(SAVE_DIR, exist_ok=True)

PAT_PATH = os.path.join(DATA_DIR, "pat.txt")          # one row per sample; ';' separated
BP_PATH  = os.path.join(DATA_DIR, "bp_data.csv")      # columns: sys_bp, dia_bp

# =========================================================
# --------------------- DATA LOADING ----------------------
# =========================================================

def ensure_data_files():
    ok = True
    if not os.path.isfile(PAT_PATH):
        ok = False
        print(f"[ERROR] Missing PAT file: {PAT_PATH}")
    if not os.path.isfile(BP_PATH):
        ok = False
        print(f"[ERROR] Missing BP CSV: {BP_PATH}")
    if not ok:
        print("\nDownload/prepare the PTT-PPG dataset and export PAT/BP to:")
        print(f"  PAT text (one row per sample, ';' separated) -> {PAT_PATH}")
        print(f"  BP CSV  (columns: sys_bp, dia_bp)           -> {BP_PATH}")
        print("Dataset link: https://www.physionet.org/content/pulse-transit-time-ppg/1.0.0/")
        sys.exit(1)

def load_pat_data_from_file(path=PAT_PATH):
    """Load PAT sequences from a ';'-separated text file, one sample per row."""
    with open(path, 'r') as f:
        lines = f.readlines()
    pat_list = []
    for i in range(len(lines)):
        temp_arr = (lines[i].strip()).split(';')
        pat_list.append(temp_arr)

    # simple stats
    lens = list(map(len, pat_list))
    pat_min = min(lens); pat_max = max(lens)
    pat_avg = sum(lens) / len(lens)
    pct_ge_30 = 100.0 * sum(l >= 30 for l in lens) / len(lens)
    print(f"Min PAT: {pat_min}, Max PAT: {pat_max}, Avg PAT: {pat_avg:.2f}, PAT>=30: {pct_ge_30:.1f}%")
    return pat_list

def normalize_pat_sequences(pat_list, fixed_len=30):
    """Trim/pad each PAT sequence to fixed_len (float32)."""
    final_pat = []
    for seq in pat_list:
        out = []
        for j in range(fixed_len):
            if j < len(seq):
                out.append(float(seq[j]))
            else:
                out.append(0.0)
        final_pat.append(out)
    return np.array(final_pat, dtype=np.float32)

def load_bp_data_csv(path=BP_PATH):
    df = pd.read_csv(path)
    if not all(col in df.columns for col in ("sys_bp", "dia_bp")):
        raise ValueError(f"BP CSV must contain 'sys_bp' and 'dia_bp' columns: {path}")
    return df

# =========================================================
# ---------------- SIMILARITY MATRICES --------------------
# =========================================================
def euclidean_distance_matrix(array):
    n, m = array.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(array[i] - array[j])
    return distance_matrix

def manhattan_distance_matrix(matrix):
    n, d = matrix.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.sum(np.abs(matrix[i] - matrix[j]))
    return distance_matrix

def clean_bp_and_matrices(sbp_arr, dbp_arr, final_pat, euclidean_matrix, manhattan_matrix,
                          sbp_range=(80, 150), dbp_range=(55, 120)):
    """
    Remove samples with implausible BP and drop them in PAT/EUC/MAN matrices.
    """
    sbp_arr = np.array(sbp_arr)
    dbp_arr = np.array(dbp_arr)
    final_pat = np.array(final_pat)
    euclidean_matrix = np.array(euclidean_matrix)
    manhattan_matrix = np.array(manhattan_matrix)

    invalid_indices = np.where(
        (sbp_arr < sbp_range[0]) | (sbp_arr > sbp_range[1]) |
        (dbp_arr < dbp_range[0]) | (dbp_arr > dbp_range[1])
    )[0]
    print(f"Invalid indices: {len(invalid_indices)}")

    if len(invalid_indices) == 0:
        return sbp_arr, dbp_arr, final_pat, euclidean_matrix, manhattan_matrix

    valid_indices = np.setdiff1d(np.arange(len(sbp_arr)), invalid_indices)
    return (sbp_arr[valid_indices],
            dbp_arr[valid_indices],
            final_pat[valid_indices],
            euclidean_matrix[np.ix_(valid_indices, valid_indices)],
            manhattan_matrix[np.ix_(valid_indices, valid_indices)])

# ==================================================================================
# ----------------------- Channel and Spatial Attention Blocks ---------------------
# ==================================================================================
class ChannelAttention(tf.keras.layers.Layer):
    """CBAM-Channel; stores last_attention for interpretability."""
    def __init__(self, filters=None, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = int(ratio)
        self.shared_layer_one = None
        self.shared_layer_two = None
        self._add = tf.keras.layers.Add()
        self._mul = tf.keras.layers.Multiply()
        self.last_attention = None  # (B,1,C)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        reduced = builtins.max(1, channels // self.ratio)
        self.shared_layer_one = tf.keras.layers.Dense(
            units=reduced, activation='relu',
            kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'
        )
        self.shared_layer_two = tf.keras.layers.Dense(
            units=channels, activation=None,
            kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
        max_pool = tf.reduce_max(inputs,  axis=1, keepdims=True)
        avg_out = self.shared_layer_two(self.shared_layer_one(avg_pool))
        max_out = self.shared_layer_two(self.shared_layer_one(max_pool))
        attention = self._add([avg_out, max_out])
        attention = tf.keras.activations.sigmoid(attention)
        attention = tf.cast(attention, inputs.dtype)
        self.last_attention = attention
        return self._mul([inputs, attention])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "ratio": self.ratio})
        return cfg

class SpatialAttention(tf.keras.layers.Layer):
    """CBAM-Spatial; stores last_attention for interpretability."""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.conv1d = None
        self._concat = tf.keras.layers.Concatenate(axis=-1)
        self._mul = tf.keras.layers.Multiply()
        self.last_attention = None  # (B,T,1)

    def build(self, input_shape):
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self.kernel_size, strides=1,
            padding='same', activation='sigmoid',
            kernel_initializer='he_normal', use_bias=False
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = self._concat([avg_pool, max_pool])
        attention = self.conv1d(x)
        attention = tf.cast(attention, inputs.dtype)
        self.last_attention = attention
        return self._mul([inputs, attention])

# =========================================================
# -------------------- MODEL DEFINITIONS ------------------
# =========================================================
def cnn_representation_model(X_train, y_train, model_name, X_val=None, y_val=None, save_dir=SAVE_DIR):
    save_path = os.path.join(save_dir, f"{model_name}_cnn_att_rep_model.h5")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        print(f"[INFO] Loading existing model: {save_path}")
        return tf.keras.models.load_model(
            save_path,
            custom_objects={"ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention},
            compile=False
        )

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                               activity_regularizer=regularizers.L1L2(l1=0.01, l2=0.01)),
        ChannelAttention(32, 8),
        SpatialAttention(7),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same',
                               activity_regularizer=regularizers.L1L2(l1=0.01, l2=0.01)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='linear'),
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=30, mode='min', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10,
                                                     min_delta=1e-3, mode='min', verbose=1)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mse'])
    model.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1,
              validation_data=(X_val, y_val), callbacks=[callback, reduce_lr])
    model.save(save_path)
    return model

def cnn_regression_model(X_train, y_train, model_name, X_val=None, y_val=None, save_dir=SAVE_DIR):
    save_path = os.path.join(save_dir, f"{model_name}_cnn_regression_model.h5")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        print(f"[INFO] Loading existing model: {save_path}")
        return tf.keras.models.load_model(save_path, compile=False)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                               activity_regularizer=regularizers.L2(0.01)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same',
                               activity_regularizer=regularizers.L2(0.01)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='linear'),
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=30, mode='min', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10,
                                                     min_delta=1e-3, mode='min', verbose=1)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mse'])
    model.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1,
              validation_data=(X_val, y_val), callbacks=[callback, reduce_lr])
    model.save(save_path)
    return model

def neural_regression_model(X_train, y_train, model_name, X_val=None, y_val=None, save_dir=SAVE_DIR):
    save_path = os.path.join(save_dir, f"{model_name}_nn_regression_model.h5")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(save_path):
        print(f"[INFO] Loading existing model: {save_path}")
        return tf.keras.models.load_model(save_path, compile=False)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear'),
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=30, mode='min', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10,
                                                     min_delta=1e-3, mode='min', verbose=1)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mse'])
    model.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1,
              validation_data=(X_val, y_val), callbacks=[callback, reduce_lr])
    model.save(save_path)
    return model

# =========================================================
# --------------- EVALUATION & PLOTTING -------------------
# =========================================================
def _full_box(ax, lw=1.6, tick_top_right=True, **_ignored):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(direction="in", length=4, width=lw*0.8,
                   top=tick_top_right, right=tick_top_right)

def _full_box_all(axs, lw=1.6, tick_top_right=True):
    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]
    for a in axs:
        _full_box(a, lw=lw, tick_top_right=tick_top_right)

def bootstrap_ci(y_true, y_pred, metric_func, n_iterations=1000, alpha=0.05):
    vals = []
    n = len(y_true); idx_all = np.arange(n)
    for _ in range(n_iterations):
        idx = np.random.choice(idx_all, n, replace=True)
        vals.append(metric_func(y_true[idx], y_pred[idx]))
    return np.percentile(vals, 100*(alpha/2)), np.percentile(vals, 100*(1-alpha/2))

def bootstrap_ci_simple(array, stat_func, n_iterations=1000, alpha=0.05):
    vals = []
    n = len(array); idx_all = np.arange(n)
    for _ in range(n_iterations):
        idx = np.random.choice(idx_all, n, replace=True)
        vals.append(stat_func(array[idx]))
    return np.percentile(vals, 100*(alpha/2)), np.percentile(vals, 100*(1-alpha/2))

def bootstrap_pearson_ci(y_true, y_pred, n_iterations=1000, alpha=0.05):
    r_vals = []
    n = len(y_true); idx_all = np.arange(n)
    for _ in range(n_iterations):
        idx = np.random.choice(idx_all, n, replace=True)
        r_vals.append(pearsonr(y_true[idx], y_pred[idx])[0])
    return np.percentile(r_vals, 100*(alpha/2)), np.percentile(r_vals, 100*(1-alpha/2))

def model_evaluation(y_true, y_pred, save_path=None):
    mse, rmse, mae, r2 = [], [], [], []
    print('*******************************************')
    for i, label in enumerate(['Systolic', 'Diastolic']):
        temp_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        temp_rmse = np.sqrt(temp_mse)
        temp_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        temp_r2 = r2_score(y_true[:, i], y_pred[:, i])
        temp_r, _ = pearsonr(y_true[:, i], y_pred[:, i])

        mse.append(temp_mse); rmse.append(temp_rmse)
        mae.append(temp_mae); r2.append(temp_r2)

        rmse_func = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        ci_rmse = bootstrap_ci(y_true[:, i], y_pred[:, i], rmse_func)
        ci_mae  = bootstrap_ci(y_true[:, i], y_pred[:, i], mean_absolute_error)
        ci_r2   = bootstrap_ci(y_true[:, i], y_pred[:, i], r2_score)
        ci_r    = bootstrap_pearson_ci(y_true[:, i], y_pred[:, i])

        diff = y_true[:, i] - y_pred[:, i]
        me = np.mean(diff)
        sd = np.std(diff, ddof=1)
        ci_me = bootstrap_ci_simple(diff, np.mean)
        ci_sd = bootstrap_ci_simple(diff, np.std)

        print(f'                    {label}')
        print(f'R-squared score: {temp_r2:.2f} (95% CI: {ci_r2[0]:.2f} - {ci_r2[1]:.2f})')
        print(f'Pearson Correlation: {temp_r:.2f} (95% CI: {ci_r[0]:.2f} - {ci_r[1]:.2f})')
        print(f'RMSE: {temp_rmse:.2f} (95% CI: {ci_rmse[0]:.2f} - {ci_rmse[1]:.2f})')
        print(f'MAE: {temp_mae:.2f} (95% CI: {ci_mae[0]:.2f} - {ci_mae[1]:.2f})')
        print(f'Mean Error: {me:.2f} (95% CI: {ci_me[0]:.2f} - {ci_me[1]:.2f})')
        print(f'Std Dev: {sd:.2f} (95% CI: {ci_sd[0]:.2f} - {ci_sd[1]:.2f})')
    print('*******************************************')

    # scatter with linear fits
    fig, axs = plt.subplots(1, 2, figsize=(6, 3.5))
    axs[0].scatter(y_true[:, 0], y_pred[:, 0], c='blue')
    coeff_sbp = np.polyfit(y_true[:, 0], y_pred[:, 0], 1)
    axs[0].plot(y_true[:, 0], np.polyval(coeff_sbp, y_true[:, 0]), color='green', linewidth=2)
    axs[0].set_xlabel('Actual SBP', fontweight='bold'); axs[0].set_ylabel('Predicted SBP', fontweight='bold')
    axs[1].scatter(y_true[:, 1], y_pred[:, 1], c='red')
    coeff_dbp = np.polyfit(y_true[:, 1], y_pred[:, 1], 1)
    axs[1].plot(y_true[:, 1], np.polyval(coeff_dbp, y_true[:, 1]), color='green', linewidth=2)
    axs[1].set_xlabel('Actual DBP', fontweight='bold'); axs[1].set_ylabel('Predicted DBP', fontweight='bold')
    _full_box_all([axs[0], axs[1]], lw=1.6, tick_top_right=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print("Linear fit saved at:", save_path)
    plt.close(fig)

def bland_altman_plot(y_true, y_pred, save_path=None):
    actual_sbp, actual_dbp = y_true[:, 0], y_true[:, 1]
    predicted_sbp, predicted_dbp = y_pred[:, 0], y_pred[:, 1]
    mean_sbp = 0.5 * (np.array(actual_sbp) + np.array(predicted_sbp))
    mean_dbp = 0.5 * (np.array(actual_dbp) + np.array(predicted_dbp))
    diff_sbp = np.array(actual_sbp) - np.array(predicted_sbp)
    diff_dbp = np.array(actual_dbp) - np.array(predicted_dbp)

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    axs[0].scatter(mean_sbp, diff_sbp, color='blue', alpha=0.7)
    axs[0].axhline(np.mean(diff_sbp), color='green', linestyle='--')
    axs[0].axhline(1.96*np.std(diff_sbp, ddof=1), color='grey', linestyle='--')
    axs[0].axhline(-1.96*np.std(diff_sbp, ddof=1), color='grey', linestyle='--')
    axs[0].set_xlabel('Mean SBP'); axs[0].set_ylabel('Difference (Actual - Pred)')
    axs[0].set_title('B-A Plot for SBP', fontweight='bold')

    axs[1].scatter(mean_dbp, diff_dbp, color='red', alpha=0.7)
    axs[1].axhline(np.mean(diff_dbp), color='green', linestyle='--')
    axs[1].axhline(1.96*np.std(diff_dbp, ddof=1), color='grey', linestyle='--')
    axs[1].axhline(-1.96*np.std(diff_dbp, ddof=1), color='grey', linestyle='--')
    axs[1].set_xlabel('Mean DBP'); axs[1].set_ylabel('Difference (Actual - Pred)')
    axs[1].set_title('B-A Plot for DBP', fontweight='bold')

    _full_box_all([axs[0], axs[1]], lw=1.6, tick_top_right=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print("Bland–Altman saved at:", save_path)
    plt.close(fig)

def cumulative_error_percentage(y_true, y_pred, thresholds=(5.0, 10.0, 15.0)):
    actual_sbp, actual_dbp = y_true[:, 0], y_true[:, 1]
    predicted_sbp, predicted_dbp = y_pred[:, 0], y_pred[:, 1]
    total = len(actual_sbp)
    for thr in thresholds:
        sbp_count = np.sum(np.abs(actual_sbp - predicted_sbp) <= thr)
        dbp_count = np.sum(np.abs(actual_dbp - predicted_dbp) <= thr)
        print(f"Cumulative Error (SBP) <= {thr} mmHg: {100*sbp_count/total:.2f}%")
        print(f"Cumulative Error (DBP) <= {thr} mmHg: {100*dbp_count/total:.2f}%")

def plot_error_distribution(y_true, y_pred, save_path_prefix, fixed_xlim=(-25, 25), normalize=True):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    err_sbp = y_pred[:, 0] - y_true[:, 0]
    err_dbp = y_pred[:, 1] - y_true[:, 1]
    all_err = np.concatenate([err_sbp, err_dbp])
    N = len(err_sbp)
    iqr = np.subtract(*np.percentile(all_err, [75, 25]))
    bw = 2 * iqr / (N ** (1/3)) if iqr > 0 else np.std(all_err) * 3.49 / (N ** (1/3) + 1e-8)
    bw = float(np.clip(bw, 0.5, 2.5))
    bins = np.arange(fixed_xlim[0], fixed_xlim[1] + bw, bw)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), dpi=1200)
    axs[0].hist(err_sbp, bins=bins, color='blue', alpha=0.85, edgecolor='black', density=normalize)
    axs[0].set_xlabel('Error (mmHg)'); axs[0].set_ylabel('Density' if normalize else 'Frequency')
    axs[0].set_xlim(fixed_xlim)
    axs[1].hist(err_dbp, bins=bins, color='red', alpha=0.85, edgecolor='black', density=normalize)
    axs[1].set_xlabel('Error (mmHg)'); axs[1].set_ylabel('Density' if normalize else 'Frequency')
    axs[1].set_xlim(fixed_xlim)
    for ax in axs:
        ax.spines['bottom'].set_linewidth(1.4); ax.spines['left'].set_linewidth(1.4)
        ax.tick_params(width=1.2, labelsize=11)
    plt.tight_layout()
    out = f"{save_path_prefix}_error_distribution_norm.png"
    plt.savefig(out, dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", out)

# =========================================================
# ---- PAT & FEATURE-MAP VISUALIZATIONS ----
# =========================================================
def _as_image(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2: return arr
    if arr.ndim == 1: return arr[np.newaxis, :]
    if arr.ndim == 3 and arr.shape[-1] == 1: return arr[..., 0]
    raise TypeError(f"Unsupported feature shape {arr.shape}")

def plot_failure_cases_featuremaps(feature_split, y_true, y_pred,
                                   split_name, model_name, save_dir, k=3):
    os.makedirs(save_dir, exist_ok=True)
    feature_split = np.asarray(feature_split)
    # SBP
    err_sbp = y_true[:, 0] - y_pred[:, 0]
    idx_sbp = np.argsort(-np.abs(err_sbp))[:k]
    fig, axes = plt.subplots(1, len(idx_sbp), figsize=(4.2*len(idx_sbp), 4), dpi=600, constrained_layout=True)
    if len(idx_sbp) == 1: axes = [axes]
    for ax, idx in zip(axes, idx_sbp):
        img = _as_image(feature_split[idx])
        im = ax.imshow(img, cmap="viridis", aspect="auto", interpolation="nearest")
        title = (f"SBP  true={y_true[idx,0]:.1f}, pred={y_pred[idx,0]:.1f}\n|err|={abs(err_sbp[idx]):.1f} mmHg")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("Time/Feature idx"); ax.set_ylabel("Row" if img.shape[0] > 1 else "")
        _full_box(ax, lw=1.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=8)
    out_path = os.path.join(save_dir, f"{model_name}_{split_name}_feature_failure_SBP_top{k}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)

    # DBP
    err_dbp = y_true[:, 1] - y_pred[:, 1]
    idx_dbp = np.argsort(-np.abs(err_dbp))[:k]
    fig, axes = plt.subplots(1, len(idx_dbp), figsize=(4.2*len(idx_dbp), 4), dpi=600, constrained_layout=True)
    if len(idx_dbp) == 1: axes = [axes]
    for ax, idx in zip(axes, idx_dbp):
        img = _as_image(feature_split[idx])
        im = ax.imshow(img, cmap="magma", aspect="auto", interpolation="nearest")
        title = (f"DBP  true={y_true[idx,1]:.1f}, pred={y_pred[idx,1]:.1f}\n|err|={abs(err_dbp[idx]):.1f} mmHg")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("Time/Feature idx"); ax.set_ylabel("Row" if img.shape[0] > 1 else "")
        _full_box(ax, lw=1.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=8)
    out_path = os.path.join(save_dir, f"{model_name}_{split_name}_feature_failure_DBP_top{k}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)

def top_failures_tables(y_true, y_pred, k=10, save_prefix=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_pred - y_true)

    def _table(col, name):
        idx = np.argsort(abs_err[:, col])
        best_idx  = idx[:k]; worst_idx = idx[::-1][:k]
        best_df = pd.DataFrame({"idx": best_idx,
                                f"{name}_true": y_true[best_idx, col],
                                f"{name}_pred": y_pred[best_idx, col],
                                f"{name}_absErr": abs_err[best_idx, col]})
        worst_df = pd.DataFrame({"idx": worst_idx,
                                 f"{name}_true": y_true[worst_idx, col],
                                 f"{name}_pred": y_pred[worst_idx, col],
                                 f"{name}_absErr": abs_err[worst_idx, col]})
        return best_df, worst_df

    sbp_best, sbp_worst = _table(0, "SBP")
    dbp_best, dbp_worst = _table(1, "DBP")

    if save_prefix:
        sbp_best.to_csv (f"{save_prefix}_sbp_best_top{k}.csv",  index=False)
        sbp_worst.to_csv(f"{save_prefix}_sbp_worst_top{k}.csv", index=False)
        dbp_best.to_csv (f"{save_prefix}_dbp_best_top{k}.csv",  index=False)
        dbp_worst.to_csv(f"{save_prefix}_dbp_worst_top{k}.csv", index=False)

    print("\n=== SBP worst ===");  print(sbp_worst.to_string(index=False))
    print("\n=== SBP best  ===");  print(sbp_best.to_string(index=False))
    print("\n=== DBP worst ===");  print(dbp_worst.to_string(index=False))
    print("\n=== DBP best  ===");  print(dbp_best.to_string(index=False))

# =========================================================
# --------- INTERPRETABILITY: ATTENTION & SALIENCY --------
# =========================================================
def _ensure(p): os.makedirs(p, exist_ok=True)
def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def _norm_rows(M):
    M = np.asarray(M, dtype=np.float32); M = np.maximum(M, 0)
    denom = M.sum(axis=1, keepdims=True) + 1e-8
    return M / denom

def _get_attentions_all(model, X, batch=64):
    sa_list, ca_list = [], []
    for i in range(0, len(X), batch):
        xb = tf.convert_to_tensor(X[i:i+batch], dtype=tf.float32)
        _ = model(xb, training=False)
        ca_layer = None; sa_layer = None
        for lyr in model.layers:
            if isinstance(lyr, ChannelAttention): ca_layer = lyr
            if isinstance(lyr, SpatialAttention): sa_layer = lyr
        if (ca_layer is None) or (sa_layer is None):
            raise RuntimeError("ChannelAttention/SpatialAttention layer not found.")
        ca = ca_layer.last_attention.numpy()[:, 0, :]
        sa = sa_layer.last_attention.numpy()[:, :, 0]
        ca_list.append(ca); sa_list.append(sa)
    return np.concatenate(sa_list, axis=0), np.concatenate(ca_list, axis=0)

@tf.function
def _grads_batch(model, xb, target_index):
    with tf.GradientTape() as tape:
        tape.watch(xb)
        y = model(xb, training=False)
        loss = tf.reduce_sum(y[:, target_index])
    return tape.gradient(loss, xb)

def _saliency_all(model, X, target_index, batch=64):
    N, T, _ = X.shape
    out = np.zeros((N, T), np.float32)
    for i in range(0, N, batch):
        xb = tf.convert_to_tensor(X[i:i+batch], dtype=tf.float32)
        g = tf.abs(_grads_batch(model, xb, target_index))[:, :, 0].numpy()
        g /= (np.max(g, axis=1, keepdims=True) + 1e-8)
        out[i:i+batch] = g
    return out

def _bar_meanstd(values, title, xlabel, ylabel, out_path):
    mean = values.mean(axis=0); std = values.std(axis=0, ddof=1)
    idx = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.bar(idx, mean, yerr=std, alpha=0.9, capsize=2, linewidth=0)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    _save(fig, out_path)

def _heatmap(matrix, title, xlabel, ylabel, out_path, cmap="magma"):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    _save(fig, out_path)

def export_avg_interpretability(model, X_split, split_name, model_name, save_dir):
    out = os.path.join(save_dir, f"{model_name}_interp_{split_name}")
    _ensure(out)
    sa_all, ca_all = _get_attentions_all(model, X_split, batch=64)
    np.save(os.path.join(out, f"{model_name}_{split_name}_attn_sa.npy"), sa_all)
    np.save(os.path.join(out, f"{model_name}_{split_name}_attn_ca.npy"), ca_all)

    _bar_meanstd(
        ca_all, title=f"{model_name} | {split_name} Channel Attention (mean±SD)",
        xlabel="Channel", ylabel="Weight",
        out_path=os.path.join(out, f"{model_name}_{split_name}_attn_ca_meanstd_bar.png")
    )

    ca_norm = _norm_rows(ca_all)
    order = np.argsort(np.argmax(ca_norm, axis=1))
    _heatmap(
        ca_norm[order],
        title=f"{model_name} | {split_name} Channel Attention — Samples × Channel",
        xlabel="Channel", ylabel="Samples (sorted by peak)",
        out_path=os.path.join(out, f"{model_name}_{split_name}_attn_ca_heatmap_samplesxchannel.png"),
        cmap="magma"
    )

    eps = 1e-8
    sa_mm = (sa_all - sa_all.min(axis=1, keepdims=True)) / (sa_all.ptp(axis=1, keepdims=True) + eps)
    ca_mm = (ca_all - ca_all.min(axis=1, keepdims=True)) / (ca_all.ptp(axis=1, keepdims=True) + eps)
    N, T = sa_mm.shape; C = ca_mm.shape[1]
    joint_per_channel = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        A_i = np.outer(sa_mm[i], ca_mm[i])
        joint_per_channel[i] = A_i.mean(axis=0)

    _bar_meanstd(
        joint_per_channel,
        title=f"{model_name} | {split_name} Channel×Spatial (mean±SD)",
        xlabel="Channel", ylabel="Weight",
        out_path=os.path.join(out, f"{model_name}_{split_name}_attn_joint_channel_meanstd_bar.png")
    )

    joint_norm = _norm_rows(joint_per_channel)
    order_j = np.argsort(np.argmax(joint_norm, axis=1))
    _heatmap(
        joint_norm[order_j],
        title=f"{model_name} | {split_name} Channel×Spatial — Samples × Channel",
        xlabel="Channel", ylabel="Samples (sorted by peak)",
        out_path=os.path.join(out, f"{model_name}_{split_name}_attn_joint_channel_heatmap_samplesxchannel.png"),
        cmap="inferno"
    )

    saN = _norm_rows(sa_all); caN = _norm_rows(ca_all)
    A = np.zeros((T, C), dtype=np.float64)
    for i in range(N):
        A += np.outer(saN[i], caN[i])
    A /= float(N)
    _heatmap(A,   f"{model_name} | {split_name} Joint (Time × Channel)",
             "Channel", "Time", os.path.join(out, "joint_time_channel_mean.png"), cmap="plasma")
    _heatmap(A.T, f"{model_name} | {split_name} Joint (Channel × Time)",
             "Time", "Channel", os.path.join(out, "joint_channel_time_mean.png"), cmap="plasma")

    sal0 = _saliency_all(model, X_split, target_index=0, batch=64)
    sal1 = _saliency_all(model, X_split, target_index=1, batch=64)
    np.save(os.path.join(out, f"{model_name}_{split_name}_sal_t0.npy"), sal0)
    np.save(os.path.join(out, f"{model_name}_{split_name}_sal_t1.npy"), sal1)

    # mean±SD curves
    def _curve_meanstd(values, title, xlabel, ylabel, out_path):
        mean = values.mean(axis=0); std = values.std(axis=0, ddof=1)
        x = np.arange(values.shape[1])
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(x, mean, linewidth=1.8)
        ax.fill_between(x, mean-std, mean+std, alpha=0.30)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        _save(fig, out_path)

    _curve_meanstd(sal0, f"{model_name} | {split_name} Saliency (SBP) mean±SD",
                   "Time", "Normalized |grad|",
                   os.path.join(out, f"{model_name}_{split_name}_saliency_sbp_meanstd.png"))
    _curve_meanstd(sal1, f"{model_name} | {split_name} Saliency (DBP) mean±SD",
                   "Time", "Normalized |grad|",
                   os.path.join(out, f"{model_name}_{split_name}_saliency_dbp_meanstd.png"))
    print(f"[OK] Saved interpretability into: {out}")

# =========================================================
# -------------- TRAIN / EVAL MAIN PIPELINE ----------------
# =========================================================
def run_training(feature_arr, target_arr, model_name, save_dir=SAVE_DIR, pat_arr=None):
    feature_arr = np.asarray(feature_arr, dtype=np.float32)
    if feature_arr.ndim == 2:
        feature_arr = feature_arr.reshape(feature_arr.shape[0], feature_arr.shape[1], 1)
    target_arr = np.asarray(target_arr, dtype=np.float32)

    X_tr, X_te, y_tr, y_te, pat_tr, pat_te = train_test_split(
        feature_arr, target_arr, pat_arr, test_size=0.2, shuffle=True, random_state=40)
    X_tr, X_val, y_tr, y_val, pat_tr, pat_val = train_test_split(
        X_tr, y_tr, pat_tr, test_size=0.1, shuffle=True, random_state=40)

    print('Train:', X_tr.shape, ' Val:', X_val.shape, ' Test:', X_te.shape)

    model = cnn_representation_model(X_tr, y_tr, model_name, X_val=X_val, y_val=y_val, save_dir=save_dir)

    print("\n--- Measuring Inference Time (TEST) ---")
    t0 = time.perf_counter()
    y_pred_te = model.predict(X_te, batch_size=16, verbose=0)
    t1 = time.perf_counter()
    total_time = t1 - t0
    avg_ms = (total_time / len(X_te)) * 1000.0
    print(f"Total: {total_time:.3f}s | Avg per sample: {avg_ms:.3f} ms")

    model_evaluation(y_te, y_pred_te, os.path.join(save_dir, f"{model_name}_linear_fit.png"))
    cumulative_error_percentage(y_te, y_pred_te)
    bland_altman_plot(y_te, y_pred_te, os.path.join(save_dir, f"{model_name}_bland_altman.png"))
    plot_error_distribution(y_te, y_pred_te, os.path.join(save_dir, model_name))

    y_pred_val = model.predict(X_val, batch_size=16, verbose=0)

    # failure-case visualizations on the *features the model consumed*
    plot_failure_cases_featuremaps(X_te.squeeze(-1), y_te, y_pred_te, "test", model_name, save_dir, k=3)
    plot_failure_cases_featuremaps(X_val.squeeze(-1), y_val, y_pred_val, "val", model_name, save_dir, k=3)

    np.savez(os.path.join(save_dir, f"{model_name}_test_results.npz"),
             y_test=y_te, y_pred=y_pred_te, total_time=total_time, avg_time_per_sample=avg_ms)
    np.savez(os.path.join(save_dir, f"{model_name}_val_results.npz"),
             y_val=y_val, y_pred=y_pred_val)
    _ = top_failures_tables(y_te, y_pred_te, k=10, save_prefix=os.path.join(save_dir, f"{model_name}_test"))

    export_avg_interpretability(model, X_val, "val", model_name, save_dir)
    export_avg_interpretability(model, X_te,  "test", model_name, save_dir)

    return model, (X_val, y_val, y_pred_val), (X_te, y_te, y_pred_te)

# =========================================================
# -------------------------- MAIN -------------------------
# =========================================================
def main():
    print("=== Blood Pressure Estimation — Public Release ===")
    print("TensorFlow:", tf.__version__)

    ensure_data_files()

    # 1) Load and preprocess data
    pat_list = load_pat_data_from_file(PAT_PATH)
    final_pat = normalize_pat_sequences(pat_list, fixed_len=30)

    target_df = load_bp_data_csv(BP_PATH)
    sbp_arr = np.array(target_df['sys_bp'].values, dtype=np.float32)
    dbp_arr = np.array(target_df['dia_bp'].values, dtype=np.float32)

    scaler = StandardScaler()
    final_pat_std = scaler.fit_transform(final_pat).astype('float32')

    # 2) Embeddings
    euc = euclidean_distance_matrix(final_pat_std)
    man = manhattan_distance_matrix(final_pat_std)

    # 3) Clean extremes (OPTIONAL: adjust ranges)
    sbp_clean, dbp_clean, pat_clean, euc_clean, man_clean = clean_bp_and_matrices(
        sbp_arr, dbp_arr, final_pat_std, euc, man,
        sbp_range=(80, 150), dbp_range=(55, 120)
    )
    target_arr = np.column_stack((sbp_clean, dbp_clean))

    print("Shapes — PAT:", pat_clean.shape, " Euc:", euc_clean.shape, " Man:", man_clean.shape,
          " Targets:", target_arr.shape)

    # 4) Run trainings you want (AttentiveConvRegNet on each representation)
    print("\n=== Training on Euclidean Matrix ===")
    run_training(euc_clean, target_arr, 'pttppg_euc', SAVE_DIR, pat_arr=pat_clean)

    print("\n=== Training on Manhattan Matrix ===")
    run_training(man_clean, target_arr, 'pttppg_man', SAVE_DIR, pat_arr=pat_clean)

    print("\n=== Training on Euclidean + Manhattan (SUM) ===")
    euc_man_sum = np.sum((euc_clean, man_clean), axis=0)
    run_training(euc_man_sum, target_arr, 'pttppg_euc_man_sum', SAVE_DIR, pat_arr=pat_clean)

    print("\n=== Training on Euclidean + Manhattan (CONCAT) ===")
    euc_man_con = np.concatenate((euc_clean, man_clean), axis=1)
    run_training(euc_man_con, target_arr, 'pttppg_euc_man_con', SAVE_DIR, pat_arr=pat_clean)

    # 5) Optional: transformed-features pipeline (set these to True to enable)
    DO_TRANSFORMED_EUC = False
    DO_TRANSFORMED_MAN = False
    if DO_TRANSFORMED_EUC:
        # load trained euc model and extract penultimate features, then train downstream
        model_path = os.path.join(SAVE_DIR, "pttppg_euc_cnn_att_rep_model.h5")
        if os.path.exists(model_path):
            loaded = tf.keras.models.load_model(
                model_path,
                custom_objects={"ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention},
                compile=False
            )
            X_euc = euc_clean.reshape(euc_clean.shape[0], euc_clean.shape[1], 1).astype('float32')
            _ = loaded.predict(X_euc[:1], verbose=0)  # warm-up
            try:
                penultimate = loaded.layers[-2].output
                inp = loaded.input
            except Exception:
                inp = tf.keras.Input(shape=X_euc.shape[1:])
                x = inp
                for lyr in loaded.layers[:-1]:
                    x = lyr(x)
                penultimate = x
            feat_model = keras.Model(inp, penultimate)
            euc_feats = feat_model.predict(X_euc, batch_size=16, verbose=1)
            run_training(euc_feats, target_arr, 'pttppg_transformed_euc', SAVE_DIR, pat_arr=pat_clean)
        else:
            print("[WARN] Skipping transformed EUC (model not found).")

    if DO_TRANSFORMED_MAN:
        model_path = os.path.join(SAVE_DIR, "pttppg_man_cnn_att_rep_model.h5")
        if os.path.exists(model_path):
            loaded = tf.keras.models.load_model(
                model_path,
                custom_objects={"ChannelAttention": ChannelAttention, "SpatialAttention": SpatialAttention},
                compile=False
            )
            X_man = man_clean.reshape(man_clean.shape[0], man_clean.shape[1], 1).astype('float32')
            _ = loaded.predict(X_man[:1], verbose=0)
            try:
                penultimate = loaded.layers[-2].output
                inp = loaded.input
            except Exception:
                inp = tf.keras.Input(shape=X_man.shape[1:])
                x = inp
                for lyr in loaded.layers[:-1]:
                    x = lyr(x)
                penultimate = x
            feat_model = keras.Model(inp, penultimate)
            man_feats = feat_model.predict(X_man, batch_size=16, verbose=1)
            run_training(man_feats, target_arr, 'pttppg_transformed_man', SAVE_DIR, pat_arr=pat_clean)
        else:
            print("[WARN] Skipping transformed MAN (model not found).")

    print("\n=== Training Complete ===")
    print("Results saved under:", SAVE_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
