import matplotlib
# Set backend to Agg to prevent "qt.qpa.plugin" errors on headless systems/WSL
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from matplotlib.gridspec import GridSpec
import os
# Ensure reproduce_model.py is in the same directory or python path
try:
    from reproduce_model import predict_proba_pos
except ImportError:
    # Fallback if import fails, purely for testing the plotting logic
    print("Warning: Could not import predict_proba_pos. Using dummy function.")
    def predict_proba_pos(model, X):
        return [0.5]

# ---------------------------------------------------------
# 1. INPUT DATA
# ---------------------------------------------------------
CURRENT_DIR = os.getcwd()

# Load Models
# NOTE: Ensure these paths are correct relative to where you run the script
try:
    reproduced_model = joblib.load('model/model.joblib')
    ref_model = joblib.load('inference_resources_v003/models/detect_cancer__002__24_06_25/model.joblib')
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please run this script from the root of the lionheart-pipeline-reproduce directory.")
    exit(1)

# --- CRITICAL FIX: Cast numpy arrays to scalar Python types (int/float) ---
# The error "setting an array element with a sequence" happens because 
# model.coef_ and model.intercept_ are arrays, not numbers.

# 1. Feature Counts (int)
our_n_features = int(np.sum(reproduced_model["logistic_regression"].coef_ != 0))
ref_n_features = int(np.sum(ref_model["model"].coef_ != 0))

# 2. Coefficients (float)
our_max_coef = float(np.max(reproduced_model['logistic_regression'].coef_))
ref_max_coef = float(np.max(ref_model['model'].coef_))

our_min_coef = float(np.min(reproduced_model['logistic_regression'].coef_))
ref_min_coef = float(np.min(ref_model['model'].coef_))

# 3. Intercept (float) - Using .item() to safely extract scalar from array
our_int = float(reproduced_model["logistic_regression"].intercept_.item())
ref_int = float(ref_model["model"].intercept_.item())

# 4. Probability (float)
try:
    X = np.load('model/feature_dataset.npy')[None,0,:]
    # predict_proba_pos might return an array, we need the scalar value
    prob_result = predict_proba_pos(reproduced_model, X)
    if isinstance(prob_result, np.ndarray):
        our_prob = float(prob_result.item()) if prob_result.size == 1 else float(prob_result[0])
    elif isinstance(prob_result, list):
        our_prob = float(prob_result[0])
    else:
        our_prob = float(prob_result)
except Exception as e:
    print(f"Warning: Could not calculate probability from dataset ({e}). Using placeholder.")
    our_prob = 0.0

metrics = {
    "Our Model": {
        "Active Features": our_n_features,
        "Max Coeff": our_max_coef,
        "Min Coeff": our_min_coef,
        "Intercept": our_int,
        "P(Cancer)": our_prob,
        "Regularization (C)": 0.04,
        "PCA Var": 0.987,
    },
    "Reference": {
        "Active Features": ref_n_features,
        "Max Coeff": ref_max_coef,
        "Min Coeff": ref_min_coef,
        "Intercept": ref_int,
        "P(Cancer)": 0.948, 
        "Regularization (C)": 0.20,
        "PCA Var": 0.988,
    }
}

# AUC Performance per dataset
auc_data = {
    "Dataset": [
        "Cristiano", "Jiang", "Mathios Val", "Budhraja", 
        "Mathios", "Endoscopy II", "GECOCA", "Nordentoft"
    ],
    "Sample Size": [474, 112, 431, 192, 270, 255, 168, 274],
    "Our AUC": [0.936, 0.882, 0.866, 0.823, 0.749, 0.701, 0.690, 0.637],
    "Ref AUC": [0.971, 0.902, 0.982, 0.983, 0.852, 0.766, 0.883, 0.862]
}
df_auc = pd.DataFrame(auc_data).sort_values("Our AUC", ascending=False)

# ---------------------------------------------------------
# 2. SETUP PLOTTING STYLE
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
# Fallback fonts in case Arial isn't installed on Linux
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
COLORS = {"Our Model": "#0077b6", "Reference": "#94a3b8"}

# ---------------------------------------------------------
# 3. GENERATE PLOTS
# ---------------------------------------------------------
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2])

# --- SUBPLOT 1: VITAL SIGNS ---
ax1 = fig.add_subplot(gs[0, 0])
labels = ["Active Features", "Intercept", "Regularization (C)", "PCA Variance"]
x_pos = np.arange(len(labels))
width = 0.35

# Use scalars for scaling
scale_factors = [1, 100, 100, 100] 

# Ensure vals are lists of simple floats/ints
vals_our = [float(metrics["Our Model"][l]) * s for l, s in zip(["Active Features", "Intercept", "Regularization (C)", "PCA Var"], scale_factors)]
vals_ref = [float(metrics["Reference"][l]) * s for l, s in zip(["Active Features", "Intercept", "Regularization (C)", "PCA Var"], scale_factors)]

rects1 = ax1.bar(x_pos - width/2, vals_our, width, label='Our Model', color=COLORS["Our Model"], alpha=0.9)
rects2 = ax1.bar(x_pos + width/2, vals_ref, width, label='Reference', color=COLORS["Reference"], alpha=0.9)

def autolabel(rects, real_values):
    for rect, val in zip(rects, real_values):
        height = rect.get_height()
        # Format label depending on value size
        label_text = f'{val:.3f}' if isinstance(val, float) and abs(val) < 10 else f'{val}'
        if isinstance(val, (int, np.integer)):
             label_text = f'{val}'
             
        ax1.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1, [metrics["Our Model"][l] for l in ["Active Features", "Intercept", "Regularization (C)", "PCA Var"]])
autolabel(rects2, [metrics["Reference"][l] for l in ["Active Features", "Intercept", "Regularization (C)", "PCA Var"]])

ax1.set_ylabel('Value (Scaled)')
ax1.set_title('Final Models Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# --- SUBPLOT 2: COEFFICIENT RANGE ---
ax2 = fig.add_subplot(gs[0, 1])
models_list = ['Our Model', 'Reference']
max_coeffs = [metrics[m]["Max Coeff"] for m in models_list]
min_coeffs = [metrics[m]["Min Coeff"] for m in models_list]

for i, _ in enumerate(models_list):
    ax2.plot([i, i], [min_coeffs[i], max_coeffs[i]], color='black', linewidth=1.5, zorder=1)
    ax2.scatter(i, max_coeffs[i], color='#10b981', s=150, label='Max Coeff' if i==0 else "", zorder=2, marker='^')
    ax2.scatter(i, min_coeffs[i], color='#ef4444', s=150, label='Min Coeff' if i==0 else "", zorder=2, marker='v')
    ax2.text(i + 0.1, max_coeffs[i], f"{max_coeffs[i]:.2f}", va='center', fontsize=11, fontweight='bold', color='#10b981')
    ax2.text(i + 0.1, min_coeffs[i], f"{min_coeffs[i]:.2f}", va='center', fontsize=11, fontweight='bold', color='#ef4444')

ax2.set_xlim(-0.5, 1.5)
# Dynamic ylim with some padding
all_coeffs = max_coeffs + min_coeffs
y_min, y_max = min(all_coeffs), max(all_coeffs)
ax2.set_ylim(y_min - 0.5, y_max + 0.5)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(models_list, fontsize=12)
ax2.set_title('Coefficient Range', fontsize=14, fontweight='bold', pad=15)
ax2.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_ylabel('Coefficient Magnitude')
ax2.legend(loc='upper left')

# --- SUBPLOT 3: PREDICTION CONFIDENCE ---
ax3 = fig.add_subplot(gs[0, 2])
probs = [metrics["Our Model"]["P(Cancer)"], metrics["Reference"]["P(Cancer)"]]
bars = ax3.bar(models_list, probs, color=[COLORS["Our Model"], COLORS["Reference"]], width=0.5)
ax3.set_ylim(0, 1.1)
ax3.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Decision Threshold')
ax3.set_title('Prediction on Test Sample', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylabel('P(Cancer)')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}',
             ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)

# --- SUBPLOT 4: GENERALIZATION HEATMAP ---
ax4 = fig.add_subplot(gs[1, :2])
y_pos = np.arange(len(df_auc))
bar_height = 0.35

rects_auc1 = ax4.barh(y_pos + bar_height/2, df_auc['Our AUC'], bar_height, label='Our Model', color=COLORS["Our Model"])
rects_auc2 = ax4.barh(y_pos - bar_height/2, df_auc['Ref AUC'], bar_height, label='Reference', color=COLORS["Reference"])

ax4.set_xlabel('ROC AUC')
ax4.set_title('Validation Performance (Cross Dataset AUC)', fontsize=14, fontweight='bold', pad=15)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(df_auc['Dataset'])
ax4.set_xlim(0.5, 1.0)
ax4.invert_yaxis()
ax4.legend()

# Add values to bars
for rect in rects_auc1:
    width_val = rect.get_width()
    ax4.text(width_val - 0.05, rect.get_y() + rect.get_height() / 2, f'{width_val:.3f}', ha='right', va='center', color='white', fontweight='bold', fontsize=9)
for rect in rects_auc2:
    width_val = rect.get_width()
    ax4.text(width_val - 0.05, rect.get_y() + rect.get_height() / 2, f'{width_val:.3f}', ha='right', va='center', color='black', fontweight='bold', fontsize=9)

# --- SUBPLOT 5: MODEL FINGERPRINT ---
ax5 = fig.add_subplot(gs[1, 2])

# 1. Load Reference Coefficients
active_coefs_ref = []
try:
    # Assuming pipeline structure
    if hasattr(ref_model, 'named_steps') and 'model' in ref_model.named_steps:
        coefs = ref_model.named_steps['model'].coef_.ravel()
    elif isinstance(ref_model, dict) and 'model' in ref_model:
         # Handle dict case if saved as dict
         coefs = ref_model['model'].coef_.ravel()
    else:
         # Fallback attempt to find coef_ in object
         coefs = ref_model.coef_.ravel() if hasattr(ref_model, 'coef_') else np.array([])
         
    active_coefs_ref = coefs[coefs != 0]
except Exception as e:
    print(f"Note: Simulating Ref coefficients ({e})")
    active_coefs_ref = np.concatenate([np.random.normal(0.5, 0.2, 40), np.random.normal(-0.5, 0.2, 38)])

# 2. Load Our Model Coefficients
active_coefs_our = []
try:
    # reproduce_model usually saves a dict with 'logistic_regression'
    if isinstance(reproduced_model, dict) and 'logistic_regression' in reproduced_model:
        coefs = reproduced_model['logistic_regression'].coef_.ravel()
    else:
        coefs = reproduced_model.coef_.ravel() if hasattr(reproduced_model, 'coef_') else np.array([])
        
    active_coefs_our = coefs[coefs != 0]
except Exception as e:
    print(f"Note: Simulating Our coefficients ({e})")
    active_coefs_our = np.concatenate([np.random.normal(0.4, 0.15, 22), np.random.normal(-0.4, 0.15, 21)])

sns.histplot(active_coefs_ref, bins=20, ax=ax5, color=COLORS["Reference"], kde=True, label='Reference', alpha=0.4, element="step")
sns.histplot(active_coefs_our, bins=20, ax=ax5, color=COLORS["Our Model"], kde=True, label='Our Model', alpha=0.6, element="step")

ax5.set_title('Model Fingerprint Comparison\n(Distribution of Non-Zero Weights)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Coefficient Value')
ax5.axvline(0, color='black', linewidth=1)
ax5.legend()

ax5.text(0.05, 0.9, f"Ref Active: {len(active_coefs_ref)}", transform=ax5.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
ax5.text(0.05, 0.82, f"Our Active: {len(active_coefs_our)}", transform=ax5.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

# --- FINALIZE ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig('lionheart_reproduction_analysis.png', dpi=300)
print("Plots generated and saved as 'lionheart_reproduction_analysis.png'")