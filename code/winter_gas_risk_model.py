"""
A Multi-Stage Probabilistic Framework for Gas-Fired Generator Performance
During Extreme Winter Weather — Reproducible Analysis Script

Authors: Sajjad Uddin Mahmud, Anamika Dubey
Washington State University

Usage
-----
1. Place hourly_dataset_NY.csv and event_dataset_NY.csv in the same
   folder as this script (or update DATA_DIR below).
2. Run:  python winter_gas_generator_risk.py
3. All outputs are saved to the folder specified by OUT_DIR.

Requirements
------------
    pip install numpy pandas matplotlib seaborn scipy scikit-learn
                jax numpyro arviz plotly

Dataset columns used
--------------------
hourly_dataset_NY.csv
    datetime      : hourly timestamp
    CEI           : Cold Exposure Index (standardised)
    D_norm        : normalised demand
    E_t           : event indicator (1 = winter contingency event, 0 = no event)

event_dataset_NY.csv
    event_index   : unique event identifier
    NAC_norm      : normalised net available capacity (0 = full outage)
    duration      : event duration (hh:mm)
    CEI           : Cold Exposure Index at event initiation (standardised)
    D_norm        : normalised demand at event initiation
"""

# ============================================================
# User Settings  ← edit these paths before running
# ============================================================

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"     # folder containing the two CSV files
OUT_DIR  = Path(__file__).parent.parent / "results"  # all figures and CSVs are saved here
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go

from scipy.special import expit

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

# ============================================================
# Global plot style
# ============================================================

FONT_SIZE = 16
plt.rcParams["font.family"]    = "Times New Roman"
plt.rcParams["font.size"]      = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"]= FONT_SIZE
plt.rcParams["ytick.labelsize"]= FONT_SIZE
plt.rcParams["legend.fontsize"]= FONT_SIZE

# ============================================================
# Load datasets
# ============================================================

print("Loading datasets ...")
hourly_df = pd.read_csv(DATA_DIR / "hourly_dataset_NY.csv", parse_dates=["datetime"])
event_df  = pd.read_csv(DATA_DIR / "event_dataset_NY.csv")

print(f"  Hourly dataset : {len(hourly_df):,} rows")
print(f"  Event dataset  : {len(event_df):,} rows")

# ============================================================
# ============================================================
#  STAGE 1 — Hourly Event Probability (Bayesian Logistic)
# ============================================================
# ============================================================

print("\n" + "="*60)
print("STAGE 1 — Hourly Event Probability")
print("="*60)

S1_DIR = OUT_DIR / "results_stage1"
S1_DIR.mkdir(parents=True, exist_ok=True)

# ── Features & target ──────────────────────────────────────
# CEI and D_norm are already standardised in the dataset

s1_data = hourly_df[["CEI", "D_norm", "E_t"]].dropna().copy()

X = s1_data[["D_norm", "CEI"]].values   # col 0 = D_norm, col 1 = CEI
y = s1_data["E_t"].values.astype(int)

print(f"  Stage 1 dataset : {len(X):,} hours  |  event rate = {y.mean()*100:.2f}%")

# ── Train / Test split ─────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_j = jnp.array(X_train)
y_train_j = jnp.array(y_train)

# ============================================================
# Bayesian Logistic Model
# ============================================================

def logistic_model(X, y=None):
    alpha  = numpyro.sample("alpha", dist.Normal(0, 5))
    beta   = numpyro.sample("beta",  dist.Normal(0, 5).expand([X.shape[1]]))
    logits = alpha + jnp.dot(X, beta)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

# ── Run MCMC ───────────────────────────────────────────────

print("\nRunning MCMC for Stage 1 ...")
kernel = NUTS(logistic_model)
mcmc   = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=2)
mcmc.run(random.PRNGKey(0), X_train_j, y_train_j)
samples = mcmc.get_samples()

# ============================================================
# Save Posterior Samples
# ============================================================

posterior_df = pd.DataFrame({
    "alpha"  : samples["alpha"],
    "beta_Dnorm": samples["beta"][:, 0],
    "beta_CEI"  : samples["beta"][:, 1],
})
posterior_df.to_csv(S1_DIR / "stage1_posterior_samples.csv", index=False)
print("  Posterior samples saved.")

# ============================================================
# Posterior Summary
# ============================================================

idata   = az.from_numpyro(mcmc)
summary = az.summary(idata)
summary.to_csv(S1_DIR / "stage1_posterior_summary.csv")
print("  Posterior summary saved.")

# ============================================================
# Posterior Density Plot
# ============================================================

params = [
    ("alpha",    r"$\beta_0$",                    "#9ecae1"),
    ("beta_Dnorm", r"$\beta_{D_{\mathrm{norm}}}$", "#a1d99b"),
    ("beta_CEI",   r"$\beta_{\mathrm{CEI}}$",      "#fcae91"),
]

fig, axes = plt.subplots(3, 1, figsize=(7, 12))

for i, (col, label, color) in enumerate(params):
    ax   = axes[i]
    vals = posterior_df[col]

    ax.hist(vals, bins=40, density=True, alpha=0.6,
            color=color, edgecolor="none")
    sns.kdeplot(vals, ax=ax, color="#696868", linewidth=1.2)

    mean_val = vals.mean()
    std_val  = vals.std()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)

    ax.text(0.97, 0.95,
            f"Mean = {mean_val:.2f}\nStd = {std_val:.2f}",
            transform=ax.transAxes,
            color="red", fontsize=FONT_SIZE,
            ha="right", va="top")

    ax.set_title(label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density" if i == 0 else "", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(S1_DIR / "stage1_posterior_density.png", dpi=300, bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_posterior_density.pdf", format="pdf", bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_posterior_density.eps", format="eps", bbox_inches="tight")
plt.close()
print("  Posterior density plot saved.")


# ============================================================
# Validation: ROC Curve — 5-fold Stratified Cross-Validation
# ============================================================

print("\nRunning 5-fold CV for ROC + calibration ...")

skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 300)

fold_tprs, fold_aucs = [], []
oof_probs, oof_labels = [], []

FOLD_COLORS = ["#E15759", "#4E79A7", "#59A14F", "#F28E2B", "#B07AA1"]

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5 ...")

    X_tr = X[tr_idx].copy()
    X_te = X[te_idx].copy()
    y_tr, y_te = y[tr_idx], y[te_idx]

    mcmc_cv = MCMC(NUTS(logistic_model),
                   num_warmup=1000, num_samples=2000,
                   progress_bar=False)
    mcmc_cv.run(random.PRNGKey(fold), jnp.array(X_tr), jnp.array(y_tr))

    s_cv      = mcmc_cv.get_samples()
    logits_cv = (s_cv["alpha"][:, None]
                 + (s_cv["beta"][:, None, :] * X_te[None, :, :]).sum(axis=-1))
    prob_cv   = expit(logits_cv).mean(axis=0)

    fpr_f, tpr_f, _ = roc_curve(y_te, prob_cv)
    fold_aucs.append(auc(fpr_f, tpr_f))
    fold_tprs.append(np.interp(mean_fpr, fpr_f, tpr_f))
    fold_tprs[-1][0] = 0.0

    oof_probs.extend(prob_cv.tolist())
    oof_labels.extend(y_te.tolist())
    print(f"    AUC = {fold_aucs[-1]:.3f}")

cv_aucs   = np.array(fold_aucs)
mean_tpr  = np.mean(fold_tprs, axis=0);  mean_tpr[-1] = 1.0
std_tpr   = np.std(fold_tprs,  axis=0)
mean_auc  = cv_aucs.mean()

pd.DataFrame({"fold": range(1, 6), "ROC_AUC": cv_aucs}).to_csv(
    S1_DIR / "stage1_cv_auc_folds.csv", index=False)
print(f"  CV AUC: {mean_auc:.3f} ± {cv_aucs.std():.3f}")

# ── Plot ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5.5))

for i, tpr_f in enumerate(fold_tprs):
    ax.plot(mean_fpr, tpr_f,
            color=FOLD_COLORS[i], linewidth=1.4, alpha=0.75,
            label=f"Fold {i+1}  (AUC = {fold_aucs[i]:.2f})")

ax.plot(mean_fpr, mean_tpr,
        color="black", linewidth=2.5,
        label=f"Mean  (AUC = {mean_auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color="#AAAAAA",
        linewidth=1.4, label="Random classifier")

ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE)
ax.set_ylabel("True Positive Rate",  fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(S1_DIR / "stage1_roc_curve_cv.png", dpi=300, bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_roc_curve_cv.pdf", format="pdf", bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_roc_curve_cv.eps", format="eps", bbox_inches="tight")
plt.close()
print("  ROC curve saved.")

# ============================================================
# Validation: Calibration Curve — pooled OOF predictions
# ============================================================

oof_probs_arr  = np.array(oof_probs)
oof_labels_arr = np.array(oof_labels)

N_CAL_BINS = 5
q_edges    = np.unique(
    np.quantile(oof_probs_arr, np.linspace(0, 1, N_CAL_BINS + 1))
)
n_bins = len(q_edges) - 1

prob_true_cv, prob_pred_cv = calibration_curve(
    oof_labels_arr, oof_probs_arr,
    n_bins=n_bins, strategy="quantile"
)

oof_df      = pd.DataFrame({"y_true": oof_labels_arr, "prob": oof_probs_arr})
oof_df["bin"] = pd.cut(oof_df["prob"], bins=q_edges,
                        include_lowest=True, labels=False)

ci_lo, ci_hi, bin_counts = [], [], []
for b in range(n_bins):
    grp = oof_df.loc[oof_df["bin"] == b, "y_true"]
    n   = len(grp);  p = grp.mean() if n > 0 else 0.0
    bin_counts.append(n)
    if n > 0:
        z = 1.96;  denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        ci_lo.append(center - margin);  ci_hi.append(center + margin)
    else:
        ci_lo.append(0.0);  ci_hi.append(0.0)

ci_lo = np.array(ci_lo);  ci_hi = np.array(ci_hi)

min_len       = min(len(prob_true_cv), len(prob_pred_cv), len(ci_lo), len(ci_hi))
prob_true_cv  = prob_true_cv[:min_len]
prob_pred_cv  = prob_pred_cv[:min_len]
ci_lo         = ci_lo[:min_len];  ci_hi = ci_hi[:min_len]

mask         = prob_pred_cv > 0.001
prob_true_cv = prob_true_cv[mask] * 100
prob_pred_cv = prob_pred_cv[mask] * 100
ci_lo        = ci_lo[mask] * 100
ci_hi        = ci_hi[mask] * 100

avg_n = int(len(oof_probs_arr) / N_CAL_BINS)
plot_max = max(prob_pred_cv.max(), ci_hi.max()) * 1.1

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot([0, plot_max], [0, plot_max], "--", color="#AAAAAA",
        linewidth=1.5, label="Perfect calibration")
ax.errorbar(prob_pred_cv, prob_true_cv,
            yerr=[np.clip(prob_true_cv - ci_lo, 0, None),
                  np.clip(ci_hi - prob_true_cv, 0, None)],
            fmt="o-", color="#4DBBD5",
            linewidth=1.8, markersize=7, capsize=4, elinewidth=1.2,
            label="Observed event rate (95% CI)")
ax.text(0.65, 0.15,
        f"5-fold CV (pooled OOF)\nQuantile bins  (n ≈ {avg_n:,}/bin)",
        transform=ax.transAxes, fontsize=FONT_SIZE, color="#555555")
ax.set_xlim(0, plot_max);  ax.set_ylim(0, plot_max)
ax.set_xlabel("Predicted Probability (%)", fontsize=FONT_SIZE)
ax.set_ylabel("Observed Frequency (%)",    fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(S1_DIR / "stage1_calibration_curve_cv.png", dpi=300, bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_calibration_curve_cv.pdf", format="pdf", bbox_inches="tight")
fig.savefig(S1_DIR / "stage1_calibration_curve_cv.eps", format="eps", bbox_inches="tight")
plt.close()
print("  Calibration curve saved.")

# ============================================================
# Visualization — Probability Surface Plot
# ============================================================

print("\nBuilding probability surface ...")
 
alpha_s = np.array(samples["alpha"])
beta_s  = np.array(samples["beta"])
 
d_grid   = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 60)
cei_grid = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 60)
F1, F2   = np.meshgrid(d_grid, cei_grid)
 
surf_mean  = np.zeros_like(F1)
surf_lower = np.zeros_like(F1)
surf_upper = np.zeros_like(F1)
 
for i in range(F1.shape[0]):
    for j in range(F1.shape[1]):
        log_ij   = alpha_s + beta_s[:, 0]*F1[i, j] + beta_s[:, 1]*F2[i, j]
        p_ij     = expit(log_ij)
        surf_mean[i, j]  = p_ij.mean()
        surf_lower[i, j] = np.percentile(p_ij, 2.5)
        surf_upper[i, j] = np.percentile(p_ij, 97.5)
 
# ── Matplotlib 3D Surface — PNG + PDF (vector) ─────────────
 
fig3d = plt.figure(figsize=(16, 11))
ax3d  = fig3d.add_subplot(111, projection="3d")
 
surf_mpl = ax3d.plot_surface(
    F1, F2, surf_mean,
    cmap="viridis", alpha=1.0, rasterized=True, zorder=1
)
ax3d.plot_surface(
    F1, F2, surf_upper,
    cmap="Blues", alpha=0.35, rasterized=True, zorder=2
)
ax3d.plot_surface(
    F1, F2, surf_lower,
    cmap="Reds", alpha=0.35, rasterized=True, zorder=3
)
 
cbar3d = fig3d.colorbar(surf_mpl, ax=ax3d, shrink=0.5, aspect=10, pad=0.08)
cbar3d.set_label("Mean Probability", fontsize=20, labelpad=10)
cbar3d.ax.tick_params(labelsize=20)
cbar3d.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
)
 
ax3d.set_xlabel(r"$D_\mathrm{norm}$", fontsize=20, labelpad=12)
ax3d.set_ylabel("CEI",                fontsize=20, labelpad=12)
ax3d.set_zlabel("")
fig3d.text(0.15, 0.50, "Probability",
           va="center", ha="center", rotation=90, fontsize=20)
 
ax3d.tick_params(axis="x", labelsize=20)
ax3d.tick_params(axis="y", labelsize=20)
ax3d.tick_params(axis="z", labelsize=20)
 
ax3d.set_zlim(0, 1)
ax3d.zaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
)
ax3d.invert_xaxis()
ax3d.view_init(elev=18, azim=45)
 
legend_elements = [
    Patch(facecolor=plt.cm.Blues(0.6), alpha=0.35, label="Upper 95% CI"),
    Patch(facecolor=plt.cm.Reds(0.6),  alpha=0.35, label="Lower 95% CI"),
]
ax3d.legend(handles=legend_elements, fontsize=20,
            loc="upper right", frameon=False)
 
plt.subplots_adjust(left=0.25, right=0.88, bottom=0.05, top=0.95)
fig3d.canvas.draw()
 
fig3d.savefig(S1_DIR / "stage1_probability_surface.png",
              dpi=300, bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S1_DIR / "stage1_probability_surface.pdf",
              format="pdf", bbox_inches="tight", pad_inches=0.35)
plt.close(fig3d)
print("  Probability surface PNG + PDF saved.")
 
print("\nStage 1 complete. All outputs in:", S1_DIR)