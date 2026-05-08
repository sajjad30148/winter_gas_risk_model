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
                jax numpyro arviz

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
from scipy.special import expit
from scipy.stats import gaussian_kde

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

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
    "alpha"     : samples["alpha"],
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

# ============================================================
# ============================================================
#  STAGE 2 — Two-Part Derate / NAC Model (Bayesian)
# ============================================================
# ============================================================

print("\n" + "="*60)
print("STAGE 2 — Two-Part Derate / NAC Model")
print("="*60)

S2_DIR     = OUT_DIR / "results_stage2"
S2_LOG_DIR = S2_DIR / "logistic"
S2_LIN_DIR = S2_DIR / "linear"
S2_CMB_DIR = S2_DIR / "combined"
for _d in [S2_LOG_DIR, S2_LIN_DIR, S2_CMB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Features & targets ─────────────────────────────────────
# event_dataset_NY.csv columns used: D_norm, CEI, NAC_norm
# D_norm   : normalised demand at event start
# CEI      : Cold Exposure Index at event start (standardised)
# NAC_norm : normalised net available capacity (0 = full outage)

s2_data = event_df[["D_norm", "CEI", "NAC_norm"]].dropna().copy()
s2_data["NAC_norm"] = s2_data["NAC_norm"].astype(float)

x1_all = jnp.array(s2_data["D_norm"].values.astype(float))
x2_all = jnp.array(s2_data["CEI"].values.astype(float))
y_bin  = jnp.array((s2_data["NAC_norm"] > 0).astype(int))  # 1=derated, 0=outage

# Linear sub-model: derated events only (NAC_norm > 0)
derated_mask = s2_data["NAC_norm"] > 0
s2_lin       = s2_data[derated_mask].copy()
x1_lin       = jnp.array(s2_lin["D_norm"].values.astype(float))
x2_lin       = jnp.array(s2_lin["CEI"].values.astype(float))
y_lin        = jnp.array(s2_lin["NAC_norm"].values.astype(float))

print(f"  Stage 2 full dataset      : {len(s2_data):,} events  |  derate rate = {float(y_bin.mean())*100:.1f}%")
print(f"  Linear subset (derated)   : {len(s2_lin):,} events")

# ── Shared 50×50 grid for all Stage 2 surfaces ─────────────
x1_grid = np.linspace(float(x1_all.min()), float(x1_all.max()), 50)
x2_grid = np.linspace(float(x2_all.min()), float(x2_all.max()), 50)
G1, G2  = np.meshgrid(x1_grid, x2_grid)

# ── Shared legend patches (used across all Stage 2 surfaces) ─
legend_elements_s2 = [
    Patch(facecolor=plt.cm.Blues(0.6),   alpha=0.35, label="Upper 95% CI"),
    Patch(facecolor=plt.cm.Oranges(0.6), alpha=0.35, label="Lower 95% CI"),
]

# ============================================================
# Stage 2 NumpyRo Models
# ============================================================

def logistic_model_derate(x1, x2, y=None):
    """Bayesian logistic for P(derated | D_norm, CEI)."""
    intercept = numpyro.sample("intercept", dist.Normal(2.2, 1.0))
    beta_1    = numpyro.sample("beta_1",    dist.Normal(0.0, 1.0))
    beta_2    = numpyro.sample("beta_2",    dist.Normal(0.0, 1.0))
    logits    = intercept + beta_1 * x1 + beta_2 * x2
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def linear_model_nac(x1, x2, y=None):
    """Bayesian linear for E[NAC_norm | D_norm, CEI, derated]."""
    intercept = numpyro.sample("intercept", dist.Normal(0.85, 0.10))
    slope_x1  = numpyro.sample("slope_x1",  dist.Normal(0.00, 0.10))
    slope_x2  = numpyro.sample("slope_x2",  dist.Normal(0.00, 0.10))
    sigma     = numpyro.sample("sigma",      dist.Exponential(0.05))
    mu        = intercept + slope_x1 * x1 + slope_x2 * x2
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

# ============================================================
#  STAGE 2, Part 01 — Logistic Regression (Derate Probability)
# ============================================================

print("\n--- Stage 2, Part 01: Logistic Regression (Derate Probability) ---")

# ── 5-fold stratified CV → ROC + calibration ───────────────
skf_s2   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 300)

fold_tprs_s2, fold_aucs_s2 = [], []
oof_probs_s2, oof_labels_s2 = [], []

x1_np = np.array(x1_all)
x2_np = np.array(x2_all)
y_np  = np.array(y_bin)

for fold, (tr_idx, te_idx) in enumerate(
        skf_s2.split(np.column_stack([x1_np, x2_np]), y_np)):
    print(f"  Logistic fold {fold+1}/5 ...")
    x1_tr, x1_te = x1_all[tr_idx], x1_all[te_idx]
    x2_tr, x2_te = x2_all[tr_idx], x2_all[te_idx]
    y_tr,  y_te  = y_bin[tr_idx],  y_bin[te_idx]

    mcmc_cv = MCMC(NUTS(logistic_model_derate),
                   num_warmup=1000, num_samples=1000,
                   progress_bar=False)
    mcmc_cv.run(random.PRNGKey(fold), x1=x1_tr, x2=x2_tr, y=y_tr)
    sc = mcmc_cv.get_samples()

    lc      = (sc["intercept"][:, None]
               + sc["beta_1"][:, None] * x1_te[None, :]
               + sc["beta_2"][:, None] * x2_te[None, :])
    prob_cv = expit(np.array(lc)).mean(axis=0)

    fpr_f, tpr_f, _ = roc_curve(np.array(y_te), prob_cv)
    fold_aucs_s2.append(auc(fpr_f, tpr_f))
    fold_tprs_s2.append(np.interp(mean_fpr, fpr_f, tpr_f))
    fold_tprs_s2[-1][0] = 0.0

    oof_probs_s2.extend(prob_cv.tolist())
    oof_labels_s2.extend(np.array(y_te).tolist())
    print(f"    AUC = {fold_aucs_s2[-1]:.3f}")

cv_aucs_s2  = np.array(fold_aucs_s2)
mean_tpr_s2 = np.mean(fold_tprs_s2, axis=0)
mean_tpr_s2[-1] = 1.0
mean_auc_s2 = cv_aucs_s2.mean()

pd.DataFrame({"fold": range(1, 6), "ROC_AUC": cv_aucs_s2}).to_csv(
    S2_LOG_DIR / "stage2_logistic_cv_auc.csv", index=False)
print(f"  CV AUC: {mean_auc_s2:.3f} ± {cv_aucs_s2.std():.3f}")

# ── ROC plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for i, tpr_f in enumerate(fold_tprs_s2):
    ax.plot(mean_fpr, tpr_f,
            color=FOLD_COLORS[i], linewidth=1.4, alpha=0.75,
            label=f"Fold {i+1}  (AUC = {fold_aucs_s2[i]:.2f})")
ax.plot(mean_fpr, mean_tpr_s2,
        color="black", linewidth=2.5,
        label=f"Mean  (AUC = {mean_auc_s2:.2f})")
ax.plot([0, 1], [0, 1], "--", color="#AAAAAA",
        linewidth=1.4, label="Random classifier")
ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE)
ax.set_ylabel("True Positive Rate",  fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LOG_DIR / "stage2_logistic_roc_cv.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_roc_cv.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_roc_cv.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  ROC curve saved.")

# ── Calibration plot ───────────────────────────────────────
oof_probs_s2_arr  = np.array(oof_probs_s2)
oof_labels_s2_arr = np.array(oof_labels_s2)

N_CAL_S2   = 5
q_edges_s2 = np.unique(
    np.quantile(oof_probs_s2_arr, np.linspace(0, 1, N_CAL_S2 + 1)))
n_bins_s2  = len(q_edges_s2) - 1

prob_true_s2, prob_pred_s2 = calibration_curve(
    oof_labels_s2_arr, oof_probs_s2_arr,
    n_bins=n_bins_s2, strategy="quantile")

oof_df_s2       = pd.DataFrame({"y_true": oof_labels_s2_arr, "prob": oof_probs_s2_arr})
oof_df_s2["bin"] = pd.cut(oof_df_s2["prob"], bins=q_edges_s2,
                           include_lowest=True, labels=False)

ci_lo_s2, ci_hi_s2 = [], []
for b in range(n_bins_s2):
    grp = oof_df_s2.loc[oof_df_s2["bin"] == b, "y_true"]
    n   = len(grp);  p = grp.mean() if n > 0 else 0.0
    if n > 0:
        z = 1.96;  denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        ci_lo_s2.append(center - margin);  ci_hi_s2.append(center + margin)
    else:
        ci_lo_s2.append(0.0);  ci_hi_s2.append(0.0)
ci_lo_s2 = np.array(ci_lo_s2);  ci_hi_s2 = np.array(ci_hi_s2)

_ml = min(len(prob_true_s2), len(prob_pred_s2), len(ci_lo_s2), len(ci_hi_s2))
prob_true_s2 = prob_true_s2[:_ml];  prob_pred_s2 = prob_pred_s2[:_ml]
ci_lo_s2     = ci_lo_s2[:_ml];      ci_hi_s2     = ci_hi_s2[:_ml]
_mask_s2     = prob_pred_s2 > 0.001
prob_true_s2 = prob_true_s2[_mask_s2] * 100
prob_pred_s2 = prob_pred_s2[_mask_s2] * 100
ci_lo_s2     = ci_lo_s2[_mask_s2] * 100
ci_hi_s2     = ci_hi_s2[_mask_s2] * 100
avg_n_s2     = int(len(oof_probs_s2_arr) / N_CAL_S2)
plot_max_s2  = max(prob_pred_s2.max(), ci_hi_s2.max()) * 1.1

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot([0, plot_max_s2], [0, plot_max_s2], "--", color="#AAAAAA",
        linewidth=1.5, label="Perfect calibration")
ax.errorbar(prob_pred_s2, prob_true_s2,
            yerr=[np.clip(prob_true_s2 - ci_lo_s2, 0, None),
                  np.clip(ci_hi_s2 - prob_true_s2, 0, None)],
            fmt="o-", color="#4DBBD5",
            linewidth=1.8, markersize=7, capsize=4, elinewidth=1.2,
            label="Observed derate rate (95% CI)")
ax.text(0.65, 0.15,
        f"5-fold CV (pooled OOF)\nQuantile bins  (n ≈ {avg_n_s2:,}/bin)",
        transform=ax.transAxes, fontsize=FONT_SIZE, color="#555555")
ax.set_xlim(0, plot_max_s2);  ax.set_ylim(0, plot_max_s2)
ax.set_xlabel("Predicted Probability (%)", fontsize=FONT_SIZE)
ax.set_ylabel("Observed Frequency (%)",    fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LOG_DIR / "stage2_logistic_calibration_cv.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_calibration_cv.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_calibration_cv.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  Calibration curve saved.")

# ── Full-data MCMC fit (for posterior density + surfaces) ──
print("\nFull-data MCMC for Stage 2 logistic ...")
mcmc_log_full = MCMC(NUTS(logistic_model_derate),
                     num_warmup=1000, num_samples=1000, num_chains=2)
mcmc_log_full.run(random.PRNGKey(42), x1=x1_all, x2=x2_all, y=y_bin)
log_samp = mcmc_log_full.get_samples()

logistic_intercept = np.array(log_samp["intercept"])
logistic_beta1     = np.array(log_samp["beta_1"])
logistic_beta2     = np.array(log_samp["beta_2"])

np.save(S2_LOG_DIR / "stage2_logistic_intercept.npy", logistic_intercept)
np.save(S2_LOG_DIR / "stage2_logistic_beta1.npy",     logistic_beta1)
np.save(S2_LOG_DIR / "stage2_logistic_beta2.npy",     logistic_beta2)

# ── Posterior density — logistic ──────────────────────────
params_log = [
    (logistic_intercept, r"$\delta_0$ (Intercept)",                "#9ecae1"),
    (logistic_beta1,     r"$\delta_1$ (Slope $D_\mathrm{norm}$)", "#a1d99b"),
    (logistic_beta2,     r"$\delta_2$ (Slope CEI)",                "#fcae91"),
]
fig, axes = plt.subplots(3, 1, figsize=(7, 12))
for i, (data, label, color) in enumerate(params_log):
    ax = axes[i]
    ax.hist(data, bins=40, density=True, alpha=0.6,
            color=color, edgecolor="none")
    sns.kdeplot(data, ax=ax, color="#696868", linewidth=1.2)
    mean_val = np.mean(data);  std_val = np.std(data)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)
    ax.text(0.97, 0.95,
            f"Mean = {mean_val:.3f}\nStd = {std_val:.3f}",
            transform=ax.transAxes, color="red",
            fontsize=FONT_SIZE, ha="right", va="top")
    ax.set_title(label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density" if i == 0 else "", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LOG_DIR / "stage2_logistic_posterior_density.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_posterior_density.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LOG_DIR / "stage2_logistic_posterior_density.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  Logistic posterior density saved.")

# ── Derate probability surface ────────────────────────────
print("\nBuilding derate probability surface ...")

Z_mean_log  = np.zeros_like(G1)
Z_lower_log = np.zeros_like(G1)
Z_upper_log = np.zeros_like(G1)

for i in range(G1.shape[0]):
    for j in range(G1.shape[1]):
        logit_ij = (logistic_intercept
                    + logistic_beta1 * G1[i, j]
                    + logistic_beta2 * G2[i, j])
        p_ij = expit(logit_ij)
        Z_mean_log[i, j]  = p_ij.mean()
        Z_lower_log[i, j] = np.percentile(p_ij, 2.5)
        Z_upper_log[i, j] = np.percentile(p_ij, 97.5)

np.save(S2_LOG_DIR / "stage2_derate_surface.npy",
        np.stack([Z_mean_log, Z_lower_log, Z_upper_log]))

fig3d = plt.figure(figsize=(16, 11))
ax3d  = fig3d.add_subplot(111, projection="3d")
sm    = ax3d.plot_surface(G1, G2, Z_mean_log,  cmap="viridis",  alpha=1.0,  rasterized=True, zorder=1)
ax3d.plot_surface(G1, G2, Z_upper_log,  cmap="Blues",   alpha=0.35, rasterized=True, zorder=2)
ax3d.plot_surface(G1, G2, Z_lower_log,  cmap="Oranges", alpha=0.35, rasterized=True, zorder=3)
cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label("P(Derated)", fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3d.set_xlabel(r"$D_\mathrm{norm}$", fontsize=20, labelpad=12)
ax3d.set_ylabel("CEI",                fontsize=20, labelpad=12)
ax3d.set_zlabel("")
fig3d.text(0.15, 0.50, "P(Derated)",
           va="center", ha="center", rotation=90, fontsize=20)
ax3d.tick_params(axis="x", labelsize=20)
ax3d.tick_params(axis="y", labelsize=20)
ax3d.tick_params(axis="z", labelsize=20)
ax3d.set_zlim(0, 1)
ax3d.zaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3d.invert_xaxis()
ax3d.view_init(elev=18, azim=45)
ax3d.legend(handles=legend_elements_s2, fontsize=20,
            loc="upper right", frameon=False)
plt.subplots_adjust(left=0.18, right=0.88, bottom=0.05, top=0.95)
fig3d.canvas.draw()
fig3d.savefig(S2_LOG_DIR / "stage2_derate_surface.png",
              dpi=300, bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LOG_DIR / "stage2_derate_surface.pdf",
              format="pdf", bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LOG_DIR / "stage2_derate_surface.eps",
              format="eps", bbox_inches="tight", pad_inches=0.35)
plt.close(fig3d)
print("  Derate probability surface saved.")

# ── Outage probability surface  (P_outage = 1 − P_derate) ──
Z_mean_out  = 1.0 - Z_mean_log
Z_upper_out = 1.0 - Z_lower_log   # upper outage CI = 1 − lower derate CI
Z_lower_out = 1.0 - Z_upper_log

np.save(S2_LOG_DIR / "stage2_outage_surface.npy",
        np.stack([Z_mean_out, Z_lower_out, Z_upper_out]))

fig3d = plt.figure(figsize=(16, 11))
ax3d  = fig3d.add_subplot(111, projection="3d")
sm    = ax3d.plot_surface(G1, G2, Z_mean_out,  cmap="viridis",  alpha=1.0,  rasterized=True, zorder=1)
ax3d.plot_surface(G1, G2, Z_upper_out, cmap="Blues",   alpha=0.35, rasterized=True, zorder=2)
ax3d.plot_surface(G1, G2, Z_lower_out, cmap="Oranges", alpha=0.35, rasterized=True, zorder=3)
cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label("P(Outage)", fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3d.set_xlabel(r"$D_\mathrm{norm}$", fontsize=20, labelpad=12)
ax3d.set_ylabel("CEI",                fontsize=20, labelpad=12)
ax3d.set_zlabel("")
fig3d.text(0.15, 0.50, "P(Outage)",
           va="center", ha="center", rotation=90, fontsize=20)
ax3d.tick_params(axis="x", labelsize=20)
ax3d.tick_params(axis="y", labelsize=20)
ax3d.tick_params(axis="z", labelsize=20)
ax3d.set_zlim(0, 1)
ax3d.zaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3d.invert_xaxis()
ax3d.view_init(elev=18, azim=45)
ax3d.legend(handles=legend_elements_s2, fontsize=20,
            loc="upper right", frameon=False)
plt.subplots_adjust(left=0.18, right=0.88, bottom=0.05, top=0.95)
fig3d.canvas.draw()
fig3d.savefig(S2_LOG_DIR / "stage2_outage_surface.png",
              dpi=300, bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LOG_DIR / "stage2_outage_surface.pdf",
              format="pdf", bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LOG_DIR / "stage2_outage_surface.eps",
              format="eps", bbox_inches="tight", pad_inches=0.35)
plt.close(fig3d)
print("  Outage probability surface saved.")

# ============================================================
#  STAGE 2, Part 02 — Linear Regression (NAC | Derated)
# ============================================================

print("\n--- Stage 2, Part 02: Linear Regression (NAC | Derated) ---")

# ── Full-data MCMC (for PPC, PSIS-LOO, posterior, surfaces) ─
print("\nFull-data MCMC for Stage 2 linear ...")
mcmc_lin_full = MCMC(NUTS(linear_model_nac),
                     num_warmup=1000, num_samples=1000, num_chains=2)
mcmc_lin_full.run(random.PRNGKey(42), x1=x1_lin, x2=x2_lin, y=y_lin)
lin_samp = mcmc_lin_full.get_samples()

linear_intercept = np.array(lin_samp["intercept"])
linear_slope1    = np.array(lin_samp["slope_x1"])
linear_slope2    = np.array(lin_samp["slope_x2"])
linear_sigma     = np.array(lin_samp["sigma"])

np.save(S2_LIN_DIR / "stage2_linear_intercept.npy", linear_intercept)
np.save(S2_LIN_DIR / "stage2_linear_slope1.npy",    linear_slope1)
np.save(S2_LIN_DIR / "stage2_linear_slope2.npy",    linear_slope2)
np.save(S2_LIN_DIR / "stage2_linear_sigma.npy",     linear_sigma)

# ── PSIS-LOO diagnostic ────────────────────────────────────
print("\nRunning PSIS-LOO for Stage 2 linear ...")
idata_lin = az.from_numpyro(mcmc_lin_full)
loo_lin   = az.loo(idata_lin, pointwise=True)
print(loo_lin)

pareto_k = np.array(loo_lin.pareto_k)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(range(len(pareto_k)), pareto_k,
           color="#4E79A7", s=18, alpha=0.7)
ax.axhline(0.5, color="#E15759", linestyle="--",
           linewidth=1.5, label="k = 0.5")
ax.axhline(0.7, color="#F28E2B", linestyle=":",
           linewidth=1.5, label="k = 0.7")
ax.set_xlabel("Observation Index", fontsize=FONT_SIZE)
ax.set_ylabel("Pareto k",          fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LIN_DIR / "stage2_linear_pareto_k.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_pareto_k.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_pareto_k.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  Pareto-k plot saved.")

# ── Posterior Predictive Check (PPC) — 300 draws ──────────
print("\nRunning PPC for Stage 2 linear ...")
n_ppc   = 300
n_post  = len(linear_intercept)
ppc_idx = np.random.choice(n_post, size=n_ppc, replace=False)

y_obs_np = np.array(y_lin)
x1_np_lin = np.array(x1_lin)
x2_np_lin = np.array(x2_lin)
xs_kde    = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(7, 4.5))
for idx in ppc_idx:
    mu_draw = (linear_intercept[idx]
               + linear_slope1[idx] * x1_np_lin
               + linear_slope2[idx] * x2_np_lin)
    y_draw  = np.random.normal(mu_draw, linear_sigma[idx])
    y_draw  = np.clip(y_draw, 0, 1)      # physical bound [0,1]
    kde_draw = gaussian_kde(y_draw)
    ax.plot(xs_kde, kde_draw(xs_kde),
            color="#AAAAAA", linewidth=0.5, alpha=0.4)

kde_obs = gaussian_kde(y_obs_np)
ax.plot(xs_kde, kde_obs(xs_kde),
        color="#E15759", linewidth=2.5, label="Observed")

ax.set_xlabel(r"$\mathrm{NAC}_{\mathrm{norm}}$", fontsize=FONT_SIZE)
ax.set_ylabel("Density",                          fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)
ax.legend(frameon=False, fontsize=FONT_SIZE)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LIN_DIR / "stage2_linear_ppc.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_ppc.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_ppc.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  PPC plot saved.")

# ── Posterior density — linear ─────────────────────────────
params_lin = [
    (linear_intercept, r"$\gamma_0$ (Intercept)",                "#9ecae1"),
    (linear_slope1,    r"$\gamma_1$ (Slope $D_\mathrm{norm}$)", "#a1d99b"),
    (linear_slope2,    r"$\gamma_2$ (Slope CEI)",                "#9ecae1"),
    (linear_sigma,     r"$\sigma$ (Sigma)",                       "#fcae91"),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()
for i, (data, label, color) in enumerate(params_lin):
    ax = axes_flat[i]
    ax.hist(data, bins=40, density=True, alpha=0.6,
            color=color, edgecolor="none")
    sns.kdeplot(data, ax=ax, color="#696868", linewidth=1.2)
    mean_val = np.mean(data);  std_val = np.std(data)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)
    ax.text(0.97, 0.95,
            f"Mean = {mean_val:.3f}\nStd = {std_val:.3f}",
            transform=ax.transAxes, color="red",
            fontsize=FONT_SIZE, ha="right", va="top")
    ax.set_title(label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density" if i % 2 == 0 else "", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(S2_LIN_DIR / "stage2_linear_posterior_density.png",  dpi=300, bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_posterior_density.pdf",  format="pdf", bbox_inches="tight")
fig.savefig(S2_LIN_DIR / "stage2_linear_posterior_density.eps",  format="eps", bbox_inches="tight")
plt.close()
print("  Linear posterior density saved.")

# ── NAC surface  E[NAC_norm | D_norm, CEI, derated] ───────
print("\nBuilding NAC surface ...")

Z_mean_nac  = np.zeros_like(G1)
Z_lower_nac = np.zeros_like(G1)
Z_upper_nac = np.zeros_like(G1)

for i in range(G1.shape[0]):
    for j in range(G1.shape[1]):
        mu_ij = (linear_intercept
                 + linear_slope1 * G1[i, j]
                 + linear_slope2 * G2[i, j])
        mu_clipped        = np.clip(mu_ij, 0, 1)
        Z_mean_nac[i, j]  = mu_clipped.mean()
        Z_lower_nac[i, j] = np.percentile(mu_clipped, 2.5)
        Z_upper_nac[i, j] = np.percentile(mu_clipped, 97.5)

np.save(S2_LIN_DIR / "stage2_nac_surface.npy",
        np.stack([Z_mean_nac, Z_lower_nac, Z_upper_nac]))

fig3d = plt.figure(figsize=(16, 11))
ax3d  = fig3d.add_subplot(111, projection="3d")
sm    = ax3d.plot_surface(G1, G2, Z_mean_nac,  cmap="viridis",  alpha=1.0,  rasterized=True, zorder=1)
ax3d.plot_surface(G1, G2, Z_upper_nac, cmap="Blues",   alpha=0.35, rasterized=True, zorder=2)
ax3d.plot_surface(G1, G2, Z_lower_nac, cmap="Oranges", alpha=0.35, rasterized=True, zorder=3)
cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label(r"$\mathbb{E}[\mathrm{NAC}_{\mathrm{norm}}]$", fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=20)
ax3d.set_xlabel(r"$D_\mathrm{norm}$", fontsize=20, labelpad=12)
ax3d.set_ylabel("CEI",                fontsize=20, labelpad=12)
ax3d.set_zlabel("")
fig3d.text(0.15, 0.50, r"$\mathbb{E}[\mathrm{NAC}_{\mathrm{norm}}]$",
           va="center", ha="center", rotation=90, fontsize=20)
ax3d.tick_params(axis="x", labelsize=20)
ax3d.tick_params(axis="y", labelsize=20)
ax3d.tick_params(axis="z", labelsize=20)
ax3d.set_zlim(0, 1)
ax3d.invert_xaxis()
ax3d.view_init(elev=18, azim=45)
ax3d.legend(handles=legend_elements_s2, fontsize=20,
            loc="upper right", frameon=False)
plt.subplots_adjust(left=0.18, right=0.88, bottom=0.05, top=0.95)
fig3d.canvas.draw()
fig3d.savefig(S2_LIN_DIR / "stage2_nac_surface.png",
              dpi=300, bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LIN_DIR / "stage2_nac_surface.pdf",
              format="pdf", bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_LIN_DIR / "stage2_nac_surface.eps",
              format="eps", bbox_inches="tight", pad_inches=0.35)
plt.close(fig3d)
print("  NAC surface saved.")

# ============================================================
#  STAGE 2, Part 03 — Combined Output
#  E[combined] = P(derated) × E[NAC_norm | derated]
# ============================================================

print("\n--- Stage 2, Part 03: Combined Output ---")
print("\nBuilding combined output surface ...")

N_CMB  = 100          # Monte Carlo draws per grid point
n_post = len(logistic_intercept)

Z_mean_cmb  = np.zeros_like(G1)
Z_lower_cmb = np.zeros_like(G1)
Z_upper_cmb = np.zeros_like(G1)

for i in range(G1.shape[0]):
    for j in range(G1.shape[1]):
        x1_ij = G1[i, j];  x2_ij = G2[i, j]
        draws = []
        for _ in range(N_CMB):
            idx = np.random.randint(0, n_post)
            # logistic: P(derated)
            logit_val = (logistic_intercept[idx]
                         + logistic_beta1[idx] * x1_ij
                         + logistic_beta2[idx] * x2_ij)
            p_derate  = 1.0 / (1.0 + np.exp(-float(logit_val)))
            # linear: E[NAC|derated] + noise
            mu_nac    = (linear_intercept[idx]
                         + linear_slope1[idx] * x1_ij
                         + linear_slope2[idx] * x2_ij)
            nac_draw  = np.random.normal(float(mu_nac), float(linear_sigma[idx]))
            draws.append(p_derate * float(np.clip(nac_draw, 0, 1)))
        Z_mean_cmb[i, j]  = np.mean(draws)
        Z_lower_cmb[i, j] = np.percentile(draws, 2.5)
        Z_upper_cmb[i, j] = np.percentile(draws, 97.5)

np.save(S2_CMB_DIR / "stage2_combined_surface.npy",
        np.stack([Z_mean_cmb, Z_lower_cmb, Z_upper_cmb]))

fig3d = plt.figure(figsize=(16, 11))
ax3d  = fig3d.add_subplot(111, projection="3d")
sm    = ax3d.plot_surface(G1, G2, Z_mean_cmb,  cmap="viridis",  alpha=1.0,  rasterized=True, zorder=1)
ax3d.plot_surface(G1, G2, Z_upper_cmb, cmap="Blues",   alpha=0.35, rasterized=True, zorder=2)
ax3d.plot_surface(G1, G2, Z_lower_cmb, cmap="Oranges", alpha=0.35, rasterized=True, zorder=3)
cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label(r"$\mathbb{E}[\mathrm{NAC}_{\mathrm{norm}}]$", fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=20)
ax3d.set_xlabel(r"$D_\mathrm{norm}$", fontsize=20, labelpad=12)
ax3d.set_ylabel("CEI",                fontsize=20, labelpad=12)
ax3d.set_zlabel("")
fig3d.text(0.15, 0.50, r"$\mathbb{E}[\mathrm{NAC}_{\mathrm{norm}}]$",
           va="center", ha="center", rotation=90, fontsize=20)
ax3d.tick_params(axis="x", labelsize=20)
ax3d.tick_params(axis="y", labelsize=20)
ax3d.tick_params(axis="z", labelsize=20)
ax3d.set_zlim(0, 1)
ax3d.invert_xaxis()
ax3d.view_init(elev=18, azim=45)
ax3d.legend(handles=legend_elements_s2, fontsize=20,
            loc="upper right", frameon=False)
plt.subplots_adjust(left=0.18, right=0.88, bottom=0.05, top=0.95)
fig3d.canvas.draw()
fig3d.savefig(S2_CMB_DIR / "stage2_combined_surface.png",
              dpi=300, bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_CMB_DIR / "stage2_combined_surface.pdf",
              format="pdf", bbox_inches="tight", pad_inches=0.35)
fig3d.savefig(S2_CMB_DIR / "stage2_combined_surface.eps",
              format="eps", bbox_inches="tight", pad_inches=0.35)
plt.close(fig3d)
print("  Combined output surface saved.")

print("\nStage 2 complete. All outputs in:", S2_DIR)

# ============================================================
# ============================================================
#  STAGE 3 — Bayesian Event Duration (Lognormal Duration Model)
# ============================================================
# ============================================================
#
# Two separate lognormal models:
#   Outage  (NAC_norm = 0): log(duration) ~ Normal(theta_0, sigma)
#   Derate  (NAC_norm > 0): log(duration) ~ Normal(theta_0 + theta_1*severity, sigma)
#                           severity = 1 - NAC_norm
#
# duration column: 'duration (hh:mm)' in hh:mm format
# ============================================================

print("\n" + "="*60)
print("STAGE 3 — Event Duration Models")
print("="*60)

S3_DIR = OUT_DIR / "results_stage3"
S3_DIR.mkdir(parents=True, exist_ok=True)

# ── HH:MM → decimal hours ──────────────────────────────────
def _hhmm_to_hours(x):
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    x = str(x).strip()
    if x == "":
        return np.nan
    if ":" not in x:
        try:
            return float(x)
        except Exception:
            return np.nan
    try:
        hh, mm = x.split(":")
        return int(hh) + int(mm) / 60.0
    except Exception:
        return np.nan

# ── Data preparation ───────────────────────────────────────
s3_raw = event_df[["NAC_norm", "duration (hh:mm)"]].dropna().copy()
s3_raw["NAC_norm"]          = pd.to_numeric(s3_raw["NAC_norm"], errors="coerce")
s3_raw["EventDuration_hrs"] = s3_raw["duration (hh:mm)"].apply(_hhmm_to_hours)
s3_raw = s3_raw.dropna(subset=["NAC_norm", "EventDuration_hrs"]).copy()
s3_raw = s3_raw[s3_raw["EventDuration_hrs"] > 0].copy()
s3_raw["log_duration"]     = np.log(s3_raw["EventDuration_hrs"])
s3_raw["Severity"]         = 1.0 - s3_raw["NAC_norm"]

data_outage_s3 = s3_raw[s3_raw["NAC_norm"] == 0].copy().reset_index(drop=True)
data_derate_s3 = s3_raw[s3_raw["NAC_norm"] >  0].copy().reset_index(drop=True)

print(f"  Stage 3 events total  : {len(s3_raw):,}")
print(f"    Outage (NAC=0)      : {len(data_outage_s3):,}")
print(f"    Derate (NAC>0)      : {len(data_derate_s3):,}")

# ── NumpyRo Models ─────────────────────────────────────────
def outage_duration_model(y=None):
    """log(duration) ~ Normal(theta_0, sigma)  [intercept-only]"""
    theta_0 = numpyro.sample("theta_0", dist.Normal(0.0, 5.0))
    sigma   = numpyro.sample("sigma",   dist.HalfNormal(2.0))
    numpyro.sample("obs", dist.Normal(theta_0, sigma), obs=y)


def derate_duration_model(x, y=None):
    """log(duration) ~ Normal(theta_0 + theta_1*severity, sigma)"""
    theta_0 = numpyro.sample("theta_0", dist.Normal(0.0, 5.0))
    theta_1 = numpyro.sample("theta_1", dist.Normal(0.0, 5.0))
    sigma   = numpyro.sample("sigma",   dist.HalfNormal(2.0))
    numpyro.sample("obs", dist.Normal(theta_0 + theta_1 * x, sigma), obs=y)

# ── Constants ──────────────────────────────────────────────
_N_SPLITS   = 5
_NUM_WARMUP = 1000
_NUM_SAMP   = 2000

# ============================================================
#  STAGE 3 — Outage Duration Model
# ============================================================

print("\n--- Stage 3: Outage Duration Model ---")

outage_results_s3 = None

if len(data_outage_s3) >= _N_SPLITS:
    y_hrs_out = data_outage_s3["EventDuration_hrs"].values.astype(float)
    y_log_out = data_outage_s3["log_duration"].values.astype(float)

    kf_out      = KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)
    oof_out     = []
    theta0_out  = []
    sigma_out   = []

    for fold, (tr, te) in enumerate(kf_out.split(y_log_out)):
        print(f"  Outage fold {fold+1}/{_N_SPLITS} ...")
        mcmc_o = MCMC(NUTS(outage_duration_model),
                      num_warmup=_NUM_WARMUP, num_samples=_NUM_SAMP,
                      progress_bar=False)
        mcmc_o.run(random.PRNGKey(fold), y=jnp.array(y_log_out[tr]))
        s_o = mcmc_o.get_samples()
        t0  = np.array(s_o["theta_0"])
        sg  = np.array(s_o["sigma"])
        theta0_out.append(t0)
        sigma_out.append(sg)

        # Lognormal mean: E[T] = exp(mu + sigma^2/2)
        mu_pred = t0[:, None]
        y_pred_samp = np.exp(mu_pred + (sg[:, None] ** 2) / 2.0)
        y_pred_mean  = np.repeat(y_pred_samp.mean(axis=0)[0],    len(te))
        y_pred_lower = np.repeat(np.percentile(y_pred_samp, 2.5,  axis=0)[0], len(te))
        y_pred_upper = np.repeat(np.percentile(y_pred_samp, 97.5, axis=0)[0], len(te))
        oof_out.append(pd.DataFrame({
            "fold":                      fold + 1,
            "Observed_EventDuration_hrs": y_hrs_out[te],
            "Predicted_EventDuration_hrs":y_pred_mean,
            "Predicted_Lower_95CI":      y_pred_lower,
            "Predicted_Upper_95CI":      y_pred_upper,
        }))

    theta0_out = np.concatenate(theta0_out)
    sigma_out  = np.concatenate(sigma_out)

    # ── Save .npz ─────────────────────────────────────────
    np.savez(S3_DIR / "outage_combined_posterior_samples.npz",
             theta_0=theta0_out, sigma=sigma_out)

    # ── Save posterior CSV ────────────────────────────────
    pd.DataFrame({"theta_0": theta0_out, "sigma": sigma_out}).to_csv(
        S3_DIR / "outage_posterior_samples.csv", index=False)

    # ── Save CV predictions CSV ───────────────────────────
    cv_out = pd.concat(oof_out, ignore_index=True)
    cv_out.to_csv(S3_DIR / "outage_cv_predictions.csv", index=False)
    print(f"  Outage CV predictions saved  ({len(cv_out):,} rows).")

    # ── Duration histogram ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(y_hrs_out, bins=40, stat="count", kde=True,
                 edgecolor="black", alpha=0.75, ax=ax)
    ax.set_xlabel("Event Duration (hours)", fontsize=FONT_SIZE)
    ax.set_ylabel("Count",                  fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "outage_duration_histogram.png",  dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_duration_histogram.pdf",  format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_duration_histogram.eps",  format="eps", bbox_inches="tight")
    plt.close()
    print("  Outage duration histogram saved.")

    # ── Posterior density ─────────────────────────────────
    params_out = [
        (theta0_out, r"$\theta_0$", "#9ecae1"),
        (sigma_out,  r"$\sigma$",   "#fcae91"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    for i, (data, label, color) in enumerate(params_out):
        ax = axes[i]
        ax.hist(data, bins=40, density=True, alpha=0.6,
                color=color, edgecolor="none")
        sns.kdeplot(data, ax=ax, color="#696868", linewidth=1.2)
        mv = np.mean(data);  sv = np.std(data)
        ax.axvline(mv, color="red", linestyle="--", linewidth=2)
        ax.text(0.97, 0.95, f"Mean = {mv:.3f}\nStd = {sv:.3f}",
                transform=ax.transAxes, color="red",
                fontsize=FONT_SIZE, ha="right", va="top")
        ax.set_title(label, fontsize=FONT_SIZE)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "outage_posterior_density_combined.png",  dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_posterior_density_combined.pdf",  format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_posterior_density_combined.eps",  format="eps", bbox_inches="tight")
    plt.close()
    print("  Outage posterior density saved.")

    # ── Event duration vs capacity-loss plot ──────────────
    # Outage: severity = 1.0 for all events — scatter at x=1
    # Show posterior predictive mean + 95% CI as horizontal band
    outage_mean_duration = np.exp(theta0_out + (sigma_out ** 2) / 2.0)
    rec_mean  = outage_mean_duration.mean()
    rec_lower = np.percentile(outage_mean_duration, 2.5)
    rec_upper = np.percentile(outage_mean_duration, 97.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.ones(len(y_hrs_out)), y_hrs_out,
               alpha=0.5, label="Observed", color="#4E79A7")
    ax.axhline(rec_mean,  color="red", linewidth=2.5,
               label=f"Predicted mean = {rec_mean:.1f} h")
    ax.fill_between([0.8, 1.2], rec_lower, rec_upper,
                    color="red", alpha=0.2,
                    label="95% Credible Interval")
    ax.set_xlim(0.7, 1.3)
    ax.set_xticks([1.0])
    ax.set_xticklabels([r"$\Delta_{\mathrm{cap}}=1$  (full outage)"],
                       fontsize=FONT_SIZE)
    ax.set_xlabel(r"$\Delta_{\mathrm{cap}} = 1 - \mathrm{NAC}_{\mathrm{norm}}$",
                  fontsize=FONT_SIZE)
    ax.set_ylabel("Event Duration (hours)", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.legend(frameon=False, fontsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "outage_event_duration_vs_capacityloss_plot.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_event_duration_vs_capacityloss_plot.pdf",
                format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "outage_event_duration_vs_capacityloss_plot.eps",
                format="eps", bbox_inches="tight")
    plt.close()
    print("  Outage event duration vs capacity-loss plot saved.")

    outage_results_s3 = {"cv_predictions": cv_out,
                         "theta_0_all": theta0_out,
                         "sigma_all":   sigma_out}

else:
    print(f"  Not enough outage rows ({len(data_outage_s3)}) for {_N_SPLITS}-fold CV. Skipping.")

# ============================================================
#  STAGE 3 — Derate Duration Model
# ============================================================

print("\n--- Stage 3: Derate Duration Model ---")

derate_results_s3 = None

if len(data_derate_s3) >= _N_SPLITS:
    X_sev     = data_derate_s3["Severity"].values.astype(float)
    y_hrs_der = data_derate_s3["EventDuration_hrs"].values.astype(float)
    y_log_der = data_derate_s3["log_duration"].values.astype(float)

    kf_der     = KFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)
    oof_der    = []
    theta0_der = []
    theta1_der = []
    sigma_der  = []

    for fold, (tr, te) in enumerate(kf_der.split(X_sev)):
        print(f"  Derate fold {fold+1}/{_N_SPLITS} ...")
        mcmc_d = MCMC(NUTS(derate_duration_model),
                      num_warmup=_NUM_WARMUP, num_samples=_NUM_SAMP,
                      progress_bar=False)
        mcmc_d.run(random.PRNGKey(fold + 100),
                   x=jnp.array(X_sev[tr]), y=jnp.array(y_log_der[tr]))
        s_d = mcmc_d.get_samples()
        t0  = np.array(s_d["theta_0"])
        t1  = np.array(s_d["theta_1"])
        sg  = np.array(s_d["sigma"])
        theta0_der.append(t0)
        theta1_der.append(t1)
        sigma_der.append(sg)

        mu_test = t0[:, None] + t1[:, None] * X_sev[te][None, :]
        y_pred_samp  = np.exp(mu_test + (sg[:, None] ** 2) / 2.0)
        oof_der.append(pd.DataFrame({
            "fold":                      fold + 1,
            "Severity":                  X_sev[te],
            "NAC_norm":                  data_derate_s3["NAC_norm"].values[te],
            "Observed_EventDuration_hrs": y_hrs_der[te],
            "Predicted_EventDuration_hrs":y_pred_samp.mean(axis=0),
            "Predicted_Lower_95CI":      np.percentile(y_pred_samp, 2.5,  axis=0),
            "Predicted_Upper_95CI":      np.percentile(y_pred_samp, 97.5, axis=0),
        }))

    theta0_der = np.concatenate(theta0_der)
    theta1_der = np.concatenate(theta1_der)
    sigma_der  = np.concatenate(sigma_der)

    # ── Save .npz ─────────────────────────────────────────
    np.savez(S3_DIR / "derate_combined_posterior_samples.npz",
             theta_0=theta0_der, theta_1=theta1_der, sigma=sigma_der)

    # ── Save posterior CSV ────────────────────────────────
    pd.DataFrame({"theta_0": theta0_der,
                  "theta_1": theta1_der,
                  "sigma":   sigma_der}).to_csv(
        S3_DIR / "derate_posterior_samples.csv", index=False)

    # ── Save CV predictions CSV ───────────────────────────
    cv_der = pd.concat(oof_der, ignore_index=True)
    cv_der.to_csv(S3_DIR / "derate_cv_predictions.csv", index=False)
    print(f"  Derate CV predictions saved  ({len(cv_der):,} rows).")

    # ── Duration histogram ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(y_hrs_der, bins=40, stat="count", kde=True,
                 edgecolor="black", alpha=0.75, ax=ax)
    ax.set_xlabel("Event Duration (hours)", fontsize=FONT_SIZE)
    ax.set_ylabel("Count",                  fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "derate_duration_histogram.png",  dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_duration_histogram.pdf",  format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_duration_histogram.eps",  format="eps", bbox_inches="tight")
    plt.close()
    print("  Derate duration histogram saved.")

    # ── Posterior density ─────────────────────────────────
    params_der = [
        (theta0_der, r"$\theta_0$ (Intercept)",                 "#9ecae1"),
        (theta1_der, r"$\theta_1$ (Slope for $\Delta_{\mathrm{cap}}$)", "#a1d99b"),
        (sigma_der,  r"$\sigma$",                                "#fcae91"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    for i, (data, label, color) in enumerate(params_der):
        ax = axes[i]
        ax.hist(data, bins=40, density=True, alpha=0.6,
                color=color, edgecolor="none")
        sns.kdeplot(data, ax=ax, color="#696868", linewidth=1.2)
        mv = np.mean(data);  sv = np.std(data)
        ax.axvline(mv, color="red", linestyle="--", linewidth=2)
        ax.text(0.97, 0.95, f"Mean = {mv:.3f}\nStd = {sv:.3f}",
                transform=ax.transAxes, color="red",
                fontsize=FONT_SIZE, ha="right", va="top")
        ax.set_title(label, fontsize=FONT_SIZE)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "derate_posterior_density_combined.png",  dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_posterior_density_combined.pdf",  format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_posterior_density_combined.eps",  format="eps", bbox_inches="tight")
    plt.close()
    print("  Derate posterior density saved.")

    # ── Event duration vs capacity-loss plot ──────────────
    plot_sev = np.linspace(X_sev.min(), X_sev.max(), 100)
    mu_pl    = theta0_der[:, None] + theta1_der[:, None] * plot_sev[None, :]
    y_pl     = np.exp(mu_pl + (sigma_der[:, None] ** 2) / 2.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_sev, y_hrs_der, alpha=0.5,
               label="Observed", color="#4E79A7")
    ax.plot(plot_sev, y_pl.mean(axis=0),
            color="red", linewidth=2.5, label="Predicted Mean")
    ax.fill_between(plot_sev,
                    np.percentile(y_pl, 2.5,  axis=0),
                    np.percentile(y_pl, 97.5, axis=0),
                    color="red", alpha=0.2, label="95% Credible Interval")
    ax.set_xlabel(r"$\Delta_{\mathrm{cap}} = 1 - \mathrm{NAC}_{\mathrm{norm}}$",
                  fontsize=FONT_SIZE)
    ax.set_ylabel("Event Duration (hours)", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.legend(frameon=False, fontsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(S3_DIR / "derate_event_duration_vs_capacityloss_plot.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_event_duration_vs_capacityloss_plot.pdf",
                format="pdf", bbox_inches="tight")
    fig.savefig(S3_DIR / "derate_event_duration_vs_capacityloss_plot.eps",
                format="eps", bbox_inches="tight")
    plt.close()
    print("  Derate event duration vs capacity-loss plot saved.")

    derate_results_s3 = {"cv_predictions": cv_der,
                         "theta_0_all": theta0_der,
                         "theta_1_all": theta1_der,
                         "sigma_all":   sigma_der}

else:
    print(f"  Not enough derate rows ({len(data_derate_s3)}) for {_N_SPLITS}-fold CV. Skipping.")

print("\nStage 3 complete. All outputs in:", S3_DIR)