from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde
import pandas as pd
def _R0_at_times(y_te, times):
    """Get the survival probability at specific times from a fitted KaplanMeierFitter.

    Args:
       y_te: survival format array with 'time' and 'event' columns
         times: array-like of times at which to get the survival
    Returns:
       array of survival probabilities at the specified times
    """
    kmf = KaplanMeierFitter()
    kmf.fit(y_te['time'], y_te['event'])
    return 1- kmf.predict(times).to_numpy(dtype=float)

def plot_calibration(y_te, S_pred, times, path=None, ax=None):
    """Plot calibration curve for a fitted KaplanMeierFitter.

    Args:
       y_te: survival format array with 'time' and 'event' columns
       S_pred: predicted survival probabilities (N * K) dictionary of arrays
       times: array-like of times at which to get the survival probabilities

    Returns:
       matplotlib axis object
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    # the number of models
    keys=list(S_pred.keys())
    len_models=len(keys)

    # Get the true survival probabilities at the specified times
    R0_true = _R0_at_times(y_te, times)

    # 3) CI ingredients (same for all horizons under your formula)
    alpha = 0.05
    z = norm.ppf(1 - alpha/2)
    O = int(np.sum(y_te['event']))
    se_logOE = np.sqrt(1.0 / max(O, 1))  # guard O=0

    # 4) Compute O/E and CI per model (arrays over horizons)
    oe = {}
    ci_lo = {}
    ci_hi = {} 
    for m, preds in S_pred.items():
        print(times[0]-1, times[-1])
        # if not numpy array, convert to numpy array
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        S_pred_m = preds[:, (times[0]-1):(times[-1])]  # (N, K)
        S_pred_mean = S_pred_m.mean(axis=0)  # (K,)
        R_pred_mean = 1 - S_pred_mean  # (K,)
        OE_t = R0_true/R_pred_mean   # (K,)
        oe[m] = OE_t
        ci_lo[m] = OE_t * np.exp(-z * se_logOE)
        ci_hi[m] = OE_t * np.exp(z * se_logOE)
    
    # 5) Plot calibration curves
    H = R0_true.shape[0]
    x=np.arange(1, H+1)

    
    # Small x-offsets so different models at same horizon don't sit on top of each other
    offsets = np.linspace(-0.15, 0.15, num=len_models)

    for off, m in zip(offsets, keys):
        y = oe[m]                          # shape (H,)
        lo = y - ci_lo[m]                  # lower error
        hi = ci_hi[m] - y                  # upper error
        yerr = np.vstack([lo, hi])         # asymmetric errors (2 x H)

        ax.errorbar(
            x + off, y, fmt='o', yerr=yerr, label=m)
            # fmt='s-', capsize=2, linewidth=1, markersize=5, label=m)

    ax.axhline(1.0, linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x])     # or ["Cycle 1", ...]
    ax.set_xlabel("Cycle", fontsize=16)
    ax.set_ylabel("Observed / Expected (O/E)", fontsize=16)
    # ax.set_title("Calibration (O/E) across cycles with 95% CI", fontsize=16)
    ax.legend(ncol=2)
    # fig.tight_layout()
    if path and created_fig:
        fig.savefig(path, dpi=800)
    return ax


