import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.stats import linregress, t

def _km_point_with_ci(times, events, t):
    """
    KM survival at time t with 95% CI (if available), then convert to risk.
    Returns (risk, risk_lo, risk_hi); CI may be NaN if lifelines can't compute.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(times, events)

    # point estimate S(t)
    S_hat = float(kmf.predict(t))
    # try to get pointwise CI at time t
    S_lo = S_hi = np.nan
    try:
        ci_df = kmf.confidence_interval_at_times([t])
        # robust to column names: take min/max across columns
        S_lo = float(ci_df.min(axis=1).iloc[0])
        S_hi = float(ci_df.max(axis=1).iloc[0])
    except Exception:
        pass

    R_hat = 1.0 - S_hat
    R_lo  = np.nan if np.isnan(S_lo) else max(0.0, 1.0 - S_lo)
    R_hi  = np.nan if np.isnan(S_hi) else min(1.0, 1.0 - S_hi)
    return R_hat, R_lo, R_hi


def plot_binned_calibration(
    y_te,
    pred,
    time,
    *,
    idx=None,            # 1-based column if pred is SURVIVAL (N,K)
    ax=None,
    n_groups=10,         # fixed at 10 by default (deciles)
    add_density=True,
    label="Model",
    return_table=False,
    strategy="uniform",
):
    """
    Decile-binned calibration plot:
    - sort by predicted probability,
    - split into 10 equal-sized groups,
    - per decile plot mean(pred) vs KM-observed risk at `time` with 95% CI.

    Parameters
    ----------
    y_te : structured array / dict-like with 'time' and 'event'
    pred : (N,) risk at `time`  OR  (N,K) SURVIVAL (use idx to select column)
    time : float, horizon for observed risk
    idx : int, 1-based column index if pred is SURVIVAL (N,K)
    ax : matplotlib Axes or None
    n_groups : int, default 10 (deciles)
    add_density : bool, show a small density band of predictions along bottom
    label : str, legend label
    return_table : bool, if True also return a DataFrame with bin stats
    strategy: 
        - uniform: split [0, 1] into equal-width bins (default)
        - quantile: split into bins with equal number of samples 

    Returns
    -------
    ax  (and optionally df with columns:
         ['bin','n','pred_mean','obs','obs_lo','obs_hi'])
    """
    times_arr = np.asarray(y_te['time'])
    events_arr = np.asarray(y_te['event']).astype(int)

    P = np.asarray(pred)
    if P.ndim == 2:
        if idx is None:
            raise ValueError("pred is (N,K) SURVIVAL; provide 1-based column index via `idx`.")
        if not (1 <= idx <= P.shape[1]):
            raise IndexError(f"`idx` must be in [1, {P.shape[1]}], got {idx}.")
        risk = 1.0 - P[:, idx - 1]     # convert survival->risk at that horizon
    elif P.ndim == 1:
        risk = P                       # already risk
    else:
        raise ValueError("`pred` must be (N,) risk or (N,K) survival.")

    # values smaller than 0 become 0, and values larger than 1 become 1.
    risk = np.clip(risk, 0.0, 1.0)
    N = len(risk)
    # slice indices for each bin
    idx_slices = []
    strategy = strategy.lower()
    if strategy == "quantile":
        # ---- Rank and split into 10 near-equal groups (deciles) ----
        # return indices that would sort the array
        order = np.argsort(risk, kind="mergesort")   # stable sort

        # sizes of each bin (shape of the returned array): n_groups, filled value: N//n_groups the number of patients in each group )
        sizes = np.full(n_groups, N // n_groups, dtype=int)

        # distribute the remainder (N % n_groups) over the first bins
        sizes[: (N % n_groups)] += 1      


        start = 0
        for s in sizes:
            stop = start + s
            idx_slices.append(order[start:stop])
            start = stop
    elif strategy == "uniform":
        # ---- Split into uniform bins ----
        bin_edges = np.linspace(0, 1, n_groups + 1)
        idx_slices = [np.where((risk >= bin_edges[i]) & (risk < bin_edges[i + 1]))[0] for i in range(n_groups)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # ---- Per-bin summaries ----
    rows = []
    for i, idxs in enumerate(idx_slices, start=1):
        if len(idxs) == 0:
            continue
        # find the positions of risk, time and event based on the idxs
        r_bin = risk[idxs]
        t_bin = times_arr[idxs]
        e_bin = events_arr[idxs]

        pred_mean = float(np.mean(r_bin))
        obs, obs_lo, obs_hi = _km_point_with_ci(t_bin, e_bin, time)

        rows.append({
            "bin": i,
            "n": int(len(idxs)),
            "pred_mean": pred_mean,
            "obs": obs,
            "obs_lo": obs_lo,
            "obs_hi": obs_hi,
        })

    df = pd.DataFrame(rows)

    # ---- Plot ----
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        created = True

    yerr = None
    # print(df["obs_lo"].isna().sum(), df["obs_hi"].isna().sum())
    if df["obs_lo"].notna().all() and df["obs_hi"].notna().all():
        yerr = np.vstack([df["obs"] - df["obs_lo"], df["obs_hi"] - df["obs"]])

    # ax.errorbar(df["pred_mean"], df["obs"], yerr=yerr, fmt="o", capsize=2, elinewidth=1, label=label)
    ax.errorbar(
        df["pred_mean"], df["obs"], yerr=yerr,
        fmt='o', capsize=3, linewidth=1
    )

    # reference line
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2)
    
    if len(df) >=2:
        x=df['pred_mean'].to_numpy()
        y=df['obs'].to_numpy()
        # m, b = np.polyfit(x, y, 1)
        # xs_line=np.array([0, 1])
        # ys_line=m*xs_line + b
        # ax.plot(xs_line, ys_line, linewidth=2, label=f"slope={m:.2f}\nintercept={b:.2f}") 
        lr = linregress(x, y)               # slope, intercept, stderr, intercept_stderr, ...
        n  = len(x)
        tcrit = t.ppf(0.975, df=n-2)        # 95% two-sided

        m = lr.slope
        b = lr.intercept
        slope_ci = (m - tcrit*lr.stderr, m + tcrit*lr.stderr)
        inter_ci = (b - tcrit*lr.intercept_stderr, b + tcrit*lr.intercept_stderr)

        xs = np.array([0.0, 1.0])
        ax.plot(xs, m*xs + b, linewidth=2,
                label=f"slope={m:.2f} [{slope_ci[0]:.2f},{slope_ci[1]:.2f}]\n"
                    f"intercept={b:.2f} [{inter_ci[0]:.2f},{inter_ci[1]:.2f}]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xlabel("Predicted probability of live birth", fontsize=20)
    # ax.set_ylabel("Observed proportion with live birth", fontsize=20)
    ax.set_title(f"Cycle {time}", fontsize=20)
    ax.legend(loc="upper left", fontsize=20)

    # Optional: a small density band (like the paper’s baseline curve)
    if add_density:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(0, 1, 200)
            kde = gaussian_kde(risk)
            dens = kde(xs)
            dens = dens / dens.max() * 0.07
            ax.fill_between(xs, 0, dens, alpha=0.25, step=None)
        except Exception:
            pass

    if created:
        fig.tight_layout()

    return (ax, df) if return_table else ax
