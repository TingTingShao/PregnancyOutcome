import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score

def _km_surv_at_times(y_train, times):
    """
    Estimate the Kaplan-Meier survival function and return survival probabilities at specified times.

    Parameters
    ----------
    y : structured array
        Array with fields 'event' (boolean) and 'time' (float).
    times : array-like
        Times at which to evaluate the survival function.

    Returns
    -------
    surv_probs : array
        Survival probabilities at the specified times.
    """
    kmf=KaplanMeierFitter().fit(y_train['time'], y_train['event'])
    # S0=np.array(kmf.predict(t) for t in times)
    return kmf.predict(times).to_numpy(dtype=float)

def _bootstrao_idx(n, rng):
    """Generate bootstrap indices for a sample of size n."""
    return rng.integers(0, n, n)

def _pct_ci(a, alpha=0.05):
    lo=np.nanpercentile(a, 100*alpha/2)
    hi=np.nanpercentile(a, 100*(1-alpha/2))
    return lo, hi

def _safe_time_grid(y_tr, X_te, y_te):
    """
    Create a safe time grid for training and test data.

    Args:
        y_tr: array-like, training labels
        y_te: array-like, test labels

    Returns:
        time_grid: array-like, safe time grid
    """
    train_max= np.max(y_tr['time'])
    # 首先保证 test data的时间点不超过train的最大时间点
    keep=y_te['time']<=train_max
    y_te=y_te[keep]
    X_te=X_te[keep]
    # 然后保证时间range 不包括最大test时间点 [1, max)
    time_max=y_te['time'].max()
    time_grid=np.arange(1, time_max)
    return X_te, y_te, time_grid

class SurvivalMetrics:
    def auc_surv_time_dependent(self, model, y_tr, X_te, y_te, name, n_boot=500, alpha=0.05, random_state=0, estimates=None, time_grid=None):
        """
        Compute time-dependent AUC with confidence intervals using bootstrapping.

        Parameters
        ----------
        model : fitted survival model
            The survival model to evaluate.
        y_tr : structured array
            Training labels with fields 'event' (boolean) and 'time' (float).
        X_te : array-like
            Test features.
        y_te : structured array
            Test labels with fields 'event' (boolean) and 'time' (float).
        name : str
            Name of the model.
        n_boot : int, optional
            Number of bootstrap samples for confidence intervals. Default is 500.
        alpha : float, optional
            Significance level for confidence intervals. Default is 0.05.
        random_state : int, optional
            Random seed for reproducibility. Default is 0.
        estimates: risk estimates for test data, shape (n_samples, n_times) or (n_samples,), higher the more likely to experience the event

        Returns
        -------
        df_auc : DataFrame
            DataFrame containing time points, AUC values, and confidence intervals.
        """
        if time_grid is None:
            X_te, y_te_fixed, time_grid = _safe_time_grid(y_tr, X_te, y_te)
        else:
            time_grid = np.asarray(time_grid)
            X_te, y_te_fixed = X_te, y_te

        # Estimate risks for test data， higher the risk, more likely to experience the event
        if estimates is None:
            estimates=model.predict(X_te)
        else:
            estimates = np.asarray(estimates)
            if estimates.ndim == 2:
                K = estimates.shape[1]
                if K != len(time_grid):
                    # common case: estimates made on a longer grid -> slice left block
                    if K > len(time_grid):
                        estimates = estimates[:, :len(time_grid)]
                    else:
                        # 如果estimates的列数比time_grid还少，说明training的time_grid比test的还短，报错
                        raise ValueError(f"estimates has {K} columns but time_grid has {len(time_grid)}.")
            # If 1-D, OK: same score used for all times


        # Compute AUC for the original sample
        auc, mean_auc = cumulative_dynamic_auc(y_tr, y_te_fixed, estimates, time_grid)

        # Bootstrap to get confidence intervals
        rng=np.random.default_rng(random_state)

        boot_aucs=np.full((n_boot, len(time_grid)), np.nan)
        boot_mean_aucs=np.full(n_boot, np.nan)   

        # the number of testing samples
        n=len(y_te_fixed['event'])

        for i in range(n_boot):
            idx=_bootstrao_idx(n, rng)
            # risk_b=estimates[idx]
            # y_te_b=y_te_fixed[idx]
            # max_fu = float(np.max(y_te_b['time']))
            # grid_b = time_grid[time_grid < max_fu]  
            y_te_b = y_te_fixed[idx]

            # trim time grid for this resample
            max_fu = float(np.max(y_te_b["time"]))
            mask_t = time_grid < max_fu
            grid_b = time_grid[mask_t]
            if grid_b.size < 2:
                continue

            # slice the estimates to the same time mask if 2D
            if estimates.ndim == 2:
                risk_b = estimates[idx][:, mask_t]     # (n_boot, grid_b.size)

            else:
                risk_b = estimates[idx]                # (n_boot,), same score for all times
            boot_auc, boot_mean_auc=cumulative_dynamic_auc(y_tr, y_te_b, risk_b, grid_b)
            boot_aucs[i, :grid_b.size] = boot_auc
            boot_mean_aucs[i] = boot_mean_auc



        ci_lo=np.array([_pct_ci(boot_aucs[:, j], alpha)[0] for j in range(len(time_grid))])
        ci_hi=np.array([_pct_ci(boot_aucs[:, j], alpha)[1] for j in range(len(time_grid))])
        mean_ci_lo, mean_ci_hi=_pct_ci(boot_mean_aucs, alpha)



        df_auc=pd.DataFrame({'model': name,
                             'time': time_grid, 
                             'auc': auc, 
                             'mean_auc': mean_auc, 
                             'auc_ci_lo': ci_lo, 
                             'auc_ci_hi': ci_hi, 
                             'mean_auc_ci_lo': mean_ci_lo, 
                             'mean_auc_ci_hi': mean_ci_hi,})

        return df_auc, boot_mean_aucs

    def brier_surv_time_dependent(self, model, y_tr, X_te, y_te, name, n_boot=500, alpha=0.05, random_state=0, estimates=None, time_grid=None):
        """
        Compute time-dependent Brier score with confidence intervals using bootstrapping.

        Parameters
        ----------
        model : fitted survival model
            The survival model to evaluate.
        y_tr : structured array
            Training labels with fields 'event' (boolean) and 'time' (float).
        X_te : array-like
            Test features.
        y_te : structured array
            Test labels with fields 'event' (boolean) and 'time' (float).
        name : str
            Name of the model.
        n_boot : int, optional
            Number of bootstrap samples for confidence intervals. Default is 500.
        alpha : float, optional
            Significance level for confidence intervals. Default is 0.05.
        random_state : int, optional
            Random seed for reproducibility. Default is 0.
        estimates : array-like, optional, survival probability estimates for test data, shape (n_samples, n_times) or (n_samples,), higher the more likely to survive

        Returns
        -------
        df_brier : DataFrame
            DataFrame containing time points, Brier scores, and confidence intervals.
        """
        if time_grid is None:
            X_te, y_te_fixed, time_grid = _safe_time_grid(y_tr, X_te, y_te)
        else:
            time_grid = np.asarray(time_grid)
            X_te, y_te_fixed = X_te, y_te

        # Estimate risks for test data
        if estimates is None: 
            estimates=model.predict_survival_function(X_te)
            preds=np.asarray([[fn(t) for t in time_grid] for fn in estimates])
        else:
            estimates = np.asarray(estimates)
            if estimates.ndim == 2:
                K = estimates.shape[1]
                if K != len(time_grid):
                    # common case: estimates made on a longer grid -> slice left block
                    if K > len(time_grid):
                        estimates = estimates[:, :len(time_grid)]
                    else:
                        raise ValueError(f"estimates has {K} columns but time_grid has {len(time_grid)}.")
            # If 1-D, OK: same score used for all times
            preds=estimates

        

        # Calculate Brier score for the original sample
        _, brier_scores = brier_score(y_tr, y_te_fixed, preds, time_grid)
        ibs = integrated_brier_score(y_tr, y_te_fixed, preds, time_grid)


        # Calculate Brier score for the null model  
        S0=_km_surv_at_times(y_tr, time_grid)

        # repeat S0 for all test samples
        surv_null_full=np.tile(S0, (len(y_te_fixed), 1))
        _, brier_scores_null = brier_score(y_tr, y_te_fixed, surv_null_full, time_grid)
        ibs_null = integrated_brier_score(y_tr, y_te_fixed, surv_null_full, time_grid)
        print(ibs_null)
        print(ibs)
        # calculate scaled brier score 
        scaled_brier=1 - brier_scores / brier_scores_null
        scaled_ibs=1 - ibs / ibs_null
        
        # Bootstrap to get confidence intervals
        rng=np.random.default_rng(random_state)
        boot_briers       = np.full((n_boot, len(time_grid)), np.nan)
        boot_briers_null  = np.full((n_boot, len(time_grid)), np.nan)
        boot_ibs          = np.full(n_boot, np.nan)
        boot_ibs_null     = np.full(n_boot, np.nan)
        boot_scaled_brier = np.full((n_boot, len(time_grid)), np.nan)
        boot_scaled_ibs   = np.full(n_boot, np.nan)

        # null model brier score

        n=len(y_te_fixed['event'])

        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)     # sample with replacement
            y_b = y_te_fixed[idx]
            preds_b_full = preds[idx]            # (n, n_times)
            surv_null_b_full = surv_null_full[idx]

            # IMPORTANT: time grid must be < max follow-up of THIS resample
            max_fu = float(np.max(y_b['time']))
            grid_b = time_grid[time_grid < max_fu]
            k = grid_b.size
            if k < 2:
                continue  # skip too-short resamples safely

            preds_b = preds_b_full[:, :k]
            null_b  = surv_null_b_full[:, :k]

            _, bs_b = brier_score(y_tr, y_b, preds_b, grid_b)
            ibs_b = integrated_brier_score(y_tr, y_b, preds_b, grid_b)

            _, bs0_b = brier_score(y_tr, y_b, null_b, grid_b)
            ibs0_b = integrated_brier_score(y_tr, y_b, null_b, grid_b)

            boot_briers[i, :k]      = bs_b
            boot_briers_null[i, :k] = bs0_b
            boot_ibs[i]             = ibs_b
            boot_ibs_null[i]        = ibs0_b

            # scaled per-resample
            boot_scaled_brier[i, :k] = 1.0 - bs_b / bs0_b
            boot_scaled_ibs[i]       = 1.0 - ibs_b / ibs0_b


        briers_lo=np.array([_pct_ci(boot_briers[:, j], alpha)[0] for j in range(len(time_grid))])
        briers_hi=np.array([_pct_ci(boot_briers[:, j], alpha)[1] for j in range(len(time_grid))])

        briers_null_lo=np.array([_pct_ci(boot_briers_null[:, j], alpha)[0] for j in range(len(time_grid))])
        briers_null_hi=np.array([_pct_ci(boot_briers_null[:, j], alpha)[1] for j in range(len(time_grid))])

        ibs_lo=_pct_ci(boot_ibs, alpha)[0]
        ibs_hi=_pct_ci(boot_ibs, alpha)[1]
        ibs_null_lo=_pct_ci(boot_ibs_null, alpha)[0]
        ibs_null_hi=_pct_ci(boot_ibs_null, alpha)[1]

        # scaled brier score ci
        scaled_briers_lo=np.array([_pct_ci(boot_scaled_brier[:, j], alpha)[0] for j in range(len(time_grid))])
        scaled_briers_hi=np.array([_pct_ci(boot_scaled_brier[:, j], alpha)[1] for j in range(len(time_grid))])

        scaled_ibs_lo=_pct_ci(boot_scaled_ibs, alpha)[0]
        scaled_ibs_hi=_pct_ci(boot_scaled_ibs, alpha)[1]

        df_brier=pd.DataFrame({'model': name,
                              'time': time_grid,
                              'brier_score': brier_scores,
                              'brier_score_ci_lo': briers_lo,
                              'brier_score_ci_hi': briers_hi,
                              'ibs': ibs,
                              'ibs_ci_lo': ibs_lo,
                              'ibs_ci_hi': ibs_hi,
                              'scaled_brier_score': scaled_brier,
                              'scaled_brier_score_ci_lo': scaled_briers_lo,
                              'scaled_brier_score_ci_hi': scaled_briers_hi,
                              'scaled_ibs': scaled_ibs,
                              'scaled_ibs_ci_lo': scaled_ibs_lo,
                              'scaled_ibs_ci_hi': scaled_ibs_hi})

        return df_brier, boot_scaled_ibs






