import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid  # <- to support dict grids
from sksurv.metrics import integrated_brier_score

class SearchRSF:
    @staticmethod
    def _evaluate_model(clf, X_tr, y_tr, X_val, y_val):
        clf.fit(X_tr, y_tr)

        # ensure test times do not exceed train max (required by IBS/IPCW)
        train_max = np.max(y_tr["time"])
        if np.max(y_val["time"]) > train_max:
            keep = y_val["time"] <= train_max
            y_val = y_val[keep]
            X_val = X_val[keep]

        # pick evaluation grid (exclude 0; ensure at least one time)
        max_time = int(np.max(y_val["time"]))
        times = np.arange(1, max_time)
        if times.size == 0:
            # fallback: use unique event times from training set
            evt_times = y_tr["time"][y_tr["event"].astype(bool)]
            times = np.unique(evt_times.astype(float))

        # get survival step functions and evaluate on 'times'
        surv_funcs = clf.predict_survival_function(X_val)  # list of StepFunction
        preds = np.asarray([fn(times) for fn in surv_funcs])  # (n_val, len(times))

        return integrated_brier_score(y_tr, y_val, preds, times)

    @staticmethod
    def _make_strat_labels(y_tr, cv):
        # y_tr is a sksurv structured array: fields 'event' (bool), 'time' (float)
        evt = y_tr["event"].astype(int)
        # time = y_tr["time"]
        # labels = evt.astype(str) + "_" + time.astype(str)
        # counts = pd.Series(labels).value_counts()
        # return labels if counts.min() >= cv else evt
        return evt

    @staticmethod
    def select_hyperparameters(X_tr, y_tr, model, param_grid, cv=5, random_state=0):
        strat = SearchRSF._make_strat_labels(y_tr, cv)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        # support dict-of-lists or list-of-dicts
        grid_iter = ParameterGrid(param_grid) if isinstance(param_grid, dict) else param_grid

        records = []
        best_score = np.inf
        best_est = None
        best_params = None

        for params in grid_iter:
            fold_scores = []
            for tr_idx, te_idx in splitter.split(X_tr, strat):
                X_tr_cv, y_tr_cv = X_tr.iloc[tr_idx], y_tr[tr_idx]
                X_val_cv, y_val_cv = X_tr.iloc[te_idx], y_tr[te_idx]

                clf = model.set_params(**params, random_state=random_state)
                score = SearchRSF._evaluate_model(clf, X_tr_cv, y_tr_cv, X_val_cv, y_val_cv)
                print(score)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            records.append({**params, "score": mean_score})

            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_est = model.set_params(**params, random_state=random_state).fit(X_tr, y_tr)

        return best_est, best_params  # (optionally also return 'records')
