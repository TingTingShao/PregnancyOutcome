import itertools
import numpy as np
import pandas as pd

def _enumerate_bits(k: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.int8)

def oracle_brier_binary_all_missing(
    X: pd.DataFrame,
    y_true: np.ndarray,               # binary labels 0/1 (success by horizon); NaN if unknown
    missing_cols: list[str],
    prob_predictor,                   # callable: df -> p_success (n,)
    copy: bool = True,
    on_nan_label: str = "skip"        # "skip" | "first" (use all-zeros)
):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame")

    Xw = X.copy() if copy else X
    y_true = np.asarray(y_true, dtype=float)
    n = len(Xw)

    k = len(missing_cols)
    combos = _enumerate_bits(k)        # (C, k) C: 32 for k=5, C: 8 for k=3
    C = combos.shape[0]

    # Precompute probs for each combo (C model calls total, not per row)
    P_all = np.empty((C, n), dtype=float)
    for c, bits in enumerate(combos):
        Xc = Xw.copy()
        for j, col in enumerate(missing_cols):
            Xc[col] = int(bits[j])
        P_all[c, :] = prob_predictor(Xc)  # (n,)

    col_idx = [Xw.columns.get_loc(c) for c in missing_cols]
    chosen_bits, chosen_prob, chosen_brier = [], [], []

    for i in range(n):
        yi = y_true[i]
        if not np.isfinite(yi):
            # np.isfinite: not finitely or not a number (NaN)
            # no label available at horizon
            if on_nan_label in ("first", "zeros"):
                jb = 0
                for j, idxj in enumerate(col_idx):
                    Xw.iat[i, idxj] = int(combos[jb, j])
                chosen_bits.append(tuple(int(b) for b in combos[jb]))
                chosen_prob.append(float(P_all[jb, i]))
                chosen_brier.append(np.nan)
            else:
                chosen_bits.append(tuple(int(v) for v in Xw.iloc[i][missing_cols].astype(int).tolist()))
                chosen_prob.append(np.nan)
                chosen_brier.append(np.nan)
            continue

        losses = (P_all[:, i] - yi) ** 2     # (C,)
        jb = int(np.argmin(losses))
        # print(f"Row {i}: true={yi}, best combo index={jb}, bits={combos[jb]}, prob={P_all[jb, i]}, brier={losses[jb]}")
        for j, idxj in enumerate(col_idx):
            # print(idxj)
            # print(combos[jb, j])
            # print(j)
            # print(combos)
            Xw.iat[i, idxj] = int(combos[jb, j])
            
            # print(Xw.iat[i, idxj])
        

        chosen_bits.append(tuple(int(b) for b in combos[jb]))
        chosen_prob.append(float(P_all[jb, i]))
        chosen_brier.append(float(losses[jb]))

    choices = pd.DataFrame(
        {"chosen_bits": chosen_bits, "chosen_prob": chosen_prob, "chosen_brier": chosen_brier},
        index=X.index
    )
    return Xw, choices

import itertools
import numpy as np
import pandas as pd

def _enumerate_bits(k: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.int8)

def oracle_brier_binary_all_missing_worse(
    X: pd.DataFrame,
    y_true: np.ndarray,               # binary labels 0/1 (success by horizon); NaN if unknown
    missing_cols: list[str],
    prob_predictor,                   # callable: df -> p_success (n,)
    copy: bool = True,
    on_nan_label: str = "skip"        # "skip" | "first" (use all-zeros)
):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame")

    Xw = X.copy() if copy else X
    y_true = np.asarray(y_true, dtype=float)
    n = len(Xw)

    k = len(missing_cols)
    combos = _enumerate_bits(k)        # (C, k) C: 32 for k=5, C: 8 for k=3
    C = combos.shape[0]

    # Precompute probs for each combo (C model calls total, not per row)
    P_all = np.empty((C, n), dtype=float)
    for c, bits in enumerate(combos):
        Xc = Xw.copy()
        for j, col in enumerate(missing_cols):
            Xc[col] = int(bits[j])
        P_all[c, :] = prob_predictor(Xc)  # (n,)

    col_idx = [Xw.columns.get_loc(c) for c in missing_cols]
    chosen_bits, chosen_prob, chosen_brier = [], [], []

    for i in range(n):
        yi = y_true[i]
        if not np.isfinite(yi):
            # np.isfinite: not finitely or not a number (NaN)
            # no label available at horizon
            if on_nan_label in ("first", "zeros"):
                jb = 0
                for j, idxj in enumerate(col_idx):
                    Xw.iat[i, idxj] = int(combos[jb, j])
                chosen_bits.append(tuple(int(b) for b in combos[jb]))
                chosen_prob.append(float(P_all[jb, i]))
                chosen_brier.append(np.nan)
            else:
                chosen_bits.append(tuple(int(v) for v in Xw.iloc[i][missing_cols].astype(int).tolist()))
                chosen_prob.append(np.nan)
                chosen_brier.append(np.nan)
            continue

        losses = (P_all[:, i] - yi) ** 2     # (C,)
        jb = int(np.argmax(losses))
        # print(f"Row {i}: true={yi}, best combo index={jb}, bits={combos[jb]}, prob={P_all[jb, i]}, brier={losses[jb]}")
        for j, idxj in enumerate(col_idx):
            # print(idxj)
            # print(combos[jb, j])
            # print(j)
            # print(combos)
            Xw.iat[i, idxj] = int(combos[jb, j])
            
            # print(Xw.iat[i, idxj])
        

        chosen_bits.append(tuple(int(b) for b in combos[jb]))
        chosen_prob.append(float(P_all[jb, i]))
        chosen_brier.append(float(losses[jb]))

    choices = pd.DataFrame(
        {"chosen_bits": chosen_bits, "chosen_prob": chosen_prob, "chosen_brier": chosen_brier},
        index=X.index
    )
    return Xw, choices
