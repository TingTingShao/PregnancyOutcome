import numpy as np

def mclernon_pre_prob_predictor(model, colmap=None):
    """Return P(success by 1 cycle) for each row."""
    def _predict(df):
        out = model.predict_pre_df(df, colmap=colmap, add_columns=False)
        return out["CumPCycle1"].to_numpy(dtype=float)
    return _predict

def mclernon_post_prob_predictor(model, colmap=None):
    """Return P(success by 2 cycles) for each row."""
    def _predict(df):
        out = model.predict_post_df(df, colmap=colmap, add_columns=False)
        return out["CumPCycle2"].to_numpy(dtype=float)
    return _predict

