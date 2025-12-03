import sys
import functions.utils as utils
import os
import pandas as pd
import numpy as np
from survstack import SurvivalStacker
utils=utils.Utils()

sys.stderr=open(snakemake.log[0], "w", buffering=1)

data=utils.load_data(snakemake.input['mclernon_data_had'])
data_refit=utils.load_data(snakemake.input['mclernon_data_for_refit'])
# data_all=utils.load_data("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl")

# fit bart with the same data to refit the mclernon
X_transformed_pre, X_te_transformed_pre, y_tr_cycle, y_ts_cycle = data_refit['unstacked']['pre']
X_transformed_post, X_te_transformed_post, y_tr_cycle, y_ts_cycle = data_refit['unstacked']['post']
# X_ext_transformed_pre, X_ext_transformed_post, y_ext = data_refit['unstacked']['ext']

# fit bart with the same features used to refit the mclernon, but not transform/engineer the features 
X_tr_pre, X_ts_pre =data['pre_had']
X_tr_post, X_ts_post=data['post_had']
# X_ext_pre, X_ext_post, y_ext=data['ext']
# print(X_ext_pre.columns)
# print(X_ext_post.columns)
# # y_tr_cycle, y_ts_cycle = data_all['y_cycle']

# X_ext_pre['OvulDisorder'] = (
#     X_ext_pre['OvulDisorder']
#       .apply(pd.to_numeric, errors='coerce')   # handles "1", "0", 1.0, etc.
#       .astype('Int64')                         # preserves NaN as <NA>
# )
# X_ext_post['OvulDisorder'] = (
#     X_ext_post['OvulDisorder']
#       .apply(pd.to_numeric, errors='coerce')   # handles "1", "0", 1.0, etc.
#       .astype('Int64')                         # preserves NaN as <NA>
# )
# construct the dataset for bart 
# ----- one with transformed features ----
def concate_together(X, y):
    # position same, safe to concat in this way by adapting df index to concat
    y_df = pd.DataFrame({"event": y["event"].astype(int),
                     "time":  y["time"].astype(float)}).set_index(X.index)
    # sanity checks
    assert len(X) == len(y_df)
    assert (X.index == y_df.index).all(), "Index mismatch: reorder or reindex first."

    df = pd.concat([X, y_df], axis=1)  # preserves original index
    return df

data_tr_pre_transformed = concate_together(X_transformed_pre, y_tr_cycle)
data_te_pre_transformed=concate_together(X_te_transformed_pre, y_ts_cycle)
data_tr_post_transformed=concate_together(X_transformed_post, y_tr_cycle)
data_te_post_transformed=concate_together(X_te_transformed_post, y_ts_cycle)
# data_ext_pre_transformed=concate_together(X_ext_transformed_pre, y_ext)
# data_ext_post_transformed=concate_together(X_ext_transformed_post, y_ext)

data_tr_pre=concate_together(X_tr_pre, y_tr_cycle)
data_te_pre=concate_together(X_ts_pre, y_ts_cycle)
data_tr_post=concate_together(X_tr_post, y_tr_cycle)
data_te_post=concate_together(X_ts_post, y_ts_cycle)
# ext_pre=concate_together(X_ext_pre, y_ext)
# ext_post=concate_together(X_ext_post, y_ext)

utils.save_data(data_tr_pre_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_tr_pre_transformed_ml_refit.csv")
utils.save_data(data_te_pre_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_te_pre_transformed_ml_refit.csv")
utils.save_data(data_tr_post_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_tr_post_transformed_ml_refit.csv")
utils.save_data(data_te_post_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_te_post_transformed_ml_refit.csv")
# utils.save_data(data_ext_pre_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/data_ext_pre_transformed_mclernon.csv")
# utils.save_data(data_ext_post_transformed, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/data_ext_post_transformed_mclernon.csv")

utils.save_data(data_tr_pre, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_tr_pre_feats_ml_refit.csv")
utils.save_data(data_te_pre, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_te_pre_feats_ml_refit.csv")
utils.save_data(data_tr_post, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_tr_post_feats_ml_refit.csv")
utils.save_data(data_te_post, snakemake.output['bart_te_post'])

# utils.save_data(ext_pre, "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/ext_pre_mclernon.csv")
# utils.save_data(ext_post, snakemake.output['ext_post'])