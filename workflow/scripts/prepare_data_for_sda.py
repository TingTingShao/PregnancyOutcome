import sys
import functions.utils as utils
import os
import pandas as pd
import numpy as np
utils=utils.Utils()
from functions.mcLernon import McLernon
from functions.mcLernon_adapters import (
    mclernon_pre_prob_predictor,
    mclernon_post_prob_predictor,
)
from functions.bestBinaryImputer import oracle_brier_binary_all_missing
mclernon=McLernon()
from survstack import SurvivalStacker
from missforest import MissForest
import copy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

sys.stderr=open(snakemake.log[0], "w", buffering=1)

data=utils.load_data(snakemake.input['inputfile'])
_, _, _, _, X_tr_pre_scaled, X_ts_pre_scaled = data['rsf']['pre']
X_tr, X_ts, _, _, X_tr_post_scaled, X_ts_post_scaled = data['rsf']['post']
y_tr_cycle, y_ts_cycle = data['y_cycle']
y_tr_transfer, y_ts_transfer = data['y_transfer']
y_tr_over1cycle, y_ts_over1cycle = data['y_over1cycle']
y_tr_over2cycle, y_ts_over2cycle = data['y_over2cycle']


# prepare data for survival stacking, model logit and rf
X_tr_pre_scaled['id']=X_tr_pre_scaled.index
X_ts_pre_scaled['id']=X_ts_pre_scaled.index
X_tr_post_scaled['id']=X_tr_post_scaled.index
X_ts_post_scaled['id']=X_ts_post_scaled.index


cov_cols_pre=list(X_tr_pre_scaled.columns)
cov_cols_post=list(X_tr_post_scaled.columns)


ss=SurvivalStacker(time_encoding="onehot")
# for original training dataset
X_tr_pre_cycle_stacked, y_tr_cycle_stacked = ss.fit_transform(X_tr_pre_scaled.to_numpy(), y_tr_cycle)
X_tr_pre_transfer_stacked, y_tr_transfer_stacked = ss.fit_transform(X_tr_pre_scaled.to_numpy(), y_tr_transfer)
X_tr_post_cycle_stacked, _ = ss.fit_transform(X_tr_post_scaled.to_numpy(), y_tr_cycle)
X_tr_post_transfer_stacked, _ = ss.fit_transform(X_tr_post_scaled.to_numpy(), y_tr_transfer)


max_cycle=y_tr_cycle['time'].max().astype(int)
max_transfer=y_tr_transfer['time'].max().astype(int)
n_transfers=X_tr_pre_transfer_stacked.shape[1]-len(cov_cols_pre)
cov_stacked_pre_cycle=cov_cols_pre+[f"risk_{t}" for t in range(1, max_cycle+1)]
cov_stacked_post_cycle=cov_cols_post+[f"risk_{t}" for t in range(1, max_cycle+1)]
cov_stacked_pre_transfer=cov_cols_pre+[f"risk_{t}" for t in range(1, n_transfers+1)]
cov_stacked_post_transfer=cov_cols_post+[f"risk_{t}" for t in range(1, n_transfers+1)]


X_tr_pre_transfer_stacked=pd.DataFrame(X_tr_pre_transfer_stacked, columns=cov_stacked_pre_transfer)
X_tr_post_transfer_stacked=pd.DataFrame(X_tr_post_transfer_stacked, columns=cov_stacked_post_transfer)
X_tr_pre_cycle_stacked=pd.DataFrame(X_tr_pre_cycle_stacked, columns=cov_stacked_pre_cycle)
X_tr_post_cycle_stacked=pd.DataFrame(X_tr_post_cycle_stacked, columns=cov_stacked_post_cycle)

X_te_pre_cycle_stacked=utils.build_stacked_test(X_ts_pre_scaled, cov_cols_pre, range(1, max_cycle+1))
X_te_post_cycle_stacked=utils.build_stacked_test(X_ts_post_scaled, cov_cols_post, range(1, max_cycle+1))
X_te_pre_transfer_stacked=utils.build_stacked_test(X_ts_pre_scaled, cov_cols_pre, range(1, max_transfer+1))
X_te_post_transfer_stacked=utils.build_stacked_test(X_ts_post_scaled, cov_cols_post, range(1, max_transfer+1))



data={
    "pre": {
        "cycle": (X_tr_pre_cycle_stacked, y_tr_cycle_stacked, X_te_pre_cycle_stacked),
        "transfer": (X_tr_pre_transfer_stacked, y_tr_transfer_stacked, X_te_pre_transfer_stacked),
    },
    "post": {
        "cycle": (X_tr_post_cycle_stacked, y_tr_cycle_stacked, X_te_post_cycle_stacked),
        "transfer": (X_tr_post_transfer_stacked, y_tr_transfer_stacked, X_te_post_transfer_stacked),
    },
    'N_test': X_ts_pre_scaled.shape[0]
}

utils.save_data(data, snakemake.output['sda_data'])



