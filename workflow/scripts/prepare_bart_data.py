import sys
import functions.utils as utils
import os
import pandas as pd
import numpy as np
utils=utils.Utils()
sys.stderr=open(snakemake.log[0], "w", buffering=1)
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# read data

data=utils.load_data(snakemake.input['inputfile'])
print("Data shape:", data.keys())

_, _, _, _, X_train_scaled_pre, X_test_scaled_pre = data['rsf']['pre']
_, _, _, _, X_train_scaled_post, X_test_scaled_post = data['rsf']['post']
y_train_cycle, y_test_cycle = data['y_cycle']
y_train_transfer, y_test_transfer = data['y_transfer']


def concate_together(X, y):
    y_df = pd.DataFrame({"event": y["event"].astype(int),
                     "time":  y["time"].astype(int)}).set_index(X.index)
    # sanity checks
    assert len(X) == len(y_df)
    assert (X.index == y_df.index).all(), "Index mismatch: reorder or reindex first."

    df = pd.concat([X, y_df], axis=1)  # preserves original index
    return df

X_train_cycle_pre = concate_together(X_train_scaled_pre, y_train_cycle)
X_test_cycle_pre = concate_together(X_test_scaled_pre, y_test_cycle)
X_train_transfer_pre = concate_together(X_train_scaled_pre, y_train_transfer)
X_test_transfer_pre = concate_together(X_test_scaled_pre, y_test_transfer)

X_train_cycle_post = concate_together(X_train_scaled_post, y_train_cycle)
X_test_cycle_post = concate_together(X_test_scaled_post, y_test_cycle)
X_train_transfer_post = concate_together(X_train_scaled_post, y_train_transfer)
X_test_transfer_post = concate_together(X_test_scaled_post, y_test_transfer)

utils.save_data(X_train_cycle_pre,snakemake.output['bart_train_cycle_pre'])
utils.save_data(X_test_cycle_pre,snakemake.output['bart_test_cycle_pre'])
utils.save_data(X_train_transfer_pre,snakemake.output['bart_train_transfer_pre'])
utils.save_data(X_test_transfer_pre,snakemake.output['bart_test_transfer_pre'])
utils.save_data(X_train_cycle_post,snakemake.output['bart_train_cycle_post'])
utils.save_data(X_test_cycle_post,snakemake.output['bart_test_cycle_post'])
utils.save_data(X_train_transfer_post,snakemake.output['bart_train_transfer_post'])
utils.save_data(X_test_transfer_post,snakemake.output['bart_test_transfer_post'])