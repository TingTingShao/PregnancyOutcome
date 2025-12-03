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
from functions.bestBinaryImputer import oracle_brier_binary_all_missing, oracle_brier_binary_all_missing_worse
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

# ext_data=pd.read_csv("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/data/external_validation_data.csv")


# prepare mclernon data
# maclernon model only has cycle as time variable
        # age, bmi, amh, FullTermBirths, MaleInfertility, POCS, Uterine, Unexplained, OvulDisorder
cols_pre_had=['female_age', 'female_BMI', 'AMH', 'ovulation_problem']
cols_pre_missing=["FullTermBirths", "MaleInfertility", "polycpcos", "Uterine", "Unexplained"]
cols_post_had=['female_age', 'female_BMI','n.oocytes_retrieved', 'ovulation_problem']
cols_post_missing=["MaleInfertility", "polycpcos", "Uterine"]

X_test_pre_had=X_ts[cols_pre_had].copy()
X_train_pre_had=X_tr[cols_pre_had].copy()
X_test_post_had=X_ts[cols_post_had].copy()
X_train_post_had=X_tr[cols_post_had].copy()

# X_ext_data_pre=ext_data[cols_pre_had].copy()
# X_ext_data_post=ext_data[cols_post_had].copy()
# y_ext=utils.construct_y_array(ext_data[["event", "time"]])

mf_template = MissForest(
    clf=RandomForestClassifier(n_jobs=-1),
    rgr=RandomForestRegressor(n_jobs=-1),
    categorical=['ovulation_problem'],
    max_iter=50
)
mf=copy.deepcopy(mf_template)
X_train_pre_had_imputed = mf.fit_transform(X_train_pre_had).sort_index(axis=1)
X_test_pre_had_imputed = mf.transform(X_test_pre_had).sort_index(axis=1)
# X_ext_data_pre_imputed = mf.transform(X_ext_data_pre).sort_index(axis=1)

mf=copy.deepcopy(mf_template)
X_train_post_had_imputed = mf.fit_transform(X_train_post_had).sort_index(axis=1)
X_test_post_had_imputed = mf.transform(X_test_post_had).sort_index(axis=1)
# X_ext_data_post_imputed = mf.transform(X_ext_data_post).sort_index(axis=1)

# rename columns to match mcLernon
map={
    'female_age':   'age',
    "female_BMI":  'bmi',
    "AMH":        'amh',
    'ovulation_problem': 'OvulDisorder',
    'n.oocytes_retrieved': 'retr',
}
X_train_pre_had_imputed.rename(columns=map, inplace=True)
X_train_post_had_imputed.rename(columns=map, inplace=True)
X_test_pre_had_imputed.rename(columns=map, inplace=True)
X_test_post_had_imputed.rename(columns=map, inplace=True)
# X_ext_data_pre_imputed.rename(columns=map, inplace=True)
# print(X_ext_data_pre_imputed.columns)
# X_ext_data_post_imputed.rename(columns=map, inplace=True)
# print(X_ext_data_post_imputed.columns)

# X_test_pre_imputed=X_test_pre_had_imputed.copy()
# X_test_post_imputed=X_test_post_had_imputed.copy()
# # create missing columns
# for c in cols_pre_missing:
#     X_test_pre_imputed[c]=np.nan
# for c in cols_post_missing:
#     X_test_post_imputed[c]=np.nan
# assert X_test_pre_had_imputed.index.equals(y_ts_over1cycle.index)
# assert X_test_post_had_imputed.index.equals(y_ts_over2cycle.index)

# X_test_pre_imputed, choices_pre=oracle_brier_binary_all_missing(
#     X_test_pre_imputed, missing_cols=cols_pre_missing,
#     y_true=y_ts_over1cycle.to_numpy(),
#     prob_predictor=mclernon_pre_prob_predictor(mclernon, None),
#     copy=True,
# )
# X_test_post_imputed, choices_post=oracle_brier_binary_all_missing(
#     X_test_post_imputed, missing_cols=cols_post_missing,
#     y_true=y_ts_over2cycle.to_numpy(),
#     prob_predictor=mclernon_post_prob_predictor(mclernon, None),
#     copy=True,
# )

# X_test_pre_imputed_worse, _=oracle_brier_binary_all_missing_worse(
#     X_test_pre_imputed, missing_cols=cols_pre_missing,
#     y_true=y_ts_over1cycle.to_numpy(),
#     prob_predictor=mclernon_pre_prob_predictor(mclernon, None),
#     copy=True,
# )
# X_test_post_imputed_worse, _=oracle_brier_binary_all_missing_worse(
#     X_test_post_imputed, missing_cols=cols_post_missing,
#     y_true=y_ts_over2cycle.to_numpy(),
#     prob_predictor=mclernon_post_prob_predictor(mclernon, None),
#     copy=True,
# )

# mclernon_data={
#     "pre": X_test_pre_imputed,
#     "post": X_test_post_imputed,
#     "pre_worse": X_test_pre_imputed_worse,
#     "post_worse": X_test_post_imputed_worse
# }

mclernon_data_had={
    "pre_had": (X_train_pre_had_imputed, X_test_pre_had_imputed),
    "post_had": (X_train_post_had_imputed, X_test_post_had_imputed),
    # "ext": (X_ext_data_pre_imputed, X_ext_data_post_imputed, y_ext)
}
# utils.save_data(mclernon_data, snakemake.output['mclernon_data'])
utils.save_data(mclernon_data_had, snakemake.output['mclernon_data_had'])


