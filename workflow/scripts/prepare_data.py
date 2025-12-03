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
data=utils.load_data(snakemake.input['postdata'])
print("Data shape:", data.shape)

# 数据为后面的分析做准备
# 总体思想
# 准备数据： survival data, discrete survival analysis (rf, logit), BART, Maclernon 
# 数据处理好后，hyperparameter tuning 
# model fit
# model evaluation

pre_feats  = snakemake.config['FEATURES']['pre']
post_feats = snakemake.config['FEATURES']['post']
print("PRE features:", pre_feats)
print("POST features:", post_feats)

# split training and testing data
data['Date of aspiration'] = pd.to_datetime(data['Date of aspiration'])
data=data.sort_values(by='Date of aspiration', ascending=True)

data['outcome_cycle']=np.where(
    data['total_cycles_taken']>6,
    0,
    data['pregnancy_outcome_binary']
)
data['cycle']=np.where(
    data['total_cycles_taken']>6,
    6,
    data['total_cycles_taken']
)
data['outcome_cycle']=data['outcome_cycle'].map({1: True, 0: False})

n_train = int(data.shape[0] * 0.7)
data_train=data.head(n_train)
data_test=data.tail(data.shape[0] - n_train)

num_features_pre=snakemake.config['FEATURES']['pre']['num_features']
num_features_post=snakemake.config['FEATURES']['post']['num_features'] + num_features_pre
cat_features_pre=snakemake.config['FEATURES']['pre']['cat_features'] 
cat_features_post=snakemake.config['FEATURES']['post']['cat_features'] + cat_features_pre

cols_pre=num_features_pre + cat_features_pre
cols_post=num_features_post + cat_features_post

def get_train_test(data_train, data_test, cols, cat_features, max_iter=50):
    X_tr=data_train[cols]
    X_ts=data_test[cols]
    print(X_tr.columns)
    print(cols)
    print(cat_features)
    mf = MissForest(
        clf=RandomForestClassifier(n_jobs=-1),
        rgr=RandomForestRegressor(n_jobs=-1),
        categorical=cat_features,
        max_iter=max_iter
    )
    X_train_imputed = mf.fit_transform(X_tr).sort_index(axis=1)
    X_test_imputed = mf.transform(X_ts).sort_index(axis=1)

    # scale data
    num_features = [c for c in cols if c not in cat_features]
    params=utils.fit_scale(X_train_imputed, num_features)
    X_train_scaled=utils.transform_scale(X_train_imputed,params)
    X_test_scaled=utils.transform_scale(X_test_imputed,params)
    return X_tr, X_ts, X_train_imputed, X_test_imputed, X_train_scaled, X_test_scaled

max_iter=50

prepared_data={
    'rsf': {
        'pre': get_train_test(data_train, data_test, cols_pre, cat_features_pre, max_iter),
        'post': get_train_test(data_train, data_test, cols_post, cat_features_post, max_iter)
    },
    'y_cycle': (utils.construct_y_array(data_train[['outcome_cycle', 'cycle']].rename(columns={'outcome_cycle': 'event', 'cycle': 'time'})), 
                utils.construct_y_array(data_test[['outcome_cycle', 'cycle']].rename(columns={'outcome_cycle': 'event', 'cycle': 'time'}))),
    "y_transfer": (utils.construct_y_array(data_train[['outcome_cycle', 'n.transfers']].rename(columns={'outcome_cycle': 'event', 'n.transfers': 'time'})), 
                   utils.construct_y_array(data_test[[ 'outcome_cycle', 'n.transfers']].rename(columns={'outcome_cycle': 'event', 'n.transfers': 'time'}))),
    'y_over1cycle': (data_train['succeed_over_1cycles'], data_test['succeed_over_1cycles']),
    'y_over2cycle': (data_train['succeed_over_2cycles'], data_test['succeed_over_2cycles']),
}

utils.save_data(prepared_data, snakemake.output[0])