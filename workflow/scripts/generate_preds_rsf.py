import sys
sys.stderr=open(snakemake.log[0], "w", buffering=1)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from functions.utils import Utils
import numpy as np
import pandas as pd
utils=Utils()

data=utils.load_data(snakemake.input['rsf_data'])
rsf_model="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/rsf_model.pkl"
model=utils.load_data(rsf_model)

_, _, _, _, _, X_test_scaled_pre = data['rsf']['pre']
_, _, _, _, _, X_test_scaled_post = data['rsf']['post']
y_cycle_train, y_cycle_test = data['y_cycle']
y_transfer_train, y_transfer_test = data['y_transfer']

estimates_pre_cycle=model['pre']['cycle'].predict_survival_function(X_test_scaled_pre)
estimates_pre_transfer=model['pre']['transfer'].predict_survival_function(X_test_scaled_pre)
estimates_post_cycle=model['post']['cycle'].predict_survival_function(X_test_scaled_post)
estimates_post_transfer=model['post']['transfer'].predict_survival_function(X_test_scaled_post)

times_transfer=np.arange(1, np.max(y_transfer_test['time'].astype(int))+1)
times_cycle=np.arange(1, np.max(y_cycle_test['time'].astype(int))+1)

S_pre_cycle=np.asarray([[fn(t) for t in times_cycle] for fn in estimates_pre_cycle])
S_pre_transfer=np.asarray([[fn(t) for t in times_transfer] for fn in estimates_pre_transfer])
S_post_cycle=np.asarray([[fn(t) for t in times_cycle] for fn in estimates_post_cycle])
S_post_transfer=np.asarray([[fn(t) for t in times_transfer] for fn in estimates_post_transfer])

preds={
    "pre": {
        "cycle": S_pre_cycle,
        "transfer": S_pre_transfer
    }
    ,
    "post": {
        "cycle": S_post_cycle,
        "transfer": S_post_transfer
    }
}
utils.save_data(preds, snakemake.output['rsf_preds'])

