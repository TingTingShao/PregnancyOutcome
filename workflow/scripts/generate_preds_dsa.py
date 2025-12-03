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

data=utils.load_data(snakemake.input['dsa_data'])
model=utils.load_data("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_model.pkl")
pre_cycle=model['pre']['cycle']
post_cycle=model['post']['cycle']
pre_transfer=model['pre']['transfer']
post_transfer=model['post']['transfer']

X_tr_pre_cycle, y_tr_pre_cycle, X_te_pre_cycle = data['pre']['cycle']
X_tr_post_cycle, y_tr_post_cycle, X_te_post_cycle = data['post']['cycle']
X_tr_pre_transfer, y_tr_pre_transfer, X_te_pre_transfer = data['pre']['transfer']
X_tr_post_transfer, y_tr_post_transfer, X_te_post_transfer = data['post']['transfer']

# number of time points equal to the number of risk columns that starts with 'risk_'
covs_pre=[col for col in X_tr_pre_cycle.columns if col!='id']
covs_post=[col for col in X_tr_post_cycle.columns if col!='id']
covs_pre_transfer=[col for col in X_tr_pre_transfer.columns if col!='id']
covs_post_transfer=[col for col in X_tr_post_transfer.columns if col!='id']
K_transfers=len([col for col in X_te_pre_transfer.columns if col.startswith('risk_')])
K_cycles=len([col for col in X_te_pre_cycle.columns if col.startswith('risk_')])
N=data['N_test']
def get_preds(model, X_te, cov_cols, N, K):
    rfm=model['rf']
    logit=model['logit']
    hazards_rfm=rfm.predict_proba(X_te[cov_cols])[:,1].reshape(N, K)
    hazards_logit=logit.predict_proba(X_te[cov_cols])[:,1].reshape(N, K)
    S_rfm=np.cumprod(1-hazards_rfm, axis=1)
    S_logit=np.cumprod(1-hazards_logit, axis=1)
    return S_rfm, S_logit

S_rfm_pre_cycle, S_logit_pre_cycle = get_preds(pre_cycle, X_te_pre_cycle, covs_pre, N, K_cycles)
S_rfm_post_cycle, S_logit_post_cycle = get_preds(post_cycle, X_te_post_cycle, covs_post, N, K_cycles)
S_rfm_pre_transfer, S_logit_pre_transfer = get_preds(pre_transfer, X_te_pre_transfer, covs_pre_transfer, N, K_transfers)
S_rfm_post_transfer, S_logit_post_transfer = get_preds(post_transfer, X_te_post_transfer, covs_post_transfer, N, K_transfers)


S_preds={
    'pre': {
        'cycle': {
            'rfm': S_rfm_pre_cycle,
            'logit': S_logit_pre_cycle
        },
        'transfer': {
            'rfm': S_rfm_pre_transfer,
            'logit': S_logit_pre_transfer
        }
    },
    'post': {
        'cycle': {
            'rfm': S_rfm_post_cycle,
            'logit': S_logit_post_cycle
        },
        'transfer': {
            'rfm': S_rfm_post_transfer,
            'logit': S_logit_post_transfer
        }
    }
}

utils.save_data(S_preds, snakemake.output['sda_preds'])