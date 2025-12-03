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

data=utils.load_data(snakemake.input['sda_data'])
print(data.keys())

X_tr_pre_cycle, y_tr_pre_cycle, _= data['pre']['cycle']
X_tr_post_cycle, y_tr_post_cycle, _= data['post']['cycle']
X_tr_pre_transfer, y_tr_pre_transfer, _= data['pre']['transfer']
X_tr_post_transfer, y_tr_post_transfer, _= data['post']['transfer']

covs_pre_cycle=[col for col in X_tr_pre_cycle.columns if col!='id']
covs_post_cycle=[col for col in X_tr_post_cycle.columns if col!='id']
covs_pre_transfer=[col for col in X_tr_pre_transfer.columns if col!='id']
covs_post_transfer=[col for col in X_tr_post_transfer.columns if col!='id']

rfm=RandomForestClassifier(random_state=0)
logit=LogisticRegression(solver='saga', penalty='elasticnet', random_state=0)

rf_parameters={
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', None],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],
}

# logistic regression
log_parameters={
    'C': np.logspace(-4, 2, num=25),
    'l1_ratio': np.linspace(0.0, 1.0, 11)
}

def search_best_params(X_tr, y_tr, cov_cols):
    # stratafied group setup
    sgkfold=StratifiedGroupKFold(n_splits=5, shuffle=False)
    cvstrat=sgkfold.split(X_tr[cov_cols], y_tr, groups=X_tr['id']) 
    sgkfold_lg=StratifiedGroupKFold(n_splits=5, shuffle=False)
    cvstrat_lg=sgkfold_lg.split(X_tr[cov_cols], y_tr, groups=X_tr['id']) 
    # grid search
    grid_search=GridSearchCV(estimator=rfm, param_grid=rf_parameters, cv=cvstrat, n_jobs=-1, scoring='neg_log_loss', verbose=2)
    grid_search_lg=GridSearchCV(estimator=logit, param_grid=log_parameters, cv=cvstrat_lg, n_jobs=-1, scoring='neg_log_loss', verbose=2)

    # fit the model
    grid_search.fit(X_tr[cov_cols], y_tr, groups=X_tr['id'])

    grid_search_lg.fit(X_tr[cov_cols], y_tr, groups=X_tr['id'])

    # best params
    best_rfm=grid_search.best_estimator_
    best_logit=grid_search_lg.best_estimator_

    # save the model
    return best_rfm, best_logit

best_rf_pre_cycle, best_lg_pre_cycle=search_best_params(X_tr_pre_cycle, y_tr_pre_cycle, covs_pre_cycle)
best_rf_post_cycle, best_lg_post_cycle=search_best_params(X_tr_post_cycle, y_tr_post_cycle, covs_post_cycle)
best_rf_pre_transfer, best_lg_pre_transfer=search_best_params(X_tr_pre_transfer, y_tr_pre_transfer, covs_pre_transfer)
best_rf_post_transfer, best_lg_post_transfer=search_best_params(X_tr_post_transfer, y_tr_post_transfer, covs_post_transfer)

models={
    "pre": {
        "cycle": {
            "rf": best_rf_pre_cycle,
            "logit": best_lg_pre_cycle
        },
        "transfer": {
            "rf": best_rf_pre_transfer,
            "logit": best_lg_pre_transfer
        }
    },
    "post": {
        "cycle": {
            "rf": best_rf_post_cycle,
            "logit": best_lg_post_cycle
        },
        "transfer": {
            "rf": best_rf_post_transfer,
            "logit": best_lg_post_transfer
        }
    }
}

utils.save_data(models, snakemake.output['dsa_model'])
