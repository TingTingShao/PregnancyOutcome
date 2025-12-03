import sys
sys.stderr=open(snakemake.log[0], "w", buffering=1)
from functions.utils import Utils
utils=Utils()
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV
import pandas as pd
from functions.searchRSF import SearchRSF
searchRSF=SearchRSF()
data=utils.load_data(snakemake.input['prepared_data'])
print(data.keys())

_, _, _, _, X_train_scaled_pre, _ = data['rsf']['pre']
_, _, _, _, X_train_scaled_post, _ = data['rsf']['post']

y_cycle_train, y_cycle_test = data['y_cycle']
y_transfer_train, y_transfer_test = data['y_transfer']

rf_parameters={
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', None],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],
}

model_pre_cycle, _=searchRSF.select_hyperparameters(
    X_train_scaled_pre,
    y_cycle_train,
    RandomSurvivalForest(),
    rf_parameters,
    cv=5,
    random_state=0
)
model_post_cycle, _=searchRSF.select_hyperparameters(
    X_train_scaled_post,
    y_cycle_train,
    RandomSurvivalForest(),
    rf_parameters,
    cv=5,
    random_state=0
)

model_pre_transfer, _=searchRSF.select_hyperparameters(
    X_train_scaled_pre,
    y_transfer_train,
    RandomSurvivalForest(),
    rf_parameters,
    cv=5,
    random_state=0
)
model_post_transfer, _=searchRSF.select_hyperparameters(
    X_train_scaled_post,
    y_transfer_train,
    RandomSurvivalForest(),
    rf_parameters,
    cv=5,
    random_state=0
)

models={
    "pre": {
        "cycle": model_pre_cycle,
        "transfer": model_pre_transfer,
    },
    "post": {
        "cycle": model_post_cycle,
        "transfer": model_post_transfer,
    }
}

utils.save_data(models, snakemake.output['rsf_model'])
