import sys
import functions.utils as utils
import functions.evalSurv as evalSurv
import pandas as pd
utils=utils.Utils()
evalSurv=evalSurv.SurvivalMetrics()
import os
# get all the predictions
sys.stderr=open(snakemake.log[0], "w", buffering=1)

data=utils.load_data(snakemake.input['mclernon_data_for_refit'])
preds=utils.load_data(snakemake.input['mclernon_preds'])

data_all=utils.load_data("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl")
y_tr_cycle, y_ts_cycle = data_all['y_cycle']

# X_ext_pre, X_ext_post, y_ext=data['unstacked']['ext']
_, X_te_pre, y_tr_pre, y_te_pre = data['unstacked']['pre']
_, X_te_post, y_tr_post, y_te_post = data['unstacked']['post']

def eval_helper(y_tr, X_te, y_te, name, surv_probs):
    print(surv_probs.shape)
    df_brier, _ = evalSurv.brier_surv_time_dependent(model=None, y_tr=y_tr, 
                                                     X_te=X_te, 
                                                     y_te=y_te, 
                                                     estimates=surv_probs, 
                                                     name=name)
    df_auc, _ = evalSurv.auc_surv_time_dependent(model=None, y_tr=y_tr, 
                                               X_te=X_te, 
                                               y_te=y_te, 
                                               estimates=1-surv_probs, 
                                               name=name)
    df=pd.merge(df_brier, df_auc, on=['model','time'])

    return df

S_preds_pre_cycle={
    'mcLernon_refit': preds['pre'],
    # 'mclernon_ext': preds['ext'][0]
}
S_preds_post_cycle={
    'mcLernon_refit': preds['post'],
    # 'mclernon_ext': preds['ext'][1]
}

rows_pre_cycle = []
for name, S in S_preds_pre_cycle.items():
    print(f"Evaluating pre cycle: {name}")
    y_tr=y_tr_pre
    y_te=y_te_pre
    X_te=X_te_pre
    df=eval_helper(y_tr=y_tr, 
                   X_te=X_te, 
                   y_te=y_te, 
                   name=name, 
                   surv_probs=S)
    rows_pre_cycle.append(df)

df_pre_cycle = pd.concat(rows_pre_cycle, axis=0, ignore_index=True)

rows_post_cycle = []
for name, S in S_preds_post_cycle.items():
    print(f"Evaluating post cycle: {name}")

    y_tr=y_tr_post
    y_te=y_te_post
    X_te=X_te_post
    df=eval_helper(y_tr=y_tr, 
                   X_te=X_te, 
                   y_te=y_te, 
                   name=name, 
                   surv_probs=S)
    rows_post_cycle.append(df)

df_post_cycle = pd.concat(rows_post_cycle, axis=0, ignore_index=True)

utils.save_data(df_pre_cycle, snakemake.output['metrics_pre_cycle'])
utils.save_data(df_post_cycle, snakemake.output['metrics_post_cycle'])
