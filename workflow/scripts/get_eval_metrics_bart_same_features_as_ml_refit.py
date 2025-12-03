import sys
import functions.utils as utils
import functions.evalSurv as evalSurv
import pandas as pd
utils=utils.Utils()
evalSurv=evalSurv.SurvivalMetrics()
import os
# get all the predictions
sys.stderr=open(snakemake.log[0], "w", buffering=1)

# preds_dsa=utils.load_data(snakemake.input['preds_dsa'])
# preds_rsf=utils.load_data(snakemake.input['preds_rsf'])
# data=utils.load_data(snakemake.input['data'])
mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl"
mclernon_data_had="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_had.pkl"
data=utils.load_data(mclernon_data_had)
data_refit=utils.load_data(mclernon_data_for_refit)

# bart preds
FOLDER="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"
S_bart_pre_transformed=pd.read_csv(f"{FOLDER}/BART_pre_cycle_transformed_performance_feats_ml_refit.csv").to_numpy()
S_bart_post_transformed=pd.read_csv(f"{FOLDER}/BART_post_cycle_transformed_performance_feats_ml_refit.csv").to_numpy()
S_bart_pre=pd.read_csv(f"{FOLDER}/BART_pre_cycle_performance_feats_ml_refit.csv").to_numpy()
S_bart_post=pd.read_csv(f"{FOLDER}/BART_post_cycle_performance_feats_ml_refit.csv").to_numpy()


S_preds_bart={
    'pre': {
        'transformed': S_bart_pre_transformed,
        'org': S_bart_pre,
    },
    'post': {
        'transformed': S_bart_post_transformed,
        'org': S_bart_post,
    }
}
X_transformed_pre, X_te_transformed_pre, y_tr_cycle, y_ts_cycle = data_refit['unstacked']['pre']
X_transformed_post, X_te_transformed_post, y_tr_cycle, y_ts_cycle = data_refit['unstacked']['post']

X_tr_pre, X_ts_pre =data['pre_had']
X_tr_post, X_ts_post=data['post_had']


print(y_tr_cycle)

utils.save_data(S_preds_bart, os.path.join(FOLDER, "preds_bart_to_diagnose_ext.pkl"))


# evaluate the predictions
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
    "transformed": S_preds_bart['pre']['transformed'],
    "org": S_preds_bart['pre']['org'],
}


S_preds_post_cycle={
    "transformed": S_preds_bart['post']['transformed'],
    "org": S_preds_bart['post']['org']
}

rows_pre_cycle = []
for name, S in S_preds_pre_cycle.items():
    print(f"Evaluating pre cycle: {name}")
    if name == "transformed":
        X_te=X_te_transformed_pre
        y_tr=y_tr_cycle
        y_te=y_ts_cycle
    else:
        X_te=X_ts_pre
        y_tr=y_tr_cycle
        y_te=y_ts_cycle
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

    if name == "transformed":
        X_te=X_te_transformed_post
        y_tr=y_tr_cycle
        y_te=y_ts_cycle
    else:
        X_te=X_ts_post
        y_tr=y_tr_cycle
        y_te=y_ts_cycle
    df=eval_helper(y_tr=y_tr, 
                    X_te=X_te, 
                    y_te=y_te, 
                    name=name, 
                    surv_probs=S)
    rows_post_cycle.append(df)

df_post_cycle = pd.concat(rows_post_cycle, axis=0, ignore_index=True)


utils.save_data(df_pre_cycle, snakemake.output['metrics_pre_cycle'])
utils.save_data(df_post_cycle, snakemake.output['metrics_post_cycle'])

