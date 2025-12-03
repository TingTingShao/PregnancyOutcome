import sys
import functions.utils as utils
import functions.evalSurv as evalSurv
import pandas as pd
utils=utils.Utils()
evalSurv=evalSurv.SurvivalMetrics()
import os
# get all the predictions
sys.stderr=open(snakemake.log[0], "w", buffering=1)

preds_dsa=utils.load_data(snakemake.input['preds_dsa'])
preds_rsf=utils.load_data(snakemake.input['preds_rsf'])
data=utils.load_data(snakemake.input['data'])


# bart preds
FOLDER="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"
S_bart_pre_cycle=pd.read_csv(f"{FOLDER}/preds_cycle_pre_best.csv").to_numpy()
S_bart_post_cycle=pd.read_csv(f"{FOLDER}/preds_cycle_post_best.csv").to_numpy()
# S_bart_pre_transfer=pd.read_csv(f"{FOLDER}/preds_transfer_pre.csv").to_numpy()
# S_bart_post_transfer=pd.read_csv(f"{FOLDER}/preds_transfer_post.csv").to_numpy()

S_preds_bart={
    'pre': {
        'cycle': S_bart_pre_cycle,
        # 'transfer': S_bart_pre_transfer
    },
    'post': {
        'cycle': S_bart_post_cycle,
        # 'transfer': S_bart_post_transfer
    }
}

utils.save_data(S_preds_bart, os.path.join(FOLDER, "preds_bart.pkl"))

y_train_cycle, y_test_cycle = data['y_cycle']
y_train_transfer, y_test_transfer = data['y_transfer']

_, _, _, _, _, X_test_scaled_pre = data['rsf']['pre']
_, _, _, _, _, X_test_scaled_post = data['rsf']['post']

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
    'BART': S_preds_bart['pre']['cycle'],
    "RF": preds_dsa['pre']['cycle']['rfm'],
    'Logit': preds_dsa['pre']['cycle']['logit'],
    'RSF': preds_rsf['pre']['cycle']
}

# S_preds_pre_transfer={
#     'BART': S_preds_bart['pre']['transfer'],
#     "RF": preds_dsa['pre']['transfer']['rfm'],
#     'Logit': preds_dsa['pre']['transfer']['logit'],
#     'RSF': preds_rsf['pre']['transfer']
# }

S_preds_post_cycle={
    'BART': S_preds_bart['post']['cycle'],
    "RF": preds_dsa['post']['cycle']['rfm'],
    'Logit': preds_dsa['post']['cycle']['logit'],
    'RSF': preds_rsf['post']['cycle']
}

# S_preds_post_transfer={
#     'BART': S_preds_bart['post']['transfer'],
#     "RF": preds_dsa['post']['transfer']['rfm'],
#     'Logit': preds_dsa['post']['transfer']['logit'],
#     'RSF': preds_rsf['post']['transfer']
# }

rows_pre_cycle = []
for name, S in S_preds_pre_cycle.items():
    print(f"Evaluating pre cycle: {name}")
    df=eval_helper(y_tr=y_train_cycle, 
                   X_te=X_test_scaled_pre, 
                   y_te=y_test_cycle, 
                   name=name, 
                   surv_probs=S)
    rows_pre_cycle.append(df)

df_pre_cycle = pd.concat(rows_pre_cycle, axis=0, ignore_index=True)

# print(y_train_transfer['time'].min(), y_train_transfer['time'].max()
#       , y_test_transfer['time'].min(), y_test_transfer['time'].max()    )
# rows_pre_transfer = []
# for name, S in S_preds_pre_transfer.items():
#     print(f"Evaluating pre transfer: {name}")
#     df=eval_helper(y_tr=y_train_transfer, 
#                    X_te=X_test_scaled_pre, 
#                    y_te=y_test_transfer, 
#                    name=name, 
#                    surv_probs=S)
#     rows_pre_transfer.append(df)

# df_pre_transfer = pd.concat(rows_pre_transfer, axis=0, ignore_index=True)

rows_post_cycle = []
for name, S in S_preds_post_cycle.items():
    print(f"Evaluating post cycle: {name}")
    df=eval_helper(y_tr=y_train_cycle, 
                   X_te=X_test_scaled_post, 
                   y_te=y_test_cycle, 
                   name=name, 
                   surv_probs=S)
    rows_post_cycle.append(df)

df_post_cycle = pd.concat(rows_post_cycle, axis=0, ignore_index=True)

# rows_post_transfer = []
# for name, S in S_preds_post_transfer.items():
#     print(f"Evaluating post transfer: {name}")
#     df=eval_helper(y_tr=y_train_transfer, 
#                    X_te=X_test_scaled_post, 
#                    y_te=y_test_transfer, 
#                    name=name,
#                    surv_probs=S)
#     rows_post_transfer.append(df)

# df_post_transfer = pd.concat(rows_post_transfer, axis=0, ignore_index=True)


utils.save_data(df_pre_cycle, snakemake.output['metrics_pre_cycle'])
# utils.save_data(df_pre_transfer, snakemake.output['metrics_pre_transfer'])
utils.save_data(df_post_cycle, snakemake.output['metrics_post_cycle'])
# utils.save_data(df_post_transfer, snakemake.output['metrics_post_transfer'])
