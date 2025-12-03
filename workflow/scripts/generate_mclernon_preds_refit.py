import sys
sys.stderr=open(snakemake.log[0], "w", buffering=1)
from functions.utils import Utils
import numpy as np
utils=Utils()
data=utils.load_data(snakemake.input['mclernon_data_for_refit'])
mclernon_model_refit=utils.load_data(snakemake.input['mclernon_model_refit'])
model_pre=mclernon_model_refit['pre']
model_post=mclernon_model_refit['post']
# prepared_data={
#     "stacked":{
#         "pre": (X_tr_pre_stacked, y_tr_stacked, X_te_pre_stacked),
#         "post": (X_tr_post_stacked, y_tr_stacked, X_te_post_stacked)
#     },
#     "unstacked":{
#         "pre": (X_tr_pre, X_te_pre, y_tr_cycle, y_ts_cycle),
#         "post": (X_tr_post, X_te_post, y_tr_cycle, y_ts_cycle)
#     }
# }

X_tr_pre_stacked, _, X_te_pre_stacked = data['stacked']['pre']
X_tr_post_stacked, _, X_te_post_stacked = data['stacked']['post']

# X_ext_pre_stacked, X_ext_post_stacked = data['stacked']['ext']
# X_ext_pre, _, y_ext = data['unstacked']['ext']

X_te_pre = data['unstacked']['pre'][1]
covs_pre=[col for col in X_tr_pre_stacked.columns if col!='id']
covs_post=[col for col in X_tr_post_stacked.columns if col!='id']

def get_preds(model, X_te, cov_cols, N, K):
    hazards_mclernon=model.predict_proba(X_te[cov_cols])[:,1].reshape(N, K)
    S_mclernon=np.cumprod(1-hazards_mclernon, axis=1)

    return S_mclernon

N=X_te_pre.shape[0]
# N_ext=X_ext_pre.shape[0]
K=6
print(X_te_pre_stacked.columns)
print(covs_pre)
S_mclernon_pre=get_preds(model_pre, X_te_pre_stacked[covs_pre], covs_pre, N, K)
S_mclernon_post=get_preds(model_post, X_te_post_stacked[covs_post], covs_post, N, K)
# S_mclernon_ext_post=get_preds(model_post, X_ext_post_stacked[covs_post], covs_post, N_ext, K)
# S_mclernon_ext_pre=get_preds(model_pre, X_ext_pre_stacked[covs_pre], covs_pre, N_ext, K)

utils.save_data({
    'pre':S_mclernon_pre,
    'post':S_mclernon_post,
    # 'ext':(S_mclernon_ext_pre, S_mclernon_ext_post)
}, snakemake.output['mclernon_preds'])

