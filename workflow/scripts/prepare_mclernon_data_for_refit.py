import sys
import functions.utils as utils
from functions.mcLernon import McLernon
import os
import pandas as pd
import numpy as np
from survstack import SurvivalStacker
utils=utils.Utils()

sys.stderr=open(snakemake.log[0], "w", buffering=1)

data=utils.load_data(snakemake.input['mclernon_data_had'])
mclernon=McLernon()

data_all=utils.load_data("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl")

X_tr_pre, X_te_pre=data['pre_had']
X_tr_post, X_te_post=data['post_had']

# X_ext_pre, X_ext_post, y_ext=data['ext']

X_tr_pre=mclernon.construct_dataset_pre(X_tr_pre)
X_te_pre=mclernon.construct_dataset_pre(X_te_pre)

X_tr_post=mclernon.construct_dataset_post(X_tr_post)
X_te_post=mclernon.construct_dataset_post(X_te_post)

# X_ext_pre=mclernon.construct_dataset_pre(X_ext_pre)
# X_ext_post=mclernon.construct_dataset_post(X_ext_post)

y_tr_cycle, y_ts_cycle = data_all['y_cycle']

X_tr_pre_id=X_tr_pre.copy()
X_tr_post_id=X_tr_post.copy()

# X_ext_pre_id=X_ext_pre.copy()
# X_ext_post_id=X_ext_post.copy()
# X_ext_pre_id['id']=X_ext_pre_id.index
# X_ext_post_id['id']=X_ext_post_id.index

X_tr_pre_id['id']=X_tr_pre_id.index
X_tr_post_id['id']=X_tr_post_id.index
X_te_pre_id=X_te_pre.copy()
X_te_post_id=X_te_post.copy()
X_te_pre_id['id']=X_te_pre_id.index
X_te_post_id['id']=X_te_post_id.index


ss=SurvivalStacker(time_encoding="onehot")
X_tr_pre_stacked, y_tr_stacked = ss.fit_transform(X_tr_pre_id.to_numpy(), y_tr_cycle)
X_tr_post_stacked, y_tr_stacked = ss.fit_transform(X_tr_post_id.to_numpy(), y_tr_cycle)


cov_stacked_pre=list(X_tr_pre_id.columns) + [f"risk_{t}" for t in range(1, 7)]
cov_stacked_post=list(X_tr_post_id.columns) + [f"risk_{t}" for t in range(1, 7)]

X_tr_pre_stacked=pd.DataFrame(X_tr_pre_stacked, columns=cov_stacked_pre)
X_tr_post_stacked=pd.DataFrame(X_tr_post_stacked, columns=cov_stacked_post)

X_te_pre_stacked=utils.build_stacked_test(X_te_pre_id, list(X_tr_pre_id.columns), range(1,7))
X_te_post_stacked=utils.build_stacked_test(X_te_post_id, list(X_tr_post_id.columns), range(1,7))
# X_ext_pre_stacked=utils.build_stacked_test(X_ext_pre_id, list(X_tr_pre_id.columns), range(1,7))
# X_ext_post_stacked=utils.build_stacked_test(X_ext_post_id, list(X_tr_post_id.columns), range(1,7))


prepared_data={
    "stacked":{
        "pre": (X_tr_pre_stacked, y_tr_stacked, X_te_pre_stacked),
        "post": (X_tr_post_stacked, y_tr_stacked, X_te_post_stacked),
        # "ext": (X_ext_pre_stacked, X_ext_post_stacked)
    },
    "unstacked":{
        "pre": (X_tr_pre, X_te_pre, y_tr_cycle, y_ts_cycle),
        "post": (X_tr_post, X_te_post, y_tr_cycle, y_ts_cycle),
        # "ext": (X_ext_pre, X_ext_post, y_ext)
    }
}

utils.save_data(prepared_data, snakemake.output['mclernon_data_for_refit'])