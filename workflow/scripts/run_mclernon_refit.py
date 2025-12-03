from sklearn.linear_model import LogisticRegression
import sys
import functions.utils as utils
utils=utils.Utils()
from sklearn.base import clone 
sys.stderr=open(snakemake.log[0], "w", buffering=1)

data=utils.load_data(snakemake.input['mclernon_data_for_refit'])


X_tr_pre_stacked, y_tr_stacked, _ = data['stacked']['pre']
X_tr_post_stacked, y_tr_stacked, _ = data['stacked']['post']

template=LogisticRegression(penalty=None,fit_intercept=True, max_iter=1000)
cols_pre=[col for col in X_tr_pre_stacked.columns if col!='id']
cols_post=[col for col in X_tr_post_stacked.columns if col!='id']

model_pre=clone(template)
model_pre.fit(X_tr_pre_stacked[cols_pre], y_tr_stacked)

model_post=clone(template)
model_post.fit(X_tr_post_stacked[cols_post], y_tr_stacked)

models={
    "pre": model_pre,
    "post": model_post
}
utils.save_data(models, snakemake.output['mclernon_model_refit'])
