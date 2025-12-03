from functions.plot_calibration import plot_calibration
import functions.utils as utils
utils=utils.Utils()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

prepared_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl"
sda_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_dsa.pkl"
rsf_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_rsf.pkl"
mclernon_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_mclernon_refit.pkl"

FOLDER="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"
S_preds_bart=utils.load_data(os.path.join(FOLDER, "preds_bart.pkl"))
# S_pred_bart_aft=utils.load_data(os.path.join(FOLDER, "preds_bart_aft_feature_selection.pkl"))
S_preds_sda=utils.load_data(sda_preds)
S_preds_rsf=utils.load_data(rsf_preds)
S_preds_mclernon=utils.load_data(mclernon_preds)


data=utils.load_data(prepared_data)


y_train_cycle, y_test_cycle = data['y_cycle']

# dsa_data=utils.load_data("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_data.pkl")
# _, _, y_ext_re, y_ext_post = dsa_data['ext']['unstacked']

# sda_preds_manual_selection="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_dsa_aft_manual_selection.pkl"
# rsf_preds_manual_selection="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_rsf_aft_manual_selection.pkl"
# sda_preds_manual_selection=utils.load_data(sda_preds_manual_selection)
# rsf_preds_manual_selection=utils.load_data(rsf_preds_manual_selection)
# S_preds_bart_pre_manual_selection=utils.load_data(os.path.join(FOLDER, "preds_cycle_pre_reduced.csv"))
# S_preds_bart_post_manual_selection=utils.load_data(os.path.join(FOLDER, "preds_cycle_post_reduced.csv"))
# S_preds_bart_pre_manual_selection5=utils.load_data(os.path.join(FOLDER, "preds_cycle_pre_reduced5.csv"))
# S_preds_bart_post_manual_selection5=utils.load_data(os.path.join(FOLDER, "preds_cycle_post_reduced5.csv"))
# S_pre_cycle_mclernon={
#     'BART': S_preds_bart['pre']['cycle'],
#     "RF": S_preds_sda['pre']['cycle']['rfm'],
#     'Logit': S_preds_sda['pre']['cycle']['logit'],
#     'RSF': S_preds_rsf['pre']['cycle'],
    
#     # 'McLernon_worst': S_preds_mclernon['pre_worse'],
#     # 'McLernon_sim': S_preds_mclernon_simplified['pre'],
# }

# S_post_cycle_mclernon={
#     'BART': S_preds_bart['post']['cycle'],
#     "RF": S_preds_sda['post']['cycle']['rfm'],
#     'Logit': S_preds_sda['post']['cycle']['logit'],
#     'RSF': S_preds_rsf['post']['cycle'],
#     'McLernon': S_preds_mclernon['post'],
#     # 'McLernon_worst': S_preds_mclernon['post_worse'],
#     # 'McLernon_sim': S_preds_mclernon_simplified['post'],
# }

S_pre_cycle={
    'BART': S_preds_bart['pre']['cycle'],
    # 'BART_sparse': S_pred_bart_aft['pre']['cycle'],
    "RF": S_preds_sda['pre']['cycle']['rfm'],
    'Logit': S_preds_sda['pre']['cycle']['logit'],
    'RSF': S_preds_rsf['pre']['cycle'],
    'McLernon': S_preds_mclernon['pre'],
    # 'McLernon': S_preds_mclernon_simplified['pre'],
}

S_post_cycle={
    'BART': S_preds_bart['post']['cycle'],
    # 'BART_sparse': S_pred_bart_aft['post']['cycle'],
    "RF": S_preds_sda['post']['cycle']['rfm'],
    'Logit': S_preds_sda['post']['cycle']['logit'],
    'RSF': S_preds_rsf['post']['cycle'],
    'McLernon': S_preds_mclernon['post']
    # 'McLernon': S_preds_mclernon_simplified['post'],
}

# S_pre_cycle_manual_selection={
#     'BART': S_preds_bart_pre_manual_selection,
#     'BART_eur': S_preds_bart_pre_manual_selection5,
#     "RF": sda_preds_manual_selection['pre']['cycle']['rfm'],
#     'Logit': sda_preds_manual_selection['pre']['cycle']['logit'],
#     'RSF': rsf_preds_manual_selection['pre']['cycle'],
# }
# S_post_cycle_manual_selection={
#     'BART': S_preds_bart_post_manual_selection,
#     'BART_eur': S_preds_bart_post_manual_selection5,
#     "RF": sda_preds_manual_selection['post']['cycle']['rfm'],
#     'Logit': sda_preds_manual_selection['post']['cycle']['logit'],
#     'RSF': rsf_preds_manual_selection['post']['cycle'],
# }

# plot_calibration(y_te=y_test_cycle, S_pred=S_pre_cycle_mclernon, times=[1, 2, 3], path=snakemake.output['pre_cycle_mclernon'])
# plot_calibration(y_te=y_test_cycle, S_pred=S_post_cycle_mclernon, times=[2, 3], path=snakemake.output['post_cycle_mclernon'])
fig, axes=plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)
plot_calibration(y_te=y_test_cycle, S_pred=S_pre_cycle, times=[1, 2, 3, 4, 5, 6], ax=axes[0])
axes[0].set_title("Pretreatment", fontsize=20)
# set legend size
axes[0].legend(fontsize=14)
plot_calibration(y_te=y_test_cycle, S_pred=S_post_cycle, times=[1, 2, 3, 4, 5, 6], ax=axes[1])
axes[1].set_title("Posttreatment", fontsize=20)
# set legend size
axes[1].legend(fontsize=14)
fig.savefig(snakemake.output['figure5'], dpi=1200)
# fig.close()

# fig, axes=plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True, sharex=True)
# plot_calibration(y_te=y_ext_re, S_pred=S_pre_cycle_manual_selection, times=[1, 2, 3, 4, 5, 6], ax=axes[0])
# axes[0].set_title("Pretreatment, external data", fontsize=20)
# plot_calibration(y_te=y_ext_re, S_pred=S_post_cycle_manual_selection, times=[1, 2, 3, 4, 5, 6], ax=axes[1])
# axes[1].set_title("Posttreatment, external data", fontsize=20)
# fig.savefig(snakemake.output['figure6'], dpi=1200)
# fig.close()