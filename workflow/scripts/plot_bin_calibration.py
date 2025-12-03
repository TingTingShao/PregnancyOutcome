from functions.plot_binned_calibration import plot_binned_calibration
import functions.utils as utils
utils=utils.Utils()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd

prepared_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl"
sda_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_dsa.pkl"
rsf_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_rsf.pkl"
mclernon_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_mclernon_refit.pkl"
# S_preds_mclernon_simplified="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_mclernon_simplified.pkl"
FOLDER="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables"

S_preds_bart=utils.load_data(os.path.join(FOLDER, "preds_bart.pkl"))
S_preds_bart_simplified=utils.load_data(os.path.join(FOLDER, "preds_bart_aft_feature_selection.pkl"))

S_preds_sda=utils.load_data(sda_preds)
S_preds_rsf=utils.load_data(rsf_preds)
S_preds_mclernon=utils.load_data(mclernon_preds)
# S_preds_mclernon_simplified=utils.load_data(S_preds_mclernon_simplified)
data=utils.load_data(prepared_data)
y_train_cycle, y_test_cycle = data['y_cycle']

bart_S_pre_cycle=S_preds_bart['pre']['cycle']
bart_S_post_cycle=S_preds_bart['post']['cycle']
bart_S_pre_cycle_simplified=pd.read_csv(os.path.join(FOLDER, "preds_cycle_pre_aft_feature_selection.csv"))
bart_S_post_cycle_simplified=pd.read_csv(os.path.join(FOLDER, "preds_cycle_post_aft_feature_selection.csv"))
# bart_S_post_cycle_simplified=S_preds_bart_simplified['post']['cycle']

sda_S_pre_cycle_logit=S_preds_sda['pre']['cycle']['logit']
sda_S_post_cycle_logit=S_preds_sda['post']['cycle']['logit']
sda_S_pre_cycle_rfm=S_preds_sda['pre']['cycle']['rfm']
sda_S_post_cycle_rfm=S_preds_sda['post']['cycle']['rfm']
rsf_S_pre_cycle=S_preds_rsf['pre']['cycle']
rsf_S_post_cycle=S_preds_rsf['post']['cycle']
Mclernon_pre_cycle=S_preds_mclernon['pre']
Mclernon_post_cycle=S_preds_mclernon['post']
Mclernon_ext_pre_cycle=S_preds_mclernon['ext'][0]
Mclernon_ext_post_cycle=S_preds_mclernon['ext'][1]


pre_cycle=[Mclernon_pre_cycle, bart_S_pre_cycle, bart_S_pre_cycle_simplified]
post_cycle=[Mclernon_post_cycle, bart_S_post_cycle, bart_S_post_cycle_simplified]

def plot_figure(preds, y_test_cycle, time, output_path=None, strategy='uniform'):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

    axes[0], table = plot_binned_calibration(
        y_te=y_test_cycle,
        pred=preds[0],
        time=time,
        idx=time,
        label="Mclernon",
        return_table=True,
        ax=axes[0],
        strategy=strategy
    )

    # axes[0, 1], table = plot_binned_calibration(
    #     y_te=y_test_cycle,
    #     pred=preds[1],
    #     time=time,
    #     idx=time,
    #     label="Mclernon ext",
    #     return_table=True,
    #     ax=axes[0, 1],
    # )


    axes[1], table = plot_binned_calibration(
        y_te=y_test_cycle,
        pred=preds[1],
        time=time,
        idx=time,
        label="Bart",
        return_table=True,
        ax=axes[1],
        strategy=strategy
    )
    axes[2], table = plot_binned_calibration(
        y_te=y_test_cycle,
        pred=preds[2],
        time=time,
        idx=time,
        label="Bart_sparse",
        return_table=True,
        ax=axes[2],
        strategy=strategy
    )

    # axes[1, 0], table = plot_binned_calibration(
    #     y_te=y_test_cycle,
    #     pred=preds[3],
    #     time=time,
    #     idx=time,
    #     label="Logit",
    #     return_table=True,
    #     ax=axes[1, 0],
    #     strategy=strategy
    # )
    # axes[1, 1], table = plot_binned_calibration(
    #     y_te=y_test_cycle,
    #     pred=preds[4],
    #     time=time,
    #     idx=time,
    #     label="RF",
    #     return_table=True,
    #     ax=axes[1, 1],
    #     strategy=strategy
    # )
    # axes[1, 2], table = plot_binned_calibration(
    #     y_te=y_test_cycle,
    #     pred=preds[5],
    #     time=time,
    #     idx=time,
    #     label="RSF",
    #     return_table=True,
    #     ax=axes[1, 2],
    #     strategy=strategy
    # )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=800)
    plt.close(fig)


plot_figure(pre_cycle, y_test_cycle, time=3, output_path=snakemake.output['pre_cycle_t3'], strategy='uniform')
plot_figure(pre_cycle, y_test_cycle, time=2, output_path=snakemake.output['pre_cycle_t2'], strategy='uniform')
plot_figure(pre_cycle, y_test_cycle, time=1, output_path=snakemake.output['pre_cycle_t1'], strategy='uniform')
plot_figure(post_cycle, y_test_cycle, time=3, output_path=snakemake.output['post_cycle_t3'], strategy='uniform')
plot_figure(post_cycle, y_test_cycle, time=2, output_path=snakemake.output['post_cycle_t2'], strategy='uniform')
plot_figure(post_cycle, y_test_cycle, time=1, output_path=snakemake.output['post_cycle_t1'], strategy='uniform')

plot_figure(pre_cycle, y_test_cycle, time=3, output_path=snakemake.output['pre_cycle_t3q'], strategy='quantile')
plot_figure(pre_cycle, y_test_cycle, time=2, output_path=snakemake.output['pre_cycle_t2q'], strategy='quantile')
plot_figure(pre_cycle, y_test_cycle, time=1, output_path=snakemake.output['pre_cycle_t1q'], strategy='quantile')
plot_figure(pre_cycle, y_test_cycle, time=4, output_path="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_quantile_t4.png", strategy='quantile')
plot_figure(post_cycle, y_test_cycle, time=3, output_path=snakemake.output['post_cycle_t3q'], strategy='quantile')
plot_figure(post_cycle, y_test_cycle, time=2, output_path=snakemake.output['post_cycle_t2q'], strategy='quantile')
plot_figure(post_cycle, y_test_cycle, time=1, output_path=snakemake.output['post_cycle_t1q'], strategy='quantile')
plot_figure(post_cycle, y_test_cycle, time=4, output_path="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_quantile_t4.png", strategy='quantile')
plot_figure(pre_cycle, y_test_cycle, time=5, output_path="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_quantile_t5.png", strategy='quantile')
plot_figure(post_cycle, y_test_cycle, time=5, output_path="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_quantile_t5.png", strategy='quantile')