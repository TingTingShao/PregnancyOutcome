from functions.plot import plot_performance
import functions.utils as utils
utils=utils.Utils()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

metrics_pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle.csv"
metrics_post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle.csv"
metrics_pre_transfer="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_transfer.csv"
metrics_post_transfer="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_transfer.csv"
metrics_pre_cycle_aft="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_bart_aft_feature_selection.csv"
metrics_post_cycle_aft="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_bart_aft_feature_selection.csv"
metrics_pre_transfer_aft="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_transfer_aft_featrue_selection.csv"
metrics_post_transfer_aft="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_transfer_aft_feature_selection.csv"
metrics_pre_cycle_mclernon_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_mclernon_refit.csv"
metrics_post_cycle_mclernon_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_mclernon_refit.csv"

metrics_pre_cycle_reduced="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_bart_reduced.csv"
metrics_post_cycle_reduced="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_bart_reduced.csv"
metrics_pre_cycle_reduced5="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_bart_reduced5.csv"
metrics_post_cycle_reduced5="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_bart_reduced5.csv"

metrics_pre_cycle=utils.load_data(metrics_pre_cycle)
metrics_post_cycle=utils.load_data(metrics_post_cycle)
metrics_pre_transfer=utils.load_data(metrics_pre_transfer)
metrics_post_transfer=utils.load_data(metrics_post_transfer)

metrics_pre_cycle_aft=utils.load_data(metrics_pre_cycle_aft)
metrics_post_cycle_aft=utils.load_data(metrics_post_cycle_aft)
metrics_pre_transfer_aft=utils.load_data(metrics_pre_transfer_aft)
metrics_post_transfer_aft=utils.load_data(metrics_post_transfer_aft)

mclernon_pre=utils.load_data(metrics_pre_cycle_mclernon_refit)
mclernon_post=utils.load_data(metrics_post_cycle_mclernon_refit)

metrics_pre_cycle_reduced=utils.load_data(metrics_pre_cycle_reduced)
metrics_post_cycle_reduced=utils.load_data(metrics_post_cycle_reduced)
metrics_pre_cycle_reduced5=utils.load_data(metrics_pre_cycle_reduced5)
metrics_post_cycle_reduced5=utils.load_data(metrics_post_cycle_reduced5)

metrics_pre_cycle = pd.concat(
    [metrics_pre_cycle,
     metrics_pre_cycle_aft.loc[metrics_pre_cycle_aft['model'] == 'BART_sparse']],
    #  mclernon_pre.loc[mclernon_pre['model'] == 'mcLernon_refit'].replace({'mcLernon_refit': 'mcLernon'})],
    ignore_index=True
)
metrics_post_cycle = pd.concat(
    [metrics_post_cycle,
     metrics_post_cycle_aft.loc[metrics_post_cycle_aft['model'] == 'BART_sparse']],
    #  mclernon_post.loc[mclernon_post['model'] == 'mcLernon_refit'].replace({'mcLernon_refit': 'mcLernon'})],
    ignore_index=True
)
metrics_pre_transfer = pd.concat(
    [metrics_pre_transfer,
     metrics_pre_transfer_aft.loc[metrics_pre_transfer_aft['model'] == 'BART_sparse']],
    ignore_index=True
)
metrics_post_transfer = pd.concat(
    [metrics_post_transfer,
     metrics_post_transfer_aft.loc[metrics_post_transfer_aft['model'] == 'BART_sparse']],
    ignore_index=True
)

metrics_pre_reduced=pd.concat(
    [metrics_pre_cycle_reduced, metrics_pre_cycle_reduced5.replace({'tmpval': 'tmpval_noeur', 'extval': 'extval_noeur'})],
    ignore_index=True
)
metrics_post_reduced=pd.concat(
    [metrics_post_cycle_reduced, metrics_post_cycle_reduced5.replace({'tmpval': 'tmpval_noeur', 'extval': 'extval_noeur'})],
    ignore_index=True
)


print("Metrics Cycle Pretreatment")

fig_brier, axes = plot_performance(metrics_pre_cycle, metric='brier', title_suffix='Cycle, Pretreatment')
fig_auc, axes = plot_performance(metrics_pre_cycle, metric='auc', title_suffix='Cycle, Pretreatment')
fig_scaled, axes = plot_performance(metrics_pre_cycle, metric='scaled', title_suffix='Cycle, Pretreatment')


def _save_three(figs, outpath):
    with PdfPages(outpath) as pdf:
        for fig in figs:
            fig.tight_layout()
            pdf.savefig(fig)
    # close to free memory
    for fig in figs:
        plt.close(fig)

# ------------ Cycle: Pretreatment ------------
fig_brier, _ = plot_performance(metrics_pre_cycle,  metric='brier',  title_suffix='Cycle, Pretreatment')
fig_auc,   _ = plot_performance(metrics_pre_cycle,  metric='auc',    title_suffix='Cycle, Pretreatment')
fig_scaled,_ = plot_performance(metrics_pre_cycle,  metric='scaled', title_suffix='Cycle, Pretreatment')
_ = _save_three([fig_auc, fig_brier, fig_scaled], snakemake.output['pre_cycle'])

# ------------ Cycle: Posttreatment ------------
fig_brier, _ = plot_performance(metrics_post_cycle, metric='brier',  title_suffix='Cycle, Posttreatment')
fig_auc,   _ = plot_performance(metrics_post_cycle, metric='auc',    title_suffix='Cycle, Posttreatment')
fig_scaled,_ = plot_performance(metrics_post_cycle, metric='scaled', title_suffix='Cycle, Posttreatment')
_ = _save_three([fig_auc, fig_brier, fig_scaled], snakemake.output['post_cycle'])

# ------------ Transfer: Pretreatment ------------
# (FIXED) use metrics_pre_transfer -> save to pre_transfer.pdf
fig_brier, _ = plot_performance(metrics_pre_transfer,  metric='brier',  title_suffix='Transfer, Pretreatment')
fig_auc,   _ = plot_performance(metrics_pre_transfer,  metric='auc',    title_suffix='Transfer, Pretreatment')
fig_scaled,_ = plot_performance(metrics_pre_transfer,  metric='scaled', title_suffix='Transfer, Pretreatment')
_ = _save_three([fig_auc, fig_brier, fig_scaled], snakemake.output['pre_transfer'])

# ------------ Transfer: Posttreatment ------------
# (FIXED) use metrics_post_transfer -> save to post_transfer.pdf
fig_brier, _ = plot_performance(metrics_post_transfer, metric='brier',  title_suffix='Transfer, Posttreatment')
fig_auc,   _ = plot_performance(metrics_post_transfer, metric='auc',    title_suffix='Transfer, Posttreatment')
fig_scaled,_ = plot_performance(metrics_post_transfer, metric='scaled', title_suffix='Transfer, Posttreatment')
_ = _save_three([fig_auc, fig_brier, fig_scaled], snakemake.output['post_transfer'])


# ------------ Cycle: Pretreatment Reduced ------------
fig_brier, _ = plot_performance(metrics_pre_reduced,  metric='brier',  title_suffix='Cycle, Pretreatment Reduced')
fig_auc,   _ = plot_performance(metrics_pre_reduced,  metric='auc',    title_suffix='Cycle, Pretreatment Reduced')
fig_scaled,_ = plot_performance(metrics_pre_reduced,  metric='scaled', title_suffix='Cycle, Pretreatment Reduced')
_ = _save_three([fig_auc, fig_brier, fig_scaled], "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_reduced.pdf")
# ------------ Cycle: Posttreatment Reduced ------------
fig_brier, _ = plot_performance(metrics_post_reduced, metric='brier',  title_suffix='Cycle, Posttreatment Reduced')
fig_auc,   _ = plot_performance(metrics_post_reduced, metric='auc',    title_suffix='Cycle, Posttreatment Reduced')
fig_scaled,_ = plot_performance(metrics_post_reduced, metric='scaled', title_suffix='Cycle, Posttreatment Reduced')
_ = _save_three([fig_auc, fig_brier, fig_scaled], "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_reduced.pdf")