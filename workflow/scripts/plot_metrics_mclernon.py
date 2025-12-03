from functions.plot import plot_performance
import functions.utils as utils
utils=utils.Utils()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

metrics_pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_mclernon_refit.csv"
metrics_post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_mclernon_refit.csv"



metrics_pre_cycle=utils.load_data(metrics_pre_cycle)
metrics_post_cycle=utils.load_data(metrics_post_cycle)

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

