import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _add_errorbars_for_bars(ax, y, lo, hi):
    """Add vertical error bars aligned to the bars in `ax`."""
    # assume exactly one container from the last barplot call
    container = ax.containers[0]
    bars = container.patches
    xs = np.array([b.get_x() + b.get_width()/2 for b in bars], dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    yerr = np.vstack([y - lo, hi - y])
    ax.errorbar(xs, y, yerr=yerr, fmt='none', ecolor='k', elinewidth=1, capsize=4, zorder=3)

def plot_performance(df, metric='brier', title_suffix='Cycle, Posttreatment', path=None):
    metric = (metric or '').lower()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if metric == 'brier':
        # left: Brier(t)
        sns.lineplot(data=df, x='time', y='brier_score', hue='model', style='model', ax=axes[0], errorbar=None)
        axes[0].set_title(f'Brier(t) — {title_suffix}')
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Brier Score')

        # right: IBS with 95% CI
        dat = (df[['model','ibs','ibs_ci_lo','ibs_ci_hi']]
               .drop_duplicates(subset=['model'])  # one row per model
               .sort_values('model'))
        sns.barplot(data=dat, x='model', y='ibs', ax=axes[1], errorbar=None)
        _add_errorbars_for_bars(axes[1],
                                y=dat['ibs'].to_numpy(dtype=float),
                                lo=dat['ibs_ci_lo'].to_numpy(dtype=float),
                                hi=dat['ibs_ci_hi'].to_numpy(dtype=float))
        axes[1].bar_label(axes[1].containers[0], fmt='%.2f', fontsize=10)
        axes[1].set_title(f'IBS — {title_suffix}')
        axes[1].grid(axis='y', alpha=0.25)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('IBS')

    elif metric == 'scaled':
        # left: Scaled Brier(t)
        sns.lineplot(data=df, x='time', y='scaled_brier_score', hue='model', style='model', ax=axes[0], errorbar=None)
        axes[0].set_title(f'Scaled Brier(t) — {title_suffix}')
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Scaled Brier Score')

        # right: scaled IBS with CI
        dat = (df[['model','scaled_ibs','scaled_ibs_ci_lo','scaled_ibs_ci_hi']]
               .drop_duplicates(subset=['model'])
               .sort_values('model'))
        sns.barplot(data=dat, x='model', y='scaled_ibs', ax=axes[1], errorbar=None)
        _add_errorbars_for_bars(axes[1],
                                y=dat['scaled_ibs'].to_numpy(dtype=float),
                                lo=dat['scaled_ibs_ci_lo'].to_numpy(dtype=float),
                                hi=dat['scaled_ibs_ci_hi'].to_numpy(dtype=float))
        axes[1].bar_label(axes[1].containers[0], fmt='%.2f', fontsize=10)
        axes[1].set_title(f'Scaled IBS — {title_suffix}')
        axes[1].grid(axis='y', alpha=0.25)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Scaled IBS')

    else:  # 'auc'
        # left: AUC(t)
        sns.lineplot(data=df, x='time', y='auc', hue='model', style='model', ax=axes[0], errorbar=None)
        axes[0].set_title(f'AUC(t) — {title_suffix}')
        axes[0].grid(alpha=0.25)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('AUC')

        # right: mean AUC with CI
        dat = (df[['model','mean_auc','mean_auc_ci_lo','mean_auc_ci_hi']]
               .drop_duplicates(subset=['model'])
               .sort_values('model'))
        sns.barplot(data=dat, x='model', y='mean_auc', ax=axes[1], errorbar=None)
        _add_errorbars_for_bars(axes[1],
                                y=dat['mean_auc'].to_numpy(dtype=float),
                                lo=dat['mean_auc_ci_lo'].to_numpy(dtype=float),
                                hi=dat['mean_auc_ci_hi'].to_numpy(dtype=float))
        axes[1].bar_label(axes[1].containers[0], fmt='%.2f', fontsize=10)
        axes[1].set_title(f'Mean AUC — {title_suffix}')
        axes[1].grid(axis='y', alpha=0.25)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Mean AUC')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=800)
    return fig, axes


