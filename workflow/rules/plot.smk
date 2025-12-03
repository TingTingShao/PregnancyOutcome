rule plot_metrics:
    output:
        pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_temp.pdf",
        post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_temp.pdf",
        pre_transfer="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_transfer_temp.pdf",
        post_transfer="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_transfer_temp.pdf",
    conda:
        "../envs/pregnancy_env.yaml"
    message: "Plotting metrics"
    log:
        "logs/plot_metrics/plot_metrics.log"
    script:
        "../scripts/plot_metrics.py"

rule plot_metrics_mclernon:
    output:
        pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_mclernon_refit.pdf",
        post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_mclernon_refit.pdf",
    conda:
        "../envs/pregnancy_env.yaml"
    message: "Plotting metrics"
    log:
        "logs/plot_metrics_mclernon/plot_metrics_mclernon.log"
    script:
        "../scripts/plot_metrics_mclernon.py"

rule plot_mean_calibratoin:
    output:
        figure5="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/submission/Figure5.png",
    conda:
        "../envs/pregnancy_env.yaml"
    message: "Plotting mean calibration"
    log:
        "logs/plot_mean_calibration/plot_mean_calibration.log"
    script:
        "../scripts/plot_mean_calibration.py"

rule plot_bin_calibration:
    output:
        pre_cycle_t3="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_t3.png",
        post_cycle_t3="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_t3.png",
        pre_cycle_t2="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_t2.png",
        pre_cycle_t1="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_t1.png",
        post_cycle_t2="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_t2.png",
        post_cycle_t1="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_t1.png",

        pre_cycle_t3q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_quantile_t3.png",
        post_cycle_t3q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_quantile_t3.png",
        pre_cycle_t2q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_quantile_t2.png",
        pre_cycle_t1q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/pre_cycle_bin_calibration_quantile_t1.png",
        post_cycle_t2q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_quantile_t2.png",
        post_cycle_t1q="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/figures/post_cycle_bin_calibration_quantile_t1.png",
    conda:
        "../envs/pregnancy_env.yaml"
    message: "Plotting bin calibration"
    log:
        "logs/plot_bin_calibration/plot_bin_calibration.log"
    script:
        "../scripts/plot_bin_calibration.py"

