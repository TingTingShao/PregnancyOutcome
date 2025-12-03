rule get_eval_metrics:
    input:
        preds_dsa="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_dsa.pkl",
        preds_rsf="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_rsf.pkl",
        data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl",
    output:
        metrics_pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle.csv",
        metrics_post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle.csv",
       
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/get_eval_metrics/get_eval_metrics.log",
    message:
        "Get evaluation metrics for BART, DSA and RSF models"
    script:
        "../scripts/get_eval_metrics.py"


rule get_eval_metrics_mclernon_refit:
    input:
        mclernon_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_mclernon_refit.pkl",
        mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl",
    output:
        metrics_pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_cycle_mclernon_refit.csv",
        metrics_post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_cycle_mclernon_refit.csv",   
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/get_eval_metrics_mclernon_refit/get_eval_metrics_mclernon_refit.log",
    message:
        "Get evaluation metrics for mclernon refit"
    script:
        "../scripts/get_eval_metrics_mclernon_refit.py"

rule get_eval_metrics_bart_same_feature_as_ml_refit:
    input:
        bart_preds_out="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/bart_data/BART_same_feature_as_ml_refit.out",
    output:
        metrics_pre_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_pre_feats_ml_refit.csv",
        metrics_post_cycle="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/metrics_post_feats_ml_refit.csv",   
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/get_eval_metrics_bart_same_feature_as_ml_refit/get_eval_metrics_bart_same_feature_as_ml_refit.log",
    message:
        "Get evaluation metrics for bart to diagnose ext performance"
    script:
        "../scripts/get_eval_metrics_bart_same_features_as_ml_refit.py"
