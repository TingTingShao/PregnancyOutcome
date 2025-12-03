rule generate_preds_dsa:
    input:
        dsa_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_data.pkl",
    output:
        sda_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_dsa.pkl"
    conda:
        "../envs/pregnancy_env.yaml"
    log:
        "logs/generate_preds_dsa/dsa_preds.log"
    script:
        "../scripts/generate_preds_dsa.py"

rule generate_preds_rsf:
    input:
        rsf_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl",
    output:
        rsf_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_rsf.pkl"
    conda:
        "../envs/pregnancy_env.yaml"
    log:
        "logs/generate_preds_rsf/rsf_preds.log"
    script:
        "../scripts/generate_preds_rsf.py"

rule generate_preds_bart:  
    input:
        bart_test_cycle_pre="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_test_cycle_pre.csv",
    output:
        bart_preds_out=touch("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/bart_data/bart_preds.out"),
    conda:
        "../envs/r_pipeline.yaml"
    log:
        "logs/bart_data/bart_preds.log"
    script:
        "../scripts/BART_analysis.sh"

rule generate_preds_mclernon_refit:
    input:
        mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl",
        mclernon_model_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_model_refit.pkl",
    output:
        mclernon_preds="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/tables/preds_mclernon_refit.pkl"
    conda:
        "../envs/r_pipeline.yaml"
    log:
        "logs/mclernon_data/mclernon_preds_refit.log"
    script: 
        "../scripts/generate_mclernon_preds_refit.py"

rule generate_preds_bart_same_feature_as_ml_refit:  
    output:
        bart_preds_out=touch("/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/bart_data/BART_same_feature_as_ml_refit.out"),
    conda:
        "../envs/r_pipeline.yaml"
    log:
        "logs/bart_data/bart_preds_to_diagnose_ext_performance.log"
    script: 
        "../scripts/BART_to_diagnose_ext_performance.sh"

