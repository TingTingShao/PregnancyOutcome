rule rsf_data:
    input:
        postdata="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/data/posttreatment_data.csv",
    output:
        prepared_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl",
    message:
        "--- Preparing data for analysis ---",
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/rsf_data/prepare_data.log",
    script:
        "../scripts/prepare_data.py"

rule dsa_data:
    input:
        inputfile=rules.rsf_data.output.prepared_data,
    output:
        sda_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_data.pkl",
    message:
        "--- Preparing data for DSA analysis ---",
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/prepare_data_for_sda_mclernon/prepare_data_for_sda_mclernon.log",
    script:
        "../scripts/prepare_data_for_sda.py"

rule mclernon_data:
    input:
        inputfile=rules.rsf_data.output.prepared_data,
    output:
        mclernon_data_had="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_had.pkl",
    message:
        "--- Preparing data for McLernonanalysis ---",
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/prepare_data_for_sda_mclernon/prepare_data_for_sda_mclernon.log",
    script:
        "../scripts/prepare_data_for_mclernon.py"

rule bart_data:
    input:
        inputfile=rules.rsf_data.output.prepared_data,
    output:
        bart_train_cycle_pre="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_train_cycle_pre.csv",
        bart_test_cycle_pre="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_test_cycle_pre.csv",
        bart_train_transfer_pre="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_train_transfer_pre.csv",
        bart_test_transfer_pre="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_test_transfer_pre.csv",
        bart_train_cycle_post="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_train_cycle_post.csv",
        bart_test_cycle_post="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_test_cycle_post.csv",
        bart_train_transfer_post="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_train_transfer_post.csv",
        bart_test_transfer_post="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_test_transfer_post.csv",
    message:
        "--- Preparing data for BART analysis ---",
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/bart_data/bart_data.log",
    script:
        "../scripts/prepare_bart_data.py"


rule mclernon_data_for_refit:
    input:
        mclernon_data_had="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_had.pkl",
    output:
        mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl",
    message:
        "---prepare mclernon data for refit"
    log:
        "logs/mclernon_data_for_refit/mclernon_data_for_refit"
    script:
        "../scripts/prepare_mclernon_data_for_refit.py"

rule prepare_data_bart_same_feature_as_mclernon_refit:
    input:
        mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl",
        mclernon_data_had="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_had.pkl",
    output:
        bart_te_post="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/bart_data_te_post_feats_ml_refit.csv",
    message:
        "-----prepare data to diagnose the external validation dataset"
    log:
        "logs/prepare_data_bart_same_feature_as_mclernon_refit/prepare_data_bart_same_feature_as_mclernon_refit.log"
    script:
        "../scripts/prepare_data_bart_same_feature_as_mclernon_refit.py" 
