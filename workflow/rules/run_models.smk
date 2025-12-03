rule run_rsf:
    input:
        prepared_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/prepared_data.pkl",
    output:
        rsf_model="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/rsf_model.pkl",
    message:
        "Run random survival forest model"
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/run_rsf/run_rsf.log"
    script:
        "../scripts/run_rsf.py"


rule run_dsa:
    input:
        sda_data="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_data.pkl",
    output:
        dsa_model="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/dsa_model.pkl",
    message:
        "Run discrete survival analysis model"
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/run_dsa/run_dsa.log"
    script:
        "../scripts/run_dsa.py"


rule run_mclernon_refit:
    input:
        mclernon_data_for_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_data_for_refit.pkl",
    output:
        mclernon_model_refit="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/data/mclernon_model_refit.pkl",
    message:
        "Run Mclernon model refit"
    conda:
        "../envs/pregnancy_env.yaml",
    log:
        "logs/run_mclernon/run_mclernon_refit.log"
    script:
        "../scripts/run_mclernon_refit.py"