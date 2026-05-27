from pathlib import Path


registered_datasets = ["ADNI3", "IXI", "ADNI_1_2_GO"]  # forgot what this is for

# can be used directly in dataloaders
train_ready_registered_datasets = [
    'paper_default',
    'counterfactual_target_age_30',
    'counterfactual_target_age_40',
    'counterfactual_target_age_50',
    'smoothed_counterfactual_target_age_30',
    'smoothed_counterfactual_target_age_40',
    'smoothed_counterfactual_target_age_50'
]

paths_cluster = {
    "base_metadata_path": "/data/timeleap-shared/MRIs/metadata",
    "adni3_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI3",  # preprocessed but not smoothed
    "ixi_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed_IXI-T1",
    "adni_1_2_go_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI_1_2_GO",
    "adni3_smoothed_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI3",
    "adni_1_2_go_smoothed_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI_1_2_GO"
}

paths_datacore = {
    "base_metadata_path": "/data/groups/cmn/valeriya.malysheva/lunky/metadata",
    "adni3_nifti_dir": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/preprocessed/ADNI3_preprocessed",  # preprocessed but not smoothed
    "ixi_nifti_dir": "/data/groups/cmn/valeriya.malysheva/lunky/IXI",  # not in datacore just yet
    "adni_1_2_go_nifti_dir": "/data/groups/cmn/valeriya.malysheva/margo/ADNI_1_2_GO",
    "ADNI3_ctf_3d": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/ctf_3d/ADNI3/",
    "ADNI_1_2_GO_ctf_3d": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/ctf_3d/ADNI_1_2_GO/",
    "adni3_smoothed_dir": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/smoothed/ADNI3",
    "adni_1_2_go_smoothed_dir": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/smoothed/ADNI_1_2_GO",
    "ADNI3_ctf_3d_smoothed_dir": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/smoothed/ctf_3d/ADNI3/",
    "ADNI_1_2_GO_ctf_3d_smoothed_dir": "/data/groups/cmn/valeriya.malysheva/lunky/MRIs/smoothed/ctf_3d/ADNI_1_2_GO/"
}
