from pathlib import Path


registered_datasets = ["ADNI3", "IXI", "ADNI_1_2_GO"]


paths_cluster = {
    "base_metadata_path": "/data/timeleap-shared/MRIs/metadata",
    "adni3_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI3",
    "ixi_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed_IXI-T1",
    "adni_1_2_go_nifti_dir": "/data/timeleap-shared/MRIs/preprocessed/ADNI_1_2_GO",
}

paths_datacore = {
    "base_metadata_path": "/data/groups/cmn/valeriya.malysheva/lunky/metadata",
    "adni3_nifti_dir": "/data/groups/cmn/valeriya.malysheva/lunky/ADNI3",
    "ixi_nifti_dir": "/data/groups/cmn/valeriya.malysheva/lunky/IXI",  # not in datacore just yet
    "adni_1_2_go_nifti_dir": "/data/groups/cmn/valeriya.malysheva/margo/ADNI_1_2_GO",
}
