import logging
from torchvision import transforms
import torch
from monai.data import DataLoader, ImageDataset
from monai import transforms
from config import RANDOM_STATE
from register import paths_datacore, train_ready_registered_datasets
from pathlib import Path
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedGroupKFold
import warnings

logger = logging.getLogger(__name__)

def read_split_data(val_rate: float = 0.25, test_rate: float = 0.25, dataset='paper_default'):  # TODO: val_rate is not used! test_rate is also not used
    """
    Obtain the paths of the NIFTIs, separate them by AD vs CN, enforce MRIs of a subject are either all in train or val.
    """
    if val_rate != 0.25:
        warnings.warn(
            "val_rate and test_rate are currently not used. "
            "Validation split is fixed by n_splits=4."
        )

    base_metadata_path = Path(paths_datacore['base_metadata_path'])
    adni_metadata = pd.read_csv(base_metadata_path / 'raw' / 'ADNI_full_metadata.csv')

    # Create efficient lookup dictionary from metadata
    id_to_group = pd.Series(
        adni_metadata['Group'].values, 
        index=adni_metadata['Image Data ID']
    ).to_dict()
    id_to_subjectid = pd.Series(
        adni_metadata['Subject'].values, 
        index=adni_metadata['Image Data ID']
    ).to_dict()

    cn_groups = {'CN'}
    ad_groups = {'SMC', 'AD', 'LMCI', 'MCI', 'EMCI'}

    rows = []
    for path in get_paths(dataset=dataset): # is an intertools.chain object
        img_id = get_img_id(path, dataset) 
        group = id_to_group.get(img_id)
        subjectid = id_to_subjectid.get(img_id)

        if group in cn_groups:  # 0=AD, 1=CN
            health_state = 1
        elif group in ad_groups:
            health_state = 0
        else:
            print(f"Unknown group label for image_id={img_id} with group={group}")
            continue
        
        row = {'path':path, 'label':health_state, 'subject':subjectid}
        rows.append(row)

    df = pd.DataFrame(rows)
 
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["subject"],   # IMPORTANT
            y=df['label']
        )
    )
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_images_path = train_df.path.tolist()
    train_images_label = train_df.label.tolist()
    val_images_path = val_df.path.tolist()
    val_images_label = val_df.label.tolist()

    return train_images_path, train_images_label, val_images_path, val_images_label

def get_paths(dataset):
    if dataset == 'paper_default':
        adni3_dir = Path(paths_datacore['adni3_nifti_dir'])
        adni_1_2_go_dir = Path(paths_datacore['adni_1_2_go_nifti_dir'])

        adni3_paths = adni3_dir.glob("*_restore.nii.gz")
        adni_1_2_go_paths = adni_1_2_go_dir.glob("*_restore.nii.gz")
        
        return itertools.chain(adni3_paths, adni_1_2_go_paths)
    elif dataset in ['counterfactual_target_age_30', 'counterfactual_target_age_40', 'counterfactual_target_age_50']:
        adni3_dir = Path(paths_datacore['ADNI3_ctf_3d']) / dataset.removeprefix("counterfactual_")  # Path() / target_age_40
        adni_1_2_go_dir = Path(paths_datacore['ADNI_1_2_GO_ctf_3d']) / dataset.removeprefix("counterfactual_")  # Path() / target_age_40

        adni3_paths = adni3_dir.glob("*.nii.gz")
        adni_1_2_go_paths = adni_1_2_go_dir.glob("*.nii.gz")
        
        return itertools.chain(adni3_paths, adni_1_2_go_paths)       
    else:
        raise NotImplementedError

def get_img_id(path, dataset):
    if dataset == 'paper_default':
        return path.stem.split("_")[0]
    elif dataset in ['counterfactual_target_age_30', 'counterfactual_target_age_40', 'counterfactual_target_age_50']:
        return path.name.replace('.nii.gz', '')
    else:
        raise NotImplementedError


data_transform = {
    "train": transforms.Compose([transforms.EnsureChannelFirst(),                            
                                 transforms.CropForeground(k_divisible=1),
                                 transforms.CenterSpatialCrop(roi_size=(112,121,64)), 
                                 transforms.RandSpatialCrop(roi_size=(80,90,44), max_roi_size=(-1,-1,-1), random_size=True), 
                                 transforms.Resize(spatial_size=(112,112,64)),  # resize
                                 transforms.NormalizeIntensity(),
                                 transforms.ScaleIntensity(),
                                 
                                 transforms.RandFlip(prob=0.5,spatial_axis=0),
                                 transforms.RandFlip(prob=0.5,spatial_axis=1),
                                 transforms.RandFlip(prob=0.5,spatial_axis=2),
                                  transforms.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
#                                  transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                                 transforms.ToTensor()  
                                ]),
    "val": transforms.Compose([transforms.EnsureChannelFirst(), 
                               
                               transforms.CropForeground(k_divisible=1),
                               transforms.CenterSpatialCrop(roi_size=(112,121,64)),
                               #transforms.RandSpatialCrop(roi_size=(112,112,64), max_roi_size=(-1,-1,-1)), 
                               #transforms.CenterSpatialCrop(roi_size=(112,112,64)),
                               transforms.Resize(spatial_size=(112,112,64)),
                               transforms.NormalizeIntensity(),
                               transforms.ScaleIntensity(),
                               #transforms.Resize(spatial_size=(112,112,64)),  # resize
                               transforms.ToTensor()
                              ]),
    "test": transforms.Compose([transforms.EnsureChannelFirst(),  
                                transforms.RepeatChannel(repeats=3),  
                               transforms.CropForeground(k_divisible=1),
                               transforms.Resize(spatial_size=(96,96,96)),  # resize
                               transforms.ToTensor(),
                                #transforms.NormalizeIntensity()
                               ])
}

def get_loaders(batch_size, no_workers, dataset='paper_default'):
    assert dataset in train_ready_registered_datasets, f"Dataset must be one of {train_ready_registered_datasets}"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(dataset=dataset)

    # Instantiate training dataset
    train_dataset = ImageDataset(image_files=train_images_path,
                            labels=train_images_label,
                            transform=data_transform["train"])

    # Instantiate validation dataset
    val_dataset = ImageDataset(image_files=val_images_path,
                            labels=val_images_label,
                            transform=data_transform["val"])

    # # Instantiate test dataset
    # test_dataset = ImageDataset(image_files=test_images_path,
    #                         labels=test_images_label,
    #                         transform=data_transform["test"])


    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=no_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=no_workers)
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          pin_memory=True,
    #                          num_workers=no_workers)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    return train_loader, val_loader, train_num, val_num