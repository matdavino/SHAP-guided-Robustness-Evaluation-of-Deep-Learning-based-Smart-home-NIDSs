import os

import numpy as np
import torchvision.transforms as transforms
from sklearn.utils import resample
from torch.utils import data
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.datasets import SVHN as TorchVisionSVHN

from . import base_dataset as basedat
from . import memory_dataset as memd
from . import networking_dataset as netdat
from .dataset_config import dataset_config
from itertools import product
from pprint import pprint

def get_loaders(datasets, num_tasks, nc_first_task, nc_incr_tasks, batch_size, num_workers, pin_memory, validation=.1,
                num_bytes=None, num_pkts=None, fields=None, seed=0, shots=-1, fseed=0, last_class=-1,
                predict_tasks=False, statistical=False, is_ml=False, pre_extracted_dataset_dir=None,
                personalized_scaling=False, min_max_dict=None, is_dataset_incr=False, filter_knw=False, filter_unknwn=False,
                balance_knowledge=False, resample_target=None):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    class_orders=[]
    dataset_offset = 0
    num_tasks = 1 if is_dataset_incr else num_tasks
    
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc.get('resize',None),
                                                      pad=dc.get('pad',None),
                                                      crop=dc.get('crop',None),
                                                      flip=dc.get('flip',None),
                                                      normalize=dc.get('normalize',None),
                                                      extend_channel=dc.get('extend_channel',None))
        class_order = dc.get('class_order', None)
        
        trn_dset, val_dset, tst_dset, curtaskcla, class_order = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                nc_incr_tasks,
                                                                validation=validation,
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=class_order,
                                                                num_bytes=num_bytes, num_pkts=num_pkts, fields=fields,
                                                                seed=seed, shots=shots, fseed=fseed,
                                                                last_class=last_class, predict_tasks=predict_tasks,
                                                                statistical=None, is_ml=is_ml, is_dataset_incr=is_dataset_incr,
                                                                pre_extracted_dataset_dir=pre_extracted_dataset_dir,
                                                                personalized_scaling=personalized_scaling, min_max_dict=min_max_dict,
                                                                resample_target=resample_target)
        
        # apply offsets in case of multiple datasets
        class_orders.append(class_order)
        
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
                
            curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]
                
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))

    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, nc_incr_tasks, validation, trn_transform, tst_transform,
                 class_order=None, num_bytes=None, num_pkts=None, fields=None, seed=0, shots=-1, fseed=0,
                 last_class=-1, predict_tasks=False, statistical=False, is_ml=False, is_dataset_incr=False, pre_extracted_dataset_dir=None,
                 personalized_scaling=False, min_max_dict=None, resample_target=None):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []
    labels_dict=None

    all_data, taskcla, class_indices, class_order = netdat.get_data(
            path, num_tasks=num_tasks, nc_first_task=nc_first_task,
            nc_incr_tasks=nc_incr_tasks, validation=validation,
            shuffle_classes=class_order is None,
            class_order=class_order, num_bytes=num_bytes,
            num_pkts=num_pkts, fields=fields, seed=seed, shots=shots,
            fseed=fseed, last_class=last_class,
            predict_tasks=predict_tasks, statistical=statistical,
            is_ml=is_ml, is_dataset_incr=is_dataset_incr, personalized_scaling=personalized_scaling, min_max_dict=min_max_dict,
            resample_target=resample_target
            ) if pre_extracted_dataset_dir is None or 'none' in pre_extracted_dataset_dir.lower() else \
                netdat.load_data(pre_extracted_dataset_dir, seed, shots, fseed)
        # set dataset type
    Dataset = netdat.NetworkingDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla, class_order

def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)
