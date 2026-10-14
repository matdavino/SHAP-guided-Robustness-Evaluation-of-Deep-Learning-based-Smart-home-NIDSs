import math
import os
import random

import numpy as np
import pandas as pd
import socket, struct
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import random
import torch

min_max = {'PL': (-1, 1500),'IAT': (-1, 60000),'DIR': (-1,1), 'LOAD': (-1, 255),
           'WIN': (-1, 65535),'FLG': (-1, 255),'TTL': (-1, 255), 'PORT_CAT': (-1, 2)}


class NetworkingDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all biflows in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        # self.seeds = data['s']
        self.seeds = data.get('s', [])
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        return x, y

    def set_modality(self, modality='all'):
        self.modality = modality

def categorize_port(x):
    x = x.copy()
    x[x < 1024] = 0  # Well-known
    x[(1024 <= x) & (x < 49152)] = 1  # Registered
    x[x >= 49152] = 2  # Dynamic
    return x

def normalize(x, n, field):
    x_n=np.zeros(x.shape)
    min_ = min_max[field][0]
    max_ = min_max[field][1]
    x_n = (x - min_) / (max_ - min_)
    x_n[x_n < 0] = 0
    x_n[x_n > 1] = 1
    return x_n

def get_min_max_values(df):
    print('[INFO] Finding Min-Max value on training set')
    print(f'[INFO] Considered classes: {",".join(np.unique(df["LABEL"].values))}')
    print(f'[INFO] PL: {min_max["PL"]} | IAT: {min_max["IAT"]} | WIN: {min_max["WIN"]} | DIR: {min_max["DIR"]}| PORT_CAT: {min_max["PORT_CAT"]}')

def apply_uniform_binning(column, num_bins):
    """
    Apply uniform binning to a column in a Pandas DataFrame.
    Each cell in the column contains an array of values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to bin.
        num_bins (int): The number of bins to use.

    Returns:
        pd.DataFrame: A DataFrame with the binned column.
    """
    # Convert column to tensors
    tensor_values = [torch.tensor(arr, dtype=torch.float32) for arr in column]
    # Find min and max values across all arrays
    all_values = torch.cat(tensor_values)  # Flatten all arrays
    min_val, max_val = all_values.min(), all_values.max()
    # Define uniform bin edges
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1)  # num_bins + 1 edges
    # Apply binning
    binned_values = [torch.bucketize(arr, bin_edges).tolist() for arr in tensor_values]
    # Convert back to DataFrame
    return binned_values


def get_data(full_path, num_tasks, nc_first_task, nc_incr_tasks, validation, shuffle_classes, class_order=None,
             num_bytes=None, num_pkts=None, fields=None, seed=0, scaling='mms_01', shots=-1, fseed=0, last_class=-1,
             predict_tasks=False, statistical=False, is_ml=False, class_augmentation=False, sample_augmentation=False,
             personalized_scaling=False, min_max_dict=None, is_dataset_incr=False, resample_target=None):
    """Prepare data: dataset splits, task partition, class order"""

    assert num_bytes or num_pkts and len(fields) > 0, (
        'NetworkingDataset requires the definition of num_bytes, or num_pkts and fields, or both.')

    assert scaling in ['mms_01', 'mms_sym1'], (
            'Scaling of features should be one of mms_01 for MinMaxScaler((0,1)), '
            'mms_sym1 for MinMaxScaler((-1,1)), or None to skip. Instead, %s is passed.' % scaling)

    data = {}
    taskcla = []
    pl_enhanced = False
    iat_enhanced = False

    path = ('/' if full_path.startswith('/') else '') + os.path.join(*full_path.split('/')[:-1])
    dataset_filename = full_path.split('/')[-1]
    dataset_extension = dataset_filename.split('.')[-1]
    
    mms_range = discr = None
    if scaling == 'mms_01':
        mms_range = (0, 1)
        discr = ''
    elif scaling == 'mms_sym1':
        print('WARNING: inputs are minmax scaled b.t.w. -1 and 1. Padding is maintained at zero.')
        mms_range = (-1, 1)
        discr = '_sym1'

    prep_df_path = os.path.join(path, dataset_filename.replace(
            '.%s' % dataset_extension, '_prep%d%s.%s' % (seed, discr, dataset_extension)))
            
    all_fields = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'] #,'LOAD']
    
    df = None

    valid_idx = None
    print('PREP_DF', dataset_filename, prep_df_path, os.path.exists(prep_df_path))
    
    if not os.path.exists(prep_df_path) or personalized_scaling:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import LabelEncoder
        
        if df is None:
            df = pd.read_parquet(full_path)

        all_fields = [f for f in all_fields if f in df]
        
        le = LabelEncoder()
        le.fit(df['LABEL'])
        df['ENC_LABEL'] = le.transform(df['LABEL'])
        np.savetxt(os.path.join(path, 'classes.txt'), le.classes_, fmt='%s')

        print('INFO: 0.7 train_ratio is applied.')
        train_idx, test_idx = [], []
        for lbl in df['LABEL'].unique():
            train_idx_tmp, test_idx_tmp = train_test_split(
                df[df['LABEL']==lbl].index.to_numpy(), train_size=.7, random_state=seed) #, stratify=df['ENC_LABEL'])
            train_idx.extend(train_idx_tmp)
            test_idx.extend(test_idx_tmp)

        if not statistical:
            pad = 'FEAT_PAD' in df or 'LOAD_PAD' in df
            for field in all_fields:
                if pad:
                    print('[INFO] Applying padding!')
                    pad_field = 'FEAT_PAD' if field != 'LOAD' else 'LOAD_PAD'
                    pad_value = -1
                    df[field] = df[[field, pad_field]].apply(
                        lambda x: np.concatenate((x[field], [pad_value] * x[pad_field])), axis=1)
                    
                df['SCALED_%s' % field] = df[field].apply(lambda x: normalize(x, num_pkts, field))
            df = df[['SCALED_%s' % field for field in all_fields] + ['ENC_LABEL']]
        else:
            del df['LABEL']
            df = df.rename(columns={'PL': 'SCALED_PL', 'WIN': 'SCALED_WIN', 'IAT': 'SCALED_IAT', 'DIR': 'SCALED_DIR'})
        
        df['IS_TRAIN'] = False
        df.loc[train_idx, 'IS_TRAIN'] = True
        
        if not personalized_scaling:
            df.to_parquet(prep_df_path)
    else:
        print('WARNING: using pre-processed dataframe.')
        df = pd.read_parquet(prep_df_path)
        train_idx, test_idx = df[df['IS_TRAIN']].index, df[~df['IS_TRAIN']].index

    if resample_target is not None:
        from sklearn.preprocessing import LabelEncoder

        with open(os.path.join(path, 'classes.txt')) as fin:
            classes = [v.strip().strip(',') for v in fin.readlines() if v.strip()]

        le = LabelEncoder()
        le.fit(classes)

        resample_target_enc = le.transform(resample_target)

        for i, v in enumerate(resample_target_enc):
            resample_target[i] = v
    
    if class_order is None:
        num_classes = len(df['ENC_LABEL'].unique())
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    if resample_target is not None:
        for i, _ in enumerate(resample_target):
            resample_target[i] = class_order.index(resample_target[i])

    if last_class != -1:
        np.random.seed(seed + last_class)
        class_order = [c for c in class_order if c != last_class]
        np.random.shuffle(class_order)
        class_order.append(last_class)

    print('Class order: ', class_order)
    
    # compute classes per task and num_tasks
    if shots == -1 and not is_dataset_incr:
        if nc_first_task <= 0:
            cpertask = np.array([num_classes // num_tasks] * num_tasks)
            for i in range(num_classes % num_tasks):
                cpertask[i] += 1
        elif nc_incr_tasks > 0:  # When nc_incr_tasks is set, a fixed-sized set of num_tasks episodes is instantiated
            assert nc_first_task + nc_incr_tasks * (num_tasks - 1) <= num_classes, "tasks want more classes than exist"
            cpertask = np.array([nc_first_task] + [nc_incr_tasks] * (num_tasks - 1))
        else:
            assert nc_first_task <= num_classes, "first task wants more classes than exist"
            remaining_classes = num_classes - nc_first_task
            assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
            if num_tasks > 1:
                cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
                for i in range(remaining_classes % (num_tasks - 1)):
                    cpertask[i + 1] += 1
            else:
                cpertask = np.array([nc_first_task])
    elif shots!=-1:
        cpertask = np.array([nc_first_task] + [nc_incr_tasks] * (num_tasks - 1))
        class_order = (class_order[:nc_first_task] + 
            [v for i in range(num_tasks - 2) for v in class_order[nc_first_task + nc_first_task * i:nc_first_task + nc_incr_tasks * (i + 1)]] + 
            class_order[-nc_incr_tasks:])
    else:
        cpertask = np.array([num_classes])

    assert num_classes >= cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    if 'PL_DIR' in fields:
        assert len(fields) == 1, 'Not Implemented Error. Provide a behavior for mixing PL_DIR and other fields.'
        field = fields[0]
        df.loc[:, 'SCALED_%s' % field] = df[['SCALED_PL', 'SCALED_DIR']].apply(
            lambda x: [pl * (dir - .5) * 2 for pl, dir in zip(x['SCALED_PL'], x['SCALED_DIR'])], axis=1)
        
    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': [], 's': []}
        data[tt]['val'] = {'x': [], 'y': [], 's': []}
        data[tt]['tst'] = {'x': [], 'y': [], 's': []}

    # If the few-shot modality is enabled, the dataset is shuffled basing on fseed
    # N.B. this function is applyied w/o impacting the external flow random state
    if shots > -1:
        st0 = np.random.get_state()
        np.random.seed(fseed)
        np.random.shuffle(np.asarray(train_idx))
        np.random.set_state(st0)

    if num_bytes and num_pkts:
        train_zip = zip(df.loc[train_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                        df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                       df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, ['SCALED_LOAD'] + ['SCALED_%s' % field for field in fields]].values,
                       df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
    elif num_bytes:
        train_zip = zip(df.loc[train_idx, 'SCALED_LOAD'], df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, 'SCALED_LOAD'], df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, 'SCALED_LOAD'], df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
    else:
        train_zip = zip(df.loc[train_idx, ['SCALED_%s' % field for field in fields]].values,
                        df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
        test_zip = zip(df.loc[test_idx, ['SCALED_%s' % field for field in fields]].values,
                       df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
        if valid_idx is not None:
            valid_zip = zip(df.loc[valid_idx, ['SCALED_%s' % field for field in fields]].values,
                       df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
        
    nsamples_per_class = df.loc[train_idx, 'ENC_LABEL'].value_counts().to_dict()
    fs_counting_samples = dict([(v, min(shots, nsamples_per_class[v])) for v in class_order[-nc_incr_tasks:]])
    if shots > -1:
        print('Shots per Class:', fs_counting_samples)
    
    # ALL OR TRAIN
    for this_row in train_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
        if this_label not in class_order or fs_counting_samples.get(this_label, -1) == 0:
            continue
        if this_label in fs_counting_samples:
            fs_counting_samples[this_label] -= 1
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
        this_seed = get_random_state(this_ip)

        data[this_task]['trn']['x'].append(this_sample)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        data[this_task]['trn']['s'].append(this_seed)

    # ALL OR TEST
    for this_row in test_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
        this_label = int(this_label)
        
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
        this_seed = get_random_state(this_ip)

        data[this_task]['tst']['x'].append(this_sample)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])
        data[this_task]['tst']['s'].append(this_seed)

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if valid_idx is None and validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.ceil(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])
    else:
        # ALL OR VALID
        for this_row in valid_zip:
            this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
            this_label = int(this_label)
            
            if this_label not in class_order:
                continue
            # If shuffling is false, it won't change the class number
            this_label = class_order.index(this_label)

            # add it to the corresponding split
            this_task = (this_label >= cpertask_cumsum).sum()
            if this_task >= num_tasks:
                continue

            this_sample = format_data(this_biflow, is_ml, num_bytes, num_pkts, fields)
            this_seed = get_random_state(this_ip)

            data[this_task]['val']['x'].append(this_sample)
            data[this_task]['val']['y'].append(this_label - init_class[this_task])
            data[this_task]['val']['s'].append(this_seed)

    if predict_tasks:
        for tt in data.keys():
            data[tt]['tst']['y'] = [0] * len(data[tt]['tst']['y'])
            data[tt]['val']['y'] = [0] * len(data[tt]['val']['y'])
            data[tt]['trn']['y'] = [0] * len(data[tt]['trn']['y'])
            data[tt]['ncla'] = 1

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order, class_order

def load_data(pre_extracted_dataset_dir, seed, shots=-1, fseed=-1):
    scenario = pre_extracted_dataset_dir.strip('/').split('/')[-1]  # the dataset dir is /path/to/discr_cl0_cl1_.../
    new_classes = [int(v) for v in scenario.split('_')[1:]]  # in discr_cl0_cl1_... cl0, cl1, ... are the new episode classes
    class_order = [v for v in range(40) if v not in new_classes] + new_classes
    print(class_order)
    data = {}
    taskcla = []

    seed_dir = os.path.join(pre_extracted_dataset_dir, f'{seed=}')

    task = 0
    task_dir = os.path.join(seed_dir, f'{task=}')

    while os.path.exists(task_dir):
        data[task] = {}

        if task == 0:
            for prt in ['trn', 'val', 'tst']:
                data[task][prt] = dict(np.load(os.path.join(task_dir, f'{prt}.npz')))
        else:
            data[task]['tst'] = dict(np.load(os.path.join(task_dir, 'tst.npz')))
            if shots == -1:
                data[task]['trn'] = dict(np.load(os.path.join(task_dir, 'trn.npz')))

                num_val = 0
                for cl in set(data[task]['trn']['y']):
                    nums = sum([v == cl for v in data[task]['trn']['y']])
                    num_val += int(np.ceil(nums * .1))

                data[task]['val'] = {}
                for k in data[task]['trn']:
                    # The validation set is concatenated with the training set
                    data[task]['val'][k] = data[task]['trn'][k][-num_val:]
                    data[task]['trn'][k] = data[task]['trn'][k][:-num_val]
            else:
                for prt in ['trn', 'val']:
                    prt2 = 'tst' if prt == 'trn' else prt
                    sht_dir = os.path.join(task_dir, f'{prt2}_eps', f'sht={shots}')
                    print(sht_dir)
                    assert os.path.exists(sht_dir)
                    eps_dir = os.path.join(sht_dir, f'eps={fseed}')
                    print(eps_dir)
                    assert os.path.exists(eps_dir)

                    data[task][prt] = dict(np.load(os.path.join(eps_dir, 'support.npz')))
        
        for typ, prp in zip([str, int], ['name', 'ncla']):
            with open(os.path.join(task_dir, prp)) as f:
                data[task][prp] = typ(f.readline().strip())

        task += 1
        task_dir = os.path.join(seed_dir, f'{task=}')

    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))

    with open(os.path.join(seed_dir, 'ncla')) as f:
        data['ncla'] = int(f.readline().strip())

    return data, taskcla, class_order


def format_data(this_biflow, is_ml, num_bytes, num_pkts, fields):
    
    if is_ml:
        this_sample = this_biflow[0]
    elif num_bytes and num_pkts:
        this_sample = (
            this_biflow[0][0][:num_bytes].reshape(1, num_bytes).astype('float32'),
            np.array([tb[:num_pkts] for tb in this_biflow[0][1:]]).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')
        )
    elif num_bytes:
        this_sample = this_biflow[0][:num_bytes].reshape(1, num_bytes).astype('float32')
    else:
        this_sample = np.array([tb[:num_pkts] for tb in this_biflow[0]]).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')

    return this_sample

def get_random_state(idx):
    """
    Compute the random_state from the server IP address 
    """
    try:
        idx_family=socket.AF_INET6 if ':' in idx else socket.AF_INET
        fmt = '!L' if idx_family==socket.AF_INET else '!'+'L'*4
        server_ip = idx.split(',')[2]
    except:
        server_ip = idx
    
    try:
        return sum(struct.unpack(fmt, socket.inet_pton(idx_family, server_ip)))
    except:
        random_32bit_integer = random.randint(0, 2**32 - 1)
        return random_32bit_integer


