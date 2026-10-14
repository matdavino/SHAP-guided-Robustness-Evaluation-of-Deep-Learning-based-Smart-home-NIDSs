from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from torch.utils.data import DataLoader, Dataset
from box import Box
from .incremental_learning import Inc_Learning_Appr
import json
from collections import OrderedDict
import numpy as np

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, freeze_after=-1, balance_new_tasks=False, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.trn_datasets = []
        self.val_datasets = []
        self.freeze_after = freeze_after
        self.balance_new_tasks = balance_new_tasks
        
        if not kwargs.get('is_ml', False):
            self.initial_weights = self.model.get_copy()
        
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--freeze-after', default=-1, type=int, required=False,
                            help='Freeze model except heads after the specified task'
                                 '(-1: normal Incremental Joint Training, no freeze) (default=%(default)s)')
        parser.add_argument('--balance-new-tasks', action='store_true',
                           help='Balancing of new classes samples to match the number'
                            'of samples in memory (default=%(default)s)')
        parser.add_argument('--load-cil-base', action='store_true',
                           help='Load CIL base model to calculate retention metrics (default=%(default)s)')
        parser.add_argument('--cil-model', default='', required=False,
                           help='CIL base model to load (default=%(default)s)')
        parser.add_argument('--last-class-scratch', default=0, required=False,
                           help='Last class (default=%(default)s)')
        parser.add_argument('--ls-factor-scratch', default=-1, required=False, type=float,
                           help='Label Smoothing factor to find base model [Retention Score] (default=%(default)s)')
        parser.add_argument('--ls-factor', default=0, required=False, type=float,
                           help='Label Smoothing factor considered (default=%(default)s)')
        return parser.parse_known_args(args)

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0
    
    def train(self, t, trn_loader, val_loader, do_train=True):
        if not self.is_ml:
            super().train(t, trn_loader, val_loader, do_train=do_train)
        else:
            self.trn_datasets.append(trn_loader.dataset)
            self.model.fit(np.concatenate([dataset.images for dataset in self.trn_datasets], axis=0),
                           np.concatenate([dataset.labels for dataset in self.trn_datasets], axis=0))

    def pre_train_process(self, t, trn_loader, val_loader=None):
        """Runs before training all epochs of the task (before the train session)"""
        # print(self.model)
        # input()
        
        if self.load_cil_base:
            print('Loading base model from cil approach...')
            args_files = glob(f'{self.cil_model}/args*')
            timestamp = 0
            for args_file in args_files:
                with open(args_file) as f:
                    args = Box(json.loads(f.read()))
                    ls = args.get('ls_factor',-1)
                    
                    if (int(args.last_class) == int(self.last_class_scratch))&(float(self.ls_factor_scratch)==float(ls)):
                        timestamp = args_file.split('.')[0].split('-')[-1]
            if timestamp != 0:
                base_model_path = f'{self.cil_model}/models/task1-{timestamp}.ckpt'
                print(f'[INFO] Loading base model from {base_model_path}')
                base_model = torch.load(base_model_path, map_location=torch.device('cpu'))
                base_model2 = OrderedDict()

                for k in base_model.keys():
                    if 'heads.0' not in k and 'heads.1' not in k:
                        base_model2['model.'+ '.'.join(k.split('.')[1:])] = base_model[k]
                base_model2['heads.0.weight'] = self.model.heads[0].weight
                base_model2['heads.0.bias'] = self.model.heads[0].bias
                self.model.load_state_dict(base_model2)                
                self.model.freeze_backbone()
                
                self.model.heads.requires_grad = True
                self.model.trainability_info()
            else:
                print('[Warning] CIL base model not found!')

        last_head_state_dict = self.model.heads[t].state_dict()
        self.initial_weights['heads.%d.weight' % t] = deepcopy(last_head_state_dict['weight'])
        self.initial_weights['heads.%d.bias' % t] = deepcopy(last_head_state_dict['bias'])
        self.model.set_state_dict(self.initial_weights)

        # continue pre-train as usual
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        if t > 0 and self._has_exemplars():
            # if has exemplars, take them
            self.trn_datasets = [self.exemplars_dataset]
            self.val_datasets = []
        
        if t == 0 or not self.balance_new_tasks:
            # if it is not balanced, or it is the first training task, simply append the entire dataset
            self.trn_datasets.append(trn_loader.dataset)
        else:
            # balancing new classes basing on how much samples per class are stored into the memory
            num_exemplars_per_class = self.exemplars_dataset.max_num_exemplars_per_class
            if not num_exemplars_per_class:
                import math
                num_exemplars_per_class = math.ceil(
                    self.exemplars_dataset.max_num_exemplars / sum(self.model.task_cls[:t]))

            new_task_exemplars_dataset = ExemplarsDataset(
                trn_loader.dataset.transform, trn_loader.dataset.class_indices[self.model.task_cls[t - 1]:],
                num_exemplars_per_class=num_exemplars_per_class, exemplar_selection='random',
                is_networking=self.exemplars_dataset.is_networking, base_class_index=sum(self.model.task_cls[:t])
            )
            
            # retrieve random balancing selection of new classes
            new_task_exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
            self.trn_datasets.append(new_task_exemplars_dataset)
        
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        
        # continue training as usual
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader, correction=False):
        """Runs after training all the epochs of the task (after the train session)"""
        if t == 0 and len(self.trn_datasets) == 0:
            # Patch to correct the missing behavior when loading pre-trained base model
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        if -1 < self.freeze_after <= t:
            self.model.freeze_all()
            for head in self.model.heads:
                for param in head.parameters():
                    param.requires_grad = True

    def train_epoch(self, t, trn_loader, e=1, correction=False):
        """Runs a single epoch"""
        if self.freeze_after < 0 or t <= self.freeze_after:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
        
        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            outputs = self.model(images)
            
            loss = self.criterion(t, outputs, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            
    def eval(self, t, val_loader):
        if not self.is_ml:
            """Contains the evaluation code"""
            outputs_tot, targets_tot, features_tot = [], [], []
            with torch.no_grad():
                total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
                self._model.eval()
                for images, targets in val_loader:
                    images, targets = self.format_inputs(images, targets)
                    outputs, features = self.model(images, return_features=True)
                    
                    outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                    features_tot.extend(features.tolist())
                    targets_tot.extend(targets.tolist())
                    loss = self.criterion(t, outputs, targets)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)
            return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot
        else:
            rets = self.model.predict(val_loader.dataset.images)
            true = val_loader.dataset.labels
            assert len(rets) == len(true)
            start_taw_idx = min(true)
            end_taw_idx = max(true) + 1
            preds_tag = np.argmax(rets, axis=1)
            preds_taw = np.argmax(rets[:, start_taw_idx : end_taw_idx], axis=1) + start_taw_idx
            loss = np.nan
            targets_tot = true
            acc_tag = sum([t == p for t, p in zip(true, preds_tag)]) / len(true)
            acc_taw = sum([t == p for t, p in zip(true, preds_taw)]) / len(true)
            outputs_tot = [r.tolist() for r in rets]
            extracted_features = None
            from sklearn.metrics import f1_score
            pc_f1_tag = f1_score(true, preds_tag, average=None)
            true_idx = [i for i, t in enumerate(sorted(set(np.concatenate((true, preds_tag))))) if t in set(true)]
            pc_f1_tag = [pc_f1_tag[i] for i in true_idx]
            print('f1_tag:', np.mean(pc_f1_tag))
            print('f1_taw:', f1_score(true, preds_taw, average='macro'))
            print('per-class f1_tag:', pc_f1_tag)
            print('per-class f1_taw:', f1_score(true, preds_taw, average=None))
            return loss, acc_taw, acc_tag, outputs_tot, targets_tot, extracted_features

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets, label_smoothing=self.ls_factor)

def tensor_to_cpu(tensor):
    if isinstance(tensor, list):
        tensor = [v.cpu() for v in tensor]
    else:
        tensor = tensor.cpu()
    return tensor

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])
        self.transform = None

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
