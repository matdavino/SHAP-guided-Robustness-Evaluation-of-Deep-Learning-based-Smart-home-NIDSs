import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import math
from datasets.exemplars_dataset import ExemplarsDataset
from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from typing import Iterator

class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None, augment=False,
                 augment_type='', augment_after=1, augment_factor=1, **kwargs):
        self.model = model
        self._model = model
        
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        
        self.augment = augment
        self.augment_after = augment_after
        self.augment_factor = augment_factor

        augment_type = 'a' if augment_type.startswith('a') else augment_type

        # Splitting the augmentation string into augmentation primitives 
        self.augment_types = []
        augment_type_iter = iter(augment_type)
        i = 0
        for v in augment_type_iter:
            if v in ['a', 'd', 'j', 't']:
                self.augment_types.append(v)
            elif v in ['s', 'h']:
                if i < len(augment_type):
                    if augment_type[i+1].isnumeric():
                        while augment_type[i+1].isnumeric():
                            t = next(augment_type_iter)
                            self.augment_types.append(v+t)
                            i += 1
                    else:
                        t = next(augment_type_iter)
                        self.augment_types.append(v+t)
                        i += 1
                else:
                    self.augment_types.append(v)
            i += 1

        
        self.optimizer = None
        self.bias_layer = None
        self.is_train = True
        self.bias_layer = None # BiasLayer(self.device).to(self.device) if 'bias_layer' is in kwargs.keys() else None
        self.__dict__.update(kwargs)
        
        if not self.is_ml:
            self.initial_weights = self._model.get_copy()

            
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--bias-layer', action='store_true', required=False,
                    help='Add bias layer correction')
        parser.add_argument('--augment', action='store_true',
                            help='Apply new samples\' augmentation based on transforms (default=%(default)s)')
        parser.add_argument('--augment-types', default='a', type=str,
                            help='Type of augmentation: (a)ll, (d)uplicate, (j)itter, (t)raslate, (s)lice,'\
                                ' and features_(h)ide. N.B. "slice" supports slicing the first half ("s1") or the'\
                                ' second half ("s2") of biflows, or both ("sa"). Also "features_hiding" supports'\
                                ' the hiding of features as "hn", where "n" is the 0-based feature index.'\
                                ' The type "all" can be only used alone. (default=%(default)s)')
        parser.add_argument('--augment-factor', default=1, type=int, required=False,
                            help='Number of times augmetation is applied (default=%(default)s)')
        parser.add_argument('--augment-after', default=1, type=int, required=False,
                            help='Episode from which the new training set is augmented (default=%(default)s)')

        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None
    
    def refresh_initial_weights(self):
        self.initial_weights = self.model.get_copy()

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        try:
            lr = self.lr if t is None or t > 0 else self.first_lr
        except:
            lr = self.lr
        try:
            momentum = self.momentum if t is None or t > 0 else self.first_momentum
        except:
            momentum = self.momentum
        return torch.optim.SGD(self._model.parameters(), lr=lr, weight_decay=self.wd, momentum=momentum)

    def save_logits(self, t, tst_loader, seed=42, is_train=True):
        print('Saving logits...')
        df_model = pd.DataFrame(columns=['Logits','Targets','is_train'])
        
        for i,dst in enumerate(tst_loader):
            images_tot = []
            model_out = []
            tar_list = []
            features_out = []
            for images, targets in dst:
                images, targets = format_inputs(self.device, images, targets)
                with torch.no_grad():
                    outputs, features = self.model(images, return_features=True)
                    images_tot.extend(np.array(tensor_to_cpu(images)).tolist())
                    model_out.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                    tar_list.extend(targets.tolist())
                    features_out.extend(np.array(tensor_to_cpu(features)).tolist())
            # is_train = True if i == (len(tst_loader) - 1) and self.exemplars_dataset is not None else False
            df_model = df_model._append(pd.DataFrame({'Inputs': [images_tot],
                                                     'Logits': [model_out], 
                                                     'Features': [features_out], 
                                                     'Targets': [tar_list], 
                                                     'is_train':is_train}), ignore_index=True)
        is_train_name = 'train' if is_train else 'test'
        self.logger.log_parquet(df_model, name=f'logits_features_targets_{is_train_name}_' + str(seed), task=t)

    def train(self, t, trn_loader, val_loader, do_train=True):
        self.weights = dict()
        for k, l in self._model.state_dict().items():
            self.weights[k] = l.cpu().numpy()
        print('Task ',t)
        """Main train structure"""
        trainclock0 = time.process_time()
        self.pre_train_process(t, trn_loader)
        trainclock1 = time.process_time()
        print('*'*108)
        print(t)
        for k, p in self._model.named_parameters():
            print(np.float32(torch.sum(p).cpu().detach().numpy()), end=' ')
            print()
        # input()
        print('*'*108)
            
        if t > 0 or do_train:
            trn_loader = self.train_loop(t, trn_loader, val_loader)
            
        else:
            print('SKIPPING TRAINING')
        trainclock2 = time.process_time()
        self.post_train_process(t, trn_loader, val_loader)
        trainclock3 = time.process_time()
        print('| Train Time: pre_train={:5.3f}s, train={:5.3f}s, post_train={:5.3f}s |'.format(
            trainclock1 - trainclock0, trainclock2 - trainclock1, trainclock3 - trainclock2))

        for k, p in self._model.named_parameters():
            print(np.float32(torch.sum(p).cpu().detach().numpy()), end=' ')
            print()
        # input()
        print('\nUnchanged')
        for k, l in self._model.state_dict().items():
            try:
                print(k, (self.weights[k] == l.cpu().numpy()).all())
            except:
                print('WARNING: key %k')
        print()



    def compute_topk_similar_classes(self, t, trn_loader, k=3):
        # Inferring per-class similarity metrics of new classes w.r.t. the already known by the model
        with torch.no_grad():
            if t > 0:
                self._model.eval()
                preds = {}
                for images, targets in trn_loader:
                    images, targets = self.format_inputs(images, targets)
                    outputs = self._model(images)
                    outputs = np.concatenate(outputs[:t], axis=1)
                    for i, target in enumerate(targets):
                        preds.setdefault(int(target.numpy()), []).append(
                            outputs[i].argmax())
                self.topk_similar_classes = {}
                for new_class in preds:
                    pred_base_classes, hits = np.unique(
                        preds[new_class], return_counts=True)
                    self.topk_similar_classes[new_class] = pred_base_classes[hits.argsort(
                    )[-k:]][::-1].tolist() + [new_class]
                    
    def pre_train_process(self, t, trn_loader, val_loader=None):
        
        if self.warmup_epochs and t > 0:
            print("pre process warm-up")
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.process_time()
                self._model.heads[-1].train()
                for images, targets in trn_loader:
                    images, targets = self.format_inputs(images, targets)
                    outputs = self._model(images)
                    loss = self.warmup_loss(outputs[t], targets - self._model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.process_time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self._model.eval()
                    for images, targets in trn_loader:
                        images, targets = self.format_inputs(images, targets)
                        outputs = self._model(images)
                        loss = self.warmup_loss(outputs[t], targets - self._model.task_offset[t])
                        pred = torch.zeros_like(targets)
                        for m in range(len(pred)):
                            this_task = (self._model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self._model.task_offset[this_task]
                        hits_taw = (pred == targets).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.process_time()
                print('| Warm-up Epoch {:3d}, time={:5.3f}s/{:5.3f}s | Train: loss={:.3f}, TAw acc={:5.3f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def refresh_initial_weights(self):
        self.initial_weights = self.model.get_copy()

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        self.is_train = True
        try:
            lr = self.lr if t > 0 else self.first_lr
        except:
            lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer(t)

        if self.bias_layer is not None:
            self.bias_layer.optimizer = self.optimizer
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.process_time()

            self.train_epoch(t, trn_loader)
            clock1 = time.process_time()
            
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.process_time()
                print('| Epoch {:3d}, time={:5.3f}s/{:5.3f}s | Train: loss={:.3f}, TAw acc={:5.3f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.3f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

			# Valid
            clock3 = time.process_time()
            valid_loss, valid_acc, _, _, _, _ = self.eval(t, val_loader)
            clock4 = time.process_time()
            print(' Valid: time={:5.3f}s loss={:.3f}, TAw acc={:5.3f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")
        
            # # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                print()
                
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    try:
                        lr /= (self.lr_factor if t > 0 else self.first_lr_factor)
                    except:
                        lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    try:
                        if lr < (self.lr_min if t > 0 else self.first_lr_min):
                            print()
                            break
                    except:
                        if lr < self.lr_min:
                            print()
                            break
                    # reset patience and recover best model so far to continue training
                    
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            
                    
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            
        self.model.set_state_dict(best_model)
        self.is_train = False
        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader=None, correction=True):
        """Runs after training all the epochs of the task (after the train session)"""
        # TODO: if required
        if len(self.exemplars_dataset) > 0:
            
            trn_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
            if self.bias_layer is not None:
                self.bias_layer.handle_head(self.model.task_cls)
                self.bias_layer.train_loop(t, trn_loader, self.device, self.model)

    def train_epoch(self, t, trn_loader, epoch=-1, correction=True):
        """Runs a single epoch"""
        self._model.train()
        if self.fix_bn and t > 0:
            self._model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            loss = self.criterion(t, outputs, targets, features)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def episode_eval(self, t, val_loader):
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            self.model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)
                outputs, features = self.model(images, return_features=True)

                if not self.is_train and self.bias_layer:
                  outputs = self.bias_layer(torch.cat(outputs, dim=1))
                
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num
    
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self._model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)
                outputs, features = self.model(images, return_features=True)

                if not self.is_train and self.bias_layer:
                  outputs = self.bias_layer(torch.cat(outputs, dim=1))
                
                outputs_tot.extend(np.concatenate(outputs, axis=1).tolist())
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

    def calculate_metrics(self, outputs, targets):
        targets = self.format_inputs(targets)
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self._model.task_cls.cumsum(0) <= tensor_to_cpu(targets[m])).sum()
            pred[m] = outputs[this_task][m].argmax() + self._model.task_offset[this_task]
        hits_taw = (pred == targets).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets=None, features=None):
        """Returns the loss value"""
        # print("CRITERION INC LEARNING")
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def format_inputs(self, images, targets=None):
        return format_inputs(self.device, images, targets)

    def compute_topk_similar_classes(self, t, trn_loader, k=3):
        # Inferring per-class similarity metrics of new classes w.r.t. the already known by the model
        with torch.no_grad():
            if t > 0:
                self._model.eval()
                preds = {}
                for images, targets in trn_loader:
                    images, targets = self.format_inputs(images, targets)
                    outputs = self._model(images)
                    outputs = np.concatenate(outputs[:t], axis=1)
                    for i, target in enumerate(targets):
                        preds.setdefault(int(target.numpy()), []).append(
                            outputs[i].argmax())
                self.topk_similar_classes = {}
                for new_class in preds:
                    pred_base_classes, hits = np.unique(
                        preds[new_class], return_counts=True)
                    self.topk_similar_classes[new_class] = pred_base_classes[hits.argsort(
                    )[-k:]][::-1].tolist() + [new_class]

def format_inputs(device, images, targets=None):
    if isinstance(images, list):
        images = [v.to(device) for v in images]
    else:
        images = images.to(device)
    if targets is not None:
        targets = targets.to(device)
        return images, targets
    return images


def tensor_to_cpu(tensor):
    if isinstance(tensor, list):
        tensor = [v.cpu() for v in tensor]
    else:
        tensor = tensor.cpu()
    return tensor

class ExperienceReplaySampler(Sampler):
    """
    Class implementing the Experience Replay (ER) mini batching proposed in https://tajanthan.github.io/il/docs/cler.pdf
    """

    def __init__(self, batch_size, kind='memory', generator=None, mem_samples_index=None, new_samples_index=None,
                 new_weight=.8, labels=None, repeat_minority=False, max_num_batches=50) -> None:
        """
        int kind: 'memory', stratification based on memory (old) and new classes; 'class', per-class stratification.
        float new_weight, default is 0.8 like in SS-IL with a mem~new proportion of 1~4
            N.B.
                the 'memory' is enforced with shuffle=True;
                the 'class' modify the batch size accordingly with the total number of classes and force shuffle=True.
        """

        self.kind = kind
        self.generator = generator

        if self.kind == 'memory':
            assert mem_samples_index is not None and new_samples_index is not None
            self.new_batch_size = int(batch_size * new_weight)
            self.mem_batch_size = batch_size - self.new_batch_size
            self.mem_samples_index = mem_samples_index
            self.new_samples_index = new_samples_index

            mem_num_batches = len(self.mem_samples_index) // self.mem_batch_size
            new_num_batches = len(self.new_samples_index) // self.new_batch_size

            # Setting the number of batches as the minimum number of mini batches for memory and new samples
            if repeat_minority:
                self.num_batches = min(max(mem_num_batches, new_num_batches), max_num_batches)
                self.repeat_factors = [math.ceil(self.num_batches / mem_num_batches), math.ceil(self.num_batches / new_num_batches)]
            else:
                self.num_batches = min(mem_num_batches, new_num_batches)
                self.repeat_factors = [1, 1]
        elif self.kind == 'class':
            assert labels is not None

            labels_uniq = set(labels)
            self.num_classes = len(labels_uniq)
            self.samples_per_class = math.ceil(batch_size / self.num_classes)
            batch_size = self.samples_per_class * self.num_classes

            print(f'INFO: new {batch_size=}, resulting in {self.samples_per_class=}.')
            
            self.classwise_samples_index = []
            for label in sorted(labels_uniq):
                self.classwise_samples_index.append(
                    np.arange(len(labels))[np.where(labels == label)])

            classwise_num_batches = [len(csi) // self.samples_per_class for csi in self.classwise_samples_index]

            # Setting the number of batches as the minimum number of mini batches accordingly with the less represeted class
            if repeat_minority:
                self.num_batches = min(max(classwise_num_batches), max_num_batches)
                self.repeat_factors = [math.ceil(self.num_batches / cnb) for cnb in classwise_num_batches]
            else:
                self.num_batches = min(classwise_num_batches)
                self.repeat_factors = [1] * len(classwise_num_batches)
        else:
            raise NotImplementedError(f'{kind=} not supported.')

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.kind == 'memory':
            # Shuffling the memory and new samples indexes to obtain shuffle=True
            self.mem_samples_index = self.mem_samples_index[
                torch.randperm(len(self.mem_samples_index), generator=generator)]
            self.new_samples_index = self.new_samples_index[
                torch.randperm(len(self.new_samples_index), generator=generator)]
            
            _mem_samples_index = np.concatenate(np.repeat([self.mem_samples_index], self.repeat_factors[0], axis=0))
            _new_samples_index = np.concatenate(np.repeat([self.new_samples_index], self.repeat_factors[1], axis=0))

            for i in range(self.num_batches):
                yield from _mem_samples_index[i * self.mem_batch_size:(i + 1) * self.mem_batch_size]
                yield from _new_samples_index[i * self.new_batch_size:(i + 1) * self.new_batch_size]
        elif self.kind == 'class':
            # Shuffling the memory and new samples indexes to obtain shuffle=True
            for i in range(self.num_classes):
                self.classwise_samples_index[i] = self.classwise_samples_index[i][
                    torch.randperm(
                        len(self.classwise_samples_index[i]), generator=generator)]
            
            _classwise_samples_index = [
                np.concatenate(np.repeat([csi], rf, axis=0)) for csi, rf in zip(self.classwise_samples_index, self.repeat_factors)]

            for i in range(self.num_batches):
                for csi in _classwise_samples_index:
                    yield from csi[i * self.samples_per_class:(i + 1) * self.samples_per_class]
