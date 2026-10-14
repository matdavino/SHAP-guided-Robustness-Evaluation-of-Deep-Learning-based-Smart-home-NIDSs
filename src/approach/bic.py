import time
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr, tensor_to_cpu, ExperienceReplaySampler

class Appr(Inc_Learning_Appr):
    """Class implementing the Bias Correction (BiC) approach described in
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf
    Original code available at https://github.com/wuyuebupt/LargeScaleIncrementalLearning
    """

    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, val_exemplar_percentage=0.1,
                 num_bias_epochs=200, T=2, lamb=-1, knowledge_distillation=None, uniform_new_distillation=False, **kwargs):
        # Sec. 6.1. CIFAR-100: 2,000 exemplars, ImageNet-1000: 20,000 exemplars, Celeb-10000: 50,000 exemplars
        # Sec. 6.2. weight decay for CIFAR-100 is 0.0002, for ImageNet-1000 and Celeb-10000 is 0.0001
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, **kwargs)
        self.val_percentage = val_exemplar_percentage
        self.bias_epochs = num_bias_epochs
        self.model_old = None
        self.T = T
        self.lamb = lamb
        self.knowledge_distillation = getattr(self, knowledge_distillation)
        self.uniform_new_distillation = uniform_new_distillation
        self.bias_layers = []

        self.x_valid_exemplars = []
        self.y_valid_exemplars = []

        self.x_validation_model = []
        self.y_validation_model = []
        self.loss_value = []
        self.loss_weights = []
        self.targets_new_task = []
        self.common_new = 0
        self.uncommon_new = 0
        
        self.trn_stored = []
        self.tst_stored = []

        if self.exemplars_dataset.max_num_exemplars != 0:
            self.num_exemplars = self.exemplars_dataset.max_num_exemplars
        elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
            self.num_exemplars_per_class = self.exemplars_dataset.max_num_exemplars_per_class

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: BiC needs exemplars.'
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 3. "lambda is set to n / (n+m)" where n=num_old_classes and m=num_new_classes - so lambda is not a param
        # To use the original, set lamb=-1, otherwise, we allow to use specific lambda for the distillation loss
        parser.add_argument('--lamb', default=-1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Sec. 6.2. "The temperature scalar T in Eq. 1 is set to 2 by following [13,2]."
        parser.add_argument('--T', default=2, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # Sec. 6.1. "the ratio of train/validation split on the exemplars is 9:1 for CIFAR-100 and ImageNet-1000"
        parser.add_argument('--val-exemplar-percentage', default=0.1, type=float, required=False,
                            help='Percentage of exemplars that will be used for validation (default=%(default)s)')
        # In the original code they define epochs_per_eval=100 and epoch_val_times=2, making a total of 200 bias epochs
        parser.add_argument('--num-bias-epochs', default=200, type=int, required=False,
                            help='Number of epochs for training bias (default=%(default)s)')
        parser.add_argument('--knowledge-distillation', default='gkd', type=str, choices=['gkd', 'tkd', 'logits_matching', 'logits_matching_gkd'],
                            help='Knowledge distillation used (default=%(default)s)')
        parser.add_argument('--uniform-new-distillation', action='store_true', default=False,
                            help='Force the distillation target for new samples as an uniform distribution (default=%(default)s)')
        parser.add_argument('--uniform-validation', action='store_true', default=False,
                            help='Force the validation set to have uniform number of samples for each class (default=%(default)s)')
        parser.add_argument('--ls-factor', default=0, type=float, required=False,
                            help='Value used for Label Smoothing Regularization (default=%(default)s)')
        parser.add_argument('--train-aug',action='store_true', default=False,
                            help='Perform data augumentation tasks (default=%(default)s)')
        parser.add_argument('--val-aug',action='store_true', default=False,
                            help='Perform data augumentation tasks (default=%(default)s)')
        parser.add_argument('--clean-train-data',action='store_true', default=False,
                            help='Cleaning data augmented (default=%(default)s)')
        parser.add_argument('--clean-val-data',action='store_true', default=False,
                            help='Cleaning data augmented (default=%(default)s)')
        parser.add_argument('--augment-correction-dataset',action='store_true', default=False,
                            help='Augmentation of correction dataset (default=%(default)s)')
        parser.add_argument('--clean-corr-data',action='store_true', default=False,
                            help='Cleaning of correction dataset (default=%(default)s)')
        parser.add_argument('--weighted-loss',action='store_true', default=False,
                            help='Weight loss in cross entropy (default=%(default)s)')
        parser.add_argument('--balance-validation',action='store_true', default=False,
                            help='Balance Validation post Augmentation process (default=%(default)s)')
        parser.add_argument('--T-new', default=1, type=int, required=False,
                            help='Temperature scaling for new classes CCE (default=%(default)s)')
        parser.add_argument('--perturb-test', default=-1, type=float, required=False,
                            help='Perturbation factor (default=%(default)s)')
        return parser.parse_known_args(args)

    def bias_forward(self, outputs):
        """Utility function --- inspired by https://github.com/sairin1202/BIC"""
        bic_outputs = []
        for m in range(len(outputs)):
            bic_outputs.append(self.bias_layers[m](outputs[m]))
            # bic_outputs.append(self.bias_layers[m](m, outputs))
        return bic_outputs

    def pre_train_process(self, t, trn_loader, correction=True):
        # add a bias layer for the new classes
        print('PRE TRAIN PROCESS')
        print(f'Correction {correction}')
        self.compute_topk_similar_classes(t, trn_loader, k=3)
        if correction:
            self.bias_layers.append(BiasLayer().to(self.device))

            # STAGE 0: EXEMPLAR MANAGEMENT -- select subset of validation to use in Stage 2 -- val_old, val_new (Fig.2)
            print('Stage 0: Select exemplars from validation')
            clock0 = time.process_time()

            # number of classes and proto samples per class
            num_cls = sum(self.model.task_cls)
            num_old_cls = sum(self.model.task_cls[:t])
            if self.exemplars_dataset.max_num_exemplars != 0:
                num_exemplars_per_class = int(np.floor(self.num_exemplars / num_cls))
                num_val_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
                num_trn_ex_cls = num_exemplars_per_class - num_val_ex_cls
                # Reset max_num_exemplars
                self.exemplars_dataset.max_num_exemplars = (num_trn_ex_cls * num_cls).item()
            elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
                num_val_ex_cls = int(np.ceil(self.val_percentage * self.num_exemplars_per_class))
                num_trn_ex_cls = self.num_exemplars_per_class - num_val_ex_cls
                # Reset max_num_exemplars
                self.exemplars_dataset.max_num_exemplars_per_class = num_trn_ex_cls

            # Remove extra exemplars from previous classes -- val_old
            if t > 0:
                if self.exemplars_dataset.max_num_exemplars != 0:
                    num_exemplars_per_class = int(np.floor(self.num_exemplars / num_old_cls))
                    num_old_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
                    for cls in range(num_old_cls):
                        assert (len(self.y_valid_exemplars[cls]) == num_old_ex_cls)
                        self.x_valid_exemplars[cls] = self.x_valid_exemplars[cls][:num_val_ex_cls]
                        self.y_valid_exemplars[cls] = self.y_valid_exemplars[cls][:num_val_ex_cls]

            # Add new exemplars for current classes -- val_new
            non_selected = []
            for curr_cls in sorted(np.unique(trn_loader.dataset.labels)): #range(num_old_cls, num_cls):
                self.x_valid_exemplars.append([])
                self.y_valid_exemplars.append([])
                if self.augment_correction_dataset:
                    num_val_new_cls = num_val_ex_cls*3 if t>0 else num_val_ex_cls
                else:
                    num_val_new_cls = num_val_ex_cls
                # get all indices from current class
                cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
                # execute_revert_random_state(random.shuffle, dict(x=cls_ind), int(self.ls_factor))
                assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                assert (num_val_new_cls <= len(cls_ind)), "Not enough samples to store for class {:d}".format(curr_cls)
                # add samples to the exemplar list
                self.x_valid_exemplars[curr_cls] = [trn_loader.dataset.images[idx] for idx in cls_ind[:num_val_new_cls]]
                self.y_valid_exemplars[curr_cls] = [trn_loader.dataset.labels[idx] for idx in cls_ind[:num_val_new_cls]]
                non_selected.extend(cls_ind[num_val_new_cls:])
            # remove selected samples from the validation data used during training
            trn_loader.dataset.images = [trn_loader.dataset.images[idx] for idx in non_selected]
            trn_loader.dataset.labels = [trn_loader.dataset.labels[idx] for idx in non_selected]
            clock1 = time.process_time()
            print(' > Selected {:d} validation exemplars, time={:5.3f}s'.format(
                sum([len(elem) for elem in self.y_valid_exemplars]), clock1 - clock0))

            # make copy to keep the type of dataset for Stage 2 -- not efficient
            self.bic_val_dataset = deepcopy(trn_loader.dataset)
            
            #########################################################
            ################ TODO: New Validation ####################
            if self.uniform_validation:
            # STAGE 0: EXEMPLAR MANAGEMENT -- select subset of validation to use in Stage 2 -- val_old, val_new (Fig.2)
                print('Stage 0 bis: Select exemplars from validation')
                clock0 = time.process_time()

                # number of classes and proto samples per class
                num_cls = sum(self.model.task_cls)
                num_old_cls = sum(self.model.task_cls[:t])
                if self.exemplars_dataset.max_num_exemplars != 0:
                    num_exemplars_per_class = int(np.floor(self.num_exemplars / num_cls))
                    num_val_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
                    num_trn_ex_cls = num_exemplars_per_class - 2*num_val_ex_cls
                    # Reset max_num_exemplars
                    self.exemplars_dataset.max_num_exemplars = (num_trn_ex_cls * num_cls).item()
                elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
                    num_val_ex_cls = int(np.ceil(self.val_percentage * self.num_exemplars_per_class))
                    num_trn_ex_cls = self.num_exemplars_per_class - 2*num_val_ex_cls
                    # Reset max_num_exemplars
                    self.exemplars_dataset.max_num_exemplars_per_class = num_trn_ex_cls

                # Remove extra exemplars from previous classes -- val_old
                if t > 0:
                    if self.exemplars_dataset.max_num_exemplars != 0:
                        num_exemplars_per_class = int(np.floor(self.num_exemplars / num_old_cls))
                        num_old_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
                        for cls in range(num_old_cls):
                            assert (len(self.y_validation_model[cls]) == num_old_ex_cls)
                            self.x_validation_model[cls] = self.x_validation_model[cls][:num_val_ex_cls]
                            self.y_validation_model[cls] = self.y_validation_model[cls][:num_val_ex_cls]

                # Add new exemplars for current classes -- val_new
                non_selected = []
                for curr_cls in range(num_old_cls, num_cls):
                    self.x_validation_model.append([])
                    self.y_validation_model.append([])
                    # get all indices from current class
                    cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
                    assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                    assert (num_val_ex_cls <= len(cls_ind)), "Not enough samples to store for class {:d}".format(curr_cls)
                    # add samples to the exemplar list
                    self.x_validation_model[curr_cls] = [trn_loader.dataset.images[idx] for idx in cls_ind[:num_val_ex_cls]]
                    self.y_validation_model[curr_cls] = [trn_loader.dataset.labels[idx] for idx in cls_ind[:num_val_ex_cls]]
                    non_selected.extend(cls_ind[num_val_ex_cls:])
                # remove selected samples from the validation data used during training
                trn_loader.dataset.images = [trn_loader.dataset.images[idx] for idx in non_selected]
                trn_loader.dataset.labels = [trn_loader.dataset.labels[idx] for idx in non_selected]
                clock1 = time.process_time()
                print(' > Selected {:d} validation exemplars to evaluate model training, time={:5.3f}s'.format(
                    sum([len(elem) for elem in self.y_validation_model]), clock1 - clock0))
                self.validation_set = deepcopy(trn_loader.dataset)
         
        # print(f'TASK {t} AFTER CORRECTION: {np.unique(trn_loader.dataset.labels)} | {id(trn_loader)}')
        # Calculate loss weights
        if t>0 and self.weighted_loss:
            self.loss_weights = []
            for cls in range(num_old_cls):
                n_samples = 7*len(np.where(np.asarray(self.exemplars_dataset.labels) == cls)[0])
                self.loss_weights.append(1/n_samples)
            for curr_cls in range(num_old_cls, num_cls):
                n_samples = len(np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0])
                self.loss_weights.append(1/n_samples)
                  
        # if t>0:
        #     super().save_logits(t, [trn_loader, torch.utils.data.DataLoader(self.exemplars_dataset,
        #                                             batch_size=trn_loader.batch_size,
        #                                             shuffle=True,
        #                                             num_workers=trn_loader.num_workers,
        #                                             pin_memory=trn_loader.pin_memory)], seed=3)  
                
        super().pre_train_process(t, trn_loader)

    def define_sampler(self, trn_loader, memory_dataset):
        if self.experience_replay_strategy == 'random':
                # When random, the shuffle should be True
                loader_kwargs = dict(shuffle=True)
        else:
            # Otherwise, the sampler implements its own shuffling strategy
            er_sampler = ExperienceReplaySampler(
                batch_size=trn_loader.batch_size, kind=self.experience_replay_strategy,
                repeat_minority=True, labels=trn_loader.dataset.labels + list(memory_dataset.labels),
                new_samples_index=np.arange(len(trn_loader.dataset)),
                mem_samples_index=np.arange(
                    len(trn_loader.dataset), len(trn_loader.dataset) + len(memory_dataset)),
            )
            loader_kwargs = dict(sampler=er_sampler)
        
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + memory_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    num_workers=0,
                                                    pin_memory=trn_loader.pin_memory,
                                                    **loader_kwargs)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop
        """
        if t > 0:
            
            if self.train_aug:
                new_mem_dataset = deepcopy(self.exemplars_dataset)
                images = deepcopy(new_mem_dataset.images)
                labels = deepcopy(new_mem_dataset.labels)
                
                augmented_images, multiplier = self.augment_data(images, augment_factor=1)
                new_mem_dataset.images = augmented_images
                new_mem_dataset.labels = labels * (multiplier-1)
                
                if self.clean_train_data:
                    cleaning_loader = torch.utils.data.DataLoader(new_mem_dataset,
                                            batch_size=new_mem_dataset.__len__(),
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=trn_loader.pin_memory)
                        
                    correct_memory_samples = []
                    correct_memory_labels = []
                    for img, lbl in cleaning_loader:
                        pred = self.model_old(img)
                        pred = [np.array(x) for x in pred][0]
                        pred = np.argmax(pred, axis=1)
                        index = np.where(pred == np.asarray(lbl))[0]
                        correct_memory_samples.extend(list(np.asarray(img)[index]))
                        correct_memory_labels.extend(list(np.asarray(lbl)[index]))
                    new_mem_dataset.images = tuple(correct_memory_samples)
                    new_mem_dataset.labels = correct_memory_labels
                    print('Correct Samples: ', len(correct_memory_samples))
                    
                new_mem_dataset.images = (list(images) + list(new_mem_dataset.images))
                new_mem_dataset.labels = (list(labels) + list(new_mem_dataset.labels))
                print(f'Augumentatation in Training Set. Starting: {len(self.exemplars_dataset.labels)}. Augumented: {len(new_mem_dataset.labels)}')
                
                # self.define_sampler(trn_loader, new_mem_dataset)
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + new_mem_dataset,
                                                        batch_size=trn_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        pin_memory=trn_loader.pin_memory)
                
            else:
                
                self.targets_new_task = np.unique(trn_loader.dataset.labels)
                self.common_new = len(np.intersect1d(self.targets_new_task, np.unique(self.exemplars_dataset.labels)))
                
                print(f'[INFO] ID: {id(trn_loader)} | New Classes: {self.targets_new_task} | Common Classes: {self.common_new}')
                # input()
                if not self.common_new:
                    print('[WARNING] Training set and memory labels do not overlap')
                    
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                        batch_size=trn_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        pin_memory=trn_loader.pin_memory)
                
            if self.uniform_validation:
                if isinstance(self.validation_set.images, list):
                    self.validation_set.images = sum(self.x_validation_model, [])
                else:
                    self.validation_set.images = np.vstack(self.x_validation_model)
                self.validation_set.labels = sum(self.y_validation_model, [])
                
                validation_set = DataLoader(self.validation_set, batch_size=trn_loader.batch_size, shuffle=True,
                                                num_workers=0, pin_memory=trn_loader.pin_memory)
                if self.val_aug:
                    new_mem_dataset = deepcopy(validation_set.dataset)
                    old_samples = np.where(np.asarray(validation_set.dataset.labels) != 39)[0]
                    images = list(np.asarray(validation_set.dataset.images)[old_samples])
                    labels = list(np.asarray(validation_set.dataset.labels)[old_samples])

                    augmented_images, multiplier = self.augment_data(images, augment_factor=1)
                    new_mem_dataset.images = augmented_images
                    new_mem_dataset.labels = labels * (multiplier-1)
                    if self.clean_val_data:
                        new_mem_dataset.images = augmented_images
                        new_mem_dataset.labels = labels * (multiplier-1)
                        cleaning_loader = torch.utils.data.DataLoader(new_mem_dataset,
                                                batch_size=new_mem_dataset.__len__(),
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=trn_loader.pin_memory)
                            
                        correct_memory_samples = []
                        correct_memory_labels = []
                        for img, lbl in cleaning_loader:
                            pred = self.model_old(img)
                            pred = [np.array(x) for x in pred][0]
                            pred = np.argmax(pred, axis=1)
                            index = np.where(pred == np.asarray(lbl))[0]
                            correct_memory_samples.extend(list(np.asarray(img)[index]))
                            correct_memory_labels.extend(list(np.asarray(lbl)[index]))
                        new_mem_dataset.images = tuple(correct_memory_samples)
                        new_mem_dataset.labels = correct_memory_labels
                        print('Correct Samples: ', len(correct_memory_samples))
                    
                    print(f'Augumentatation in Validation Set. Starting: {len(validation_set.dataset.labels)}. Augumented: {len(new_mem_dataset.labels)}')
                    
                    if self.balance_validation:
                        n_samples_per_class = min(np.unique(new_mem_dataset.labels, return_counts=True)[1])
                        print(f'Number of samples per class to obtain Balanced Validation Set: {n_samples_per_class}')
                        balanced_images_validation_set = []
                        balanced_label_validation_set = []
                        for label in np.unique(new_mem_dataset.labels):
                            index = np.where(np.asarray(new_mem_dataset.labels) == label)[0]
                            samples_per_class = np.asarray(new_mem_dataset.images)[index]
                            balanced_images_validation_set.extend(samples_per_class[:n_samples_per_class])
                            balanced_label_validation_set.extend([label]*n_samples_per_class)
                        
                        new_mem_dataset.images = tuple(balanced_images_validation_set)
                        new_mem_dataset.labels = balanced_label_validation_set
                        val_loader.dataset.images[:min(len(val_loader.dataset.images), n_samples_per_class)]
                        val_loader.dataset.labels[:min(len(val_loader.dataset.labels), n_samples_per_class)]
                        
                    val_loader = torch.utils.data.DataLoader(validation_set.dataset + new_mem_dataset + val_loader.dataset,
                                                        batch_size=val_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        pin_memory=val_loader.pin_memory)
                else:
                    val_loader = torch.utils.data.DataLoader(validation_set.dataset,
                                                            batch_size=len(val_loader.dataset.labels), # TODO: debug
                                                            shuffle=True,
                                                            num_workers=0,
                                                            pin_memory=val_loader.pin_memory)

        print('Stage 1: Training model with distillation')
        super().train_loop(t, trn_loader, val_loader)
        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader, correction=True):
        # From LwF: Restore best and save model for future tasks
        print('POST TRAIN PROCESS')
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        
        if correction:
            # STAGE 2: BIAS CORRECTION
            if t > 0 and len(self.bias_layers)>0:
                print('Stage 2: Training bias correction layers')
                # Fill bic_val_loader with validation protoset
                if isinstance(self.bic_val_dataset.images, list):
                    self.bic_val_dataset.images = sum(self.x_valid_exemplars, [])
                else:
                    self.bic_val_dataset.images = np.vstack(self.x_valid_exemplars)
                self.bic_val_dataset.labels = sum(self.y_valid_exemplars, [])
                bic_val_loader = DataLoader(self.bic_val_dataset, batch_size=trn_loader.batch_size, shuffle=True,
                                            num_workers=0, pin_memory=trn_loader.pin_memory)
                n_tot_samples = len(bic_val_loader.dataset.labels)
                
                if self.augment_correction_dataset:
                    new_mem_dataset = deepcopy(bic_val_loader.dataset)
                    # old_samples = np.where(np.asarray(bic_val_loader.dataset.labels) != 39)[0]
                    images = list(np.asarray(bic_val_loader.dataset.images))
                    labels = list(np.asarray(bic_val_loader.dataset.labels))

                    augmented_images, multiplier = self.augment_data(images, augment_factor=1)
                    new_mem_dataset.images = augmented_images
                    new_mem_dataset.labels = labels * (multiplier-1)
                    
                    if self.clean_corr_data:
                        new_mem_dataset.images = augmented_images
                        new_mem_dataset.labels = labels * (multiplier-1)
                        cleaning_loader = torch.utils.data.DataLoader(new_mem_dataset,
                                                batch_size=new_mem_dataset.__len__(),
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=trn_loader.pin_memory)
                            
                        correct_memory_samples = []
                        correct_memory_labels = []
                        for img, lbl in cleaning_loader:
                            pred = self.model_old(img)
                            pred = [np.array(x) for x in pred][0]
                            pred = np.argmax(pred, axis=1)
                            index = np.where(pred == np.asarray(lbl))[0]
                            correct_memory_samples.extend(list(np.asarray(img)[index]))
                            correct_memory_labels.extend(list(np.asarray(lbl)[index]))
                        new_mem_dataset.images = tuple(correct_memory_samples)
                        new_mem_dataset.labels = correct_memory_labels
                        print('Correct Samples: ', len(correct_memory_samples))
                            
                    n_tot_samples = len(new_mem_dataset.labels) + len(bic_val_loader.dataset.labels)
                    print(f'Number of samples for correction {n_tot_samples}')
                    bic_val_loader = torch.utils.data.DataLoader(new_mem_dataset + bic_val_loader.dataset,
                                                            batch_size=val_loader.batch_size,
                                                            shuffle=True,
                                                            num_workers=val_loader.num_workers,
                                                            pin_memory=val_loader.pin_memory)
                # bias optimization on validation
                self.model.eval()
                # Allow to learn the alpha and beta for the current task
                self.bias_layers[t].alpha.requires_grad = True
                self.bias_layers[t].beta.requires_grad = True

                # In their code is specified that momentum is always 0.9
                bic_optimizer = torch.optim.SGD(self.bias_layers[t].parameters(), lr=self.lr, momentum=0.9)
                # Loop epochs
                for e in range(self.bias_epochs):
                    # Train bias correction layers
                    clock0 = time.process_time()
                    total_loss, total_acc = 0, 0
                    for inputs, targets in bic_val_loader:
                        inputs, targets = self.format_inputs(inputs, targets)
                        # Forward current model
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            old_cls_outs = self.bias_forward(outputs[:t])
                        new_cls_outs = self.bias_layers[t](outputs[t])
                        pred_all_classes = torch.cat([torch.cat(old_cls_outs, dim=1), new_cls_outs], dim=1)
                        # Eqs. 4-5: outputs from previous tasks are not modified (any alpha or beta from those is fixed),
                        #           only alpha and beta from the new task is learned. No temperature scaling used.
                        loss = torch.nn.functional.cross_entropy(pred_all_classes, targets)
                        # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                        loss += 0.1 * ((self.bias_layers[t].beta[0] ** 2) / 2)
                        # Log
                        total_loss += loss.item() * len(targets)
                        total_acc += ((pred_all_classes.argmax(1) == targets).float()).sum().item()
                        # Backward
                        bic_optimizer.zero_grad()
                        loss.backward()
                        bic_optimizer.step()
                    clock1 = time.process_time()
                    # reducing the amount of verbose
                    if (e + 1) % (self.bias_epochs / 4) == 0:
                        print('| Epoch {:3d}, time={:5.3f}s | Train: loss={:.3f}, TAg acc={:5.3f}% |'.format(
                            e + 1, clock1 - clock0, total_loss / n_tot_samples,
                            100 * total_acc / n_tot_samples))
                # Fix alpha and beta after learning them
                self.bias_layers[t].alpha.requires_grad = False
                self.bias_layers[t].beta.requires_grad = False

            # Print all alpha and beta values
            if len(self.bias_layers)>0:
                for task in range(t + 1):
                    print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(
                        task, self.bias_layers[task].alpha.item(), self.bias_layers[task].beta.item()))
            else:
                print('No bias correction layer for this modalities!')

        # STAGE 3: EXEMPLAR MANAGEMENT
        if t < 1:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def get_used_memory(self):
        import psutil
        # Get memory usage of the current process
        process = psutil.Process()
        memory_info = process.memory_info()
                        
        # Return the used memory in bytes
        return memory_info.rss / (1024.0 ** 3)

    def train_epoch(self, t, trn_loader, epoch=-1):
        """Runs a single epoch"""
        if self.uniform_new_distillation:
            first_new_cls_label = sum(self.model.task_cls[:t])
            self.targets_new_task = [x for x in self.targets_new_task if x<first_new_cls_label]
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for i, (images, targets) in enumerate(trn_loader):
            images, targets = self.format_inputs(images, targets)
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images)
                targets_old = self.bias_forward(targets_old) if len(self.bias_layers) > 0 else targets_old # apply bias correction

                if self.uniform_new_distillation:  # Assigning to new samples a target of ones (uniform distribution after softmax)
                    new_index = [x in self.targets_new_task for x in targets] # Distillation solo sulle classi già note aggiunte all'episodio 1
                    if True in new_index:
                        for to in targets_old:
                            to[new_index] = (1-self.ls_factor)*to[new_index] + self.ls_factor/(sum(self.model.task_cls))
            # Forward current model
            outputs = self.bias_forward(self.model(images))
            loss = self.criterion(t, outputs, targets, targets_old, epoch=epoch) 
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        if self.uniform_new_distillation:
            first_new_cls_label = sum(self.model.task_cls[:t])
            self.targets_new_task = [x for x in self.targets_new_task if x<first_new_cls_label]
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            
            if self.perturb_test != -1 and not self.is_train:
                    print(f'[INFO] Perturbing Test Set! Perturbation Factor {self.perturb_test}')
                    new_mem_dataset = deepcopy(val_loader.dataset)
                    augmented_biflows, multiplier = self.augment_data(new_mem_dataset.images, augment_factor=1, features='all', 
                                                                 factor=self.perturb_test, augment_types=['j'])
                    new_mem_dataset.images = augmented_biflows
                    val_loader = torch.utils.data.DataLoader(new_mem_dataset,
                                                        batch_size=val_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        pin_memory=val_loader.pin_memory)
            
            for images, targets in val_loader:            
                images, targets = self.format_inputs(images, targets)
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                    targets_old = self.bias_forward(targets_old) if len(self.bias_layers)>0 else targets_old  # apply bias correction
                    if self.uniform_new_distillation:
                        # new_index = targets >= first_new_cls_label
                        new_index = [x in self.targets_new_task for x in targets] # Distillation solo sulle classi già note aggiunte all'episodio 1
                        if True in new_index:
                            for to in targets_old:
                                # to[new_index] = torch.ones_like(to[new_index])/(sum(self.model.task_cls))
                                to[new_index] = (1-self.ls_factor)*to[new_index] + self.ls_factor/(sum(self.model.task_cls))
                # Forward current model
                outputs, features = self.model(images, return_features=True)
                features_tot.extend(features.tolist())
                outputs = self.bias_forward(outputs) if len(self.bias_layers)>0 else outputs # apply bias correction
                
                outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets, targets_old, train=False)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        outputs /= self.T
        targets /= self.T

        tar = torch.nn.functional.softmax(targets, dim=1)
        return torch.nn.functional.cross_entropy(outputs, tar)

    def gkd(self, outputs_old, outputs, t):
        return self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)

    def tkd(self, outputs_old, outputs, t):
        return sum(self.cross_entropy(outputs[j], outputs_old[j], exp=1.0 / self.T) for j in range(t))
    
    def logits_matching(self, outputs_old, outputs, t):
        return torch.nn.functional.mse_loss(torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1))

    def logits_matching_gkd(self, outputs_old, outputs, t):
        return 0.5*self.logits_matching(outputs_old, outputs, t) + 0.5*self.gkd(outputs_old, outputs, t)

    def criterion(self, t, outputs, targets, targets_old, epoch=-1, train=True):
        """Returns the loss value"""
        loss_dist = 0
        if t > 0:
            loss_dist += float(self.knowledge_distillation(targets_old, outputs, t)) #float(self.knowledge_distillation(targets_old, outputs, t))
    
        cat_outputs = torch.cat(outputs, dim=1)
        
        if t > 0 and self.T_new!=1:
            logits_to_scale = torch.Tensor([False] * len(cat_outputs)).type(torch.bool)
            for new_class in self.topk_similar_classes:
                logits_to_scale |= (targets == new_class)
            cat_outputs[logits_to_scale] /= self.T_new

        if self.weighted_loss and train:
            cce = torch.nn.functional.cross_entropy(cat_outputs, targets, weight=torch.tensor(self.loss_weights))
        else:
            cce = torch.nn.functional.cross_entropy(cat_outputs, targets)

        if self.lamb == -1:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
            return (1.0 - lamb) * cce + lamb * loss_dist
        elif self.lamb == -2:
            lamb = 1 - (self.common_new / self.model.task_cls.sum()).to(self.device)
            return (1.0 - lamb) * cce + lamb * loss_dist
        else:
            return self.lamb*cce +  (1-self.lamb) * loss_dist

class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x):
        return self.alpha * x + self.beta