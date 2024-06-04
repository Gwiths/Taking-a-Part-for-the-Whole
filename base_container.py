import os
import torch
import argparse
import yaml
import copy
import sys
import numpy as np
import random
from torch.utils.data import DataLoader
from dataloaders.myDataset import MyDataset
from easydict import EasyDict as edict
from models.getModels import getModel
from utils.utils import load_pretrained_model, load_checkpoint
from utils.loss import LossSelector

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, metavar='CFG', help='config file')

class BaseContainer(object):
    def __init__(self):
        args = parser.parse_args()
        fi = open(args.cfg, 'r')
        args = yaml.load(fi, Loader=yaml.FullLoader)
        self.args = edict(args)
        self.args.training.cuda = not self.args.training.get('no_cuda',False)
        self.args.training.gpus = torch.cuda.device_count()

        torch.backends.cudnn.benchmark = True
        torch.manual_seed(1)

    def init_training_container(self):
        # Define dataset
        self.train_set = MyDataset(self.args.dataset, split='train')
        self.val_set = MyDataset(self.args.dataset, split='val')

        # Define dataloader
        self.train_loader = DataLoader(
            self.train_set,
            batch_size = self.args.training.batch_size,
            worker_init_fn = worker_init_fn_seed,
            shuffle = True,
            drop_last = True,
            num_workers = self.args.training.get('num_workers', 4)
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size = self.args.validation.batch_size,
            worker_init_fn = worker_init_fn_seed,
            num_workers = self.args.training.get('num_workers', 4)
        )

        # Define network
        self.model = getModel(self.args.models)
        self.model = self.model.cuda()

        start_it = 0
        stage = 0

        self.gen_optimizer(self.model.param_groups(), (stage + 1) // 2)
        self.start_epoch = 0

        self.start_it = start_it
        self.stage = stage

        # Define Criterion
        self.criterion = LossSelector(self.args.training)

    def init_testing_container(self):
        self.model = getModel(self.args.models)
        state_dict, _, _, _ = load_checkpoint(checkpoint_path=self.args.testing.trained_model_path)
        load_pretrained_model(self.model, state_dict)
        self.model = self.model.cuda()
        self.test_set = MyDataset(self.args.dataset, split='test', sample_mode='id')
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.args.testing.batch_size,
            worker_init_fn=worker_init_fn_seed,
            num_workers=self.args.training.get('num_workers', 4)
        )

    def gen_optimizer(self, train_params, stage=0):
        opti_args = self.args.training.optimizer
        model_args = self.args.models.Arch

        params = []
        for i in model_args.keys():
            params += train_params[i]

        if opti_args.optim_method == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                momentum = opti_args.get('momentum', 0.0),
                lr = opti_args.lr * opti_args.get('lr_decay', 1) ** stage,
                weight_decay = opti_args.get('weight_decay', 0),
                nesterov = opti_args.get('nesterov', False)
            )
        elif opti_args.optim_method == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = opti_args.lr * opti_args.get('lr_decay', 1) ** stage,
                weight_decay= opti_args.get('weight_decay', 0),
            )
        elif opti_args.optim_method == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr = opti_args.lr * opti_args.get('lr_decay', 1) ** stage,
                weight_decay = opti_args.get('weight_decay', 0),
                betas = opti_args.get('betas', (0.9, 0.999))
            )
        else:
            raise NotImplementedError(
                "optimizer %s not implemented!" % opti_args.optim_method)

    def training(self):
        pass

    def validation(self):
        pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def worker_init_fn_seed(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2**31)
    set_seed(seed)