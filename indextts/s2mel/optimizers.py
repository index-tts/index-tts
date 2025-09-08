#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def scheduler_state_dict(self):
        state_dicts = [(key, self.schedulers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def load_scheduler_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.schedulers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step_batch(*args) for key in self.keys]

def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])

    return scheduler

def build_optimizer(model_dict, lr, type='AdamW'):
    optim = {}
    for key, model in model_dict.items():
        model_parameters = model.parameters()
        parameters_names = []
        parameters_names.append(
            [
                name_param_pair[0]
                for name_param_pair in model.named_parameters()
            ]
        )
        if type == 'AdamW':
            optim[key] = AdamW(
                model_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=0.1,
            )
        else:
            raise ValueError('Unknown optimizer type: %s' % type)

    schedulers = dict([(key, torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999996))
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim