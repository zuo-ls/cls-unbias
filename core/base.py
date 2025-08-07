import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List
import weakref
import torch
import numpy as np
import random
import os
import logging
from core.Hooks import BaseHook
from core.util import MetricsRecorder
from tqdm import tqdm

def get_hook(trainer,cls_name):
    """
    get hook from trainer by Hook class name
    """
    for hook in trainer.hooks:
        if hook.__class__.__name__ == cls_name:
            return hook
    return Warning(f"Hook {cls_name} not found in trainer.")

class BaseSimpleTrainer:
    def __init__(self,model,max_epoch,train_loader,valid_loader,test_loader,valid_freq,*args,**kwargs):
        self.hooks = []
        self.model = model
        self.max_epoch = max_epoch
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_freq = valid_freq
        self.current_epoch = 0
        self.current_iter = 0
        self.len_loader = len(train_loader)
        self.train_recorder = MetricsRecorder()
        self.valid_recorder = MetricsRecorder()

    def register_hooks(self, hooks):
        for each_hook in hooks:
            assert isinstance(each_hook, BaseHook)
            each_hook.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)
    def before_train(self):
        for each_hook in self.hooks:
            each_hook.before_train()
    def after_train(self):
        for each_hook in self.hooks:
            each_hook.after_train()
    def before_iter(self):
        for each_hook in self.hooks:
            each_hook.before_iter()
    def after_iter(self):
        for each_hook in self.hooks:
            each_hook.after_iter()
    def after_valid(self):
        for each_hook in self.hooks:
            each_hook.after_valid()
    def before_valid(self):
        for each_hook in self.hooks:
            each_hook.before_valid()
    def run(self):
        self.before_train()
        for epoch in tqdm(range(self.max_epoch)):
            self.current_epoch = epoch
            self._iter_train_loader = iter(self.train_loader)
            for batch in range(self.len_loader):
                self.current_iter = batch + self.current_epoch * self.len_loader
                
                self.before_iter()

                # train one iter
                self.model.train()
                batch = next(self._iter_train_loader)
                self.train_one_iter(batch)
                self.model.eval()
                
                # valid all
                if self.current_iter % self.valid_freq == 0:
                    with torch.no_grad():
                        self.before_valid()
                        self.valid()
                        self.after_valid()
                        # Check for early stopping
                        if hasattr(self, 'early_stop') and self.early_stop:
                            print(f'\nEarly stopping triggered at epoch {self.current_epoch}. Best epoch was {self.best_epoch}')
                            self.after_train()
                            return
                self.after_iter() 
        self.after_train()

    def train_one_iter(self,batch):
        raise NotImplementedError
    
    def valid(self):
        raise NotImplementedError
    
    def test(self):
        pass

