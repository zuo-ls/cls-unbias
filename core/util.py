import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import numpy as np
import random
import os
import logging
import matplotlib.pyplot as plt
import itertools as it

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_logger(log_pth,log_file_name):
    # remove previous logger if exists
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.exists(log_pth):
        os.makedirs(log_pth)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_pth,log_file_name))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def plt_clear():
    plt.clf()
    plt.cla()
    plt.close()

def plot_dict(path=None,caption='Title',subplot_size=3,caption_size=10,**metrics_dict):
    n_plots = len(metrics_dict)
    f,axs = plt.subplots(1,n_plots)
    f.set_size_inches(subplot_size*n_plots,subplot_size)
    f.suptitle(caption, fontsize=caption_size)
    axs = (axs,) if n_plots==1 else axs
    for each_ax,each_k in zip(axs,metrics_dict):
        each_ax.plot(metrics_dict[each_k])
        each_ax.set_title(each_k)
    plt.tight_layout()
    if path is not None:
        f_name = caption[:-1] if caption.endswith('.') else caption
        plt.savefig(os.path.join(path,f_name+'.pdf'))
        plt.savefig(os.path.join(path,f_name+'.png'))
    plt_clear()

class MetricsRecorder:
    """
    Metrics recorder. 

    Two kinds of recording methods:
    1) autometicly average the metrics in metrics_dict and save in summary_metrics_dict;
    2) manually record and save in manual_summary_metrics_dict;

    See required params in register() function.
    """
    def __init__(self,):
        # batch dict
        self.metrics_dict = {}
        # epoch dict
        self.summary_metrics_dict = {}
        # metrics dict for manually record (summary)
        self.manual_summary_metrics_dict = {}
        # metrics dict for manually record (batch)
        self.manual_metrics_dict = {}
    
    # Registering metrics is required before recording 
    def register(self,names=None,manual_summary_metrics=None,manual_metrics=None):
        """
        register lists are:
        - names: list of metrics names. automatically average and save in summary_metrics_dict.
        - manual_summary_metrics: list of manual summary metrics names. manually record and save in manual_summary_metrics_dict.
        - manual_metrics: list of manual metrics names. manually record each step and save in manual_metrics_dict. could be used for later analysis and later saving in manual_summary_metrics_dict.
        """
        if names is not None:
            assert isinstance(names,list)
            for each_name in names:
                self.metrics_dict[each_name] = []
                self.summary_metrics_dict[each_name] = []
        if manual_summary_metrics is not None:
            for each_manual in manual_summary_metrics:
                self.manual_summary_metrics_dict[each_manual] = []
        if manual_metrics is not None:
            for each_manual in manual_metrics:
                self.manual_metrics_dict[each_manual] = []

    @torch.no_grad()
    def log(self,k,v):
        """
        log to metrics_dict
        log metrics values of a single batch
        """
        self.metrics_dict[k]+=[v]
    
    @torch.no_grad()
    def log_manual(self,k,v):
        """
        log to manual_metrics_dict
        log manual metrics values of a single batch
        """
        self.manual_metrics_dict[k]+=[v]
    
    @torch.no_grad()
    def log_manual_summary_metric(self,k,v):
        """
        log to manual_summary_metrics_dict
        """
        self.manual_summary_metrics_dict[k]+=[v]
    
    # clear batch dict
    def _clear(self,*names,do_clear_all=True):
        if do_clear_all:
            for each_name in self.metrics_dict.keys():
                self.metrics_dict[each_name] = []
        for each_name in names:
            self.metrics_dict[each_name] = []
    
    # clear manual batch dict
    def clear_manual(self,*names,do_clear_all=False):
        if do_clear_all:
            for each_name in self.manual_metrics_dict.keys():
                self.manual_metrics_dict[each_name] = []
        elif set(names).issubset(self.manual_metrics_dict):
            for each_name in names:
                self.manual_metrics_dict[each_name] = []
        else:
            raise Exception('Specified metric(s) is not in the registered manual_metrics_dict! cannot be cleared! ')

    
    @torch.no_grad()
    def calculate_metrics(self):
        # update epoch dict based on batch dict
        for each_name in self.metrics_dict.keys():
            self.summary_metrics_dict[each_name].append(np.mean(self.metrics_dict[each_name]))
        # clear batch dict
        self._clear(do_clear_all=True)
    
    def plot_summary_metrics(
        self,
        path = None,
        caption='Plots of summary metrics.',
        subplot_size=3,
        caption_size=10,
        ):
        plot_dict(
            **self.summary_metrics_dict,
            **self.manual_summary_metrics_dict,
            path=path,
            caption=caption,
            subplot_size=subplot_size,
            caption_size=caption_size,
        )

class Config_feeder:
    """
    grid search.
    Usage:
    hparams = dict(
    bs=[32,64],
    lr=[0.001,0.0001],
    epoch=[10,20],)

    con_feeder = Config_feeder()
    con_feeder.register_config(**hparams)
    for each in con_feeder.config_generator:
        print(each)
    """
    def __init__(self,):
        self.config = dict()
    def register_config(self, **kwargs):
        self.config.update(kwargs)
    @property
    def config_generator(self):
        keys = list(self.config.keys())
        values = list(self.config.values())
        for each_option in it.product(*values):
            yield dict(zip(keys,each_option))
    def __len__(self):
        return np.prod([len(each) for each in self.config.values()])