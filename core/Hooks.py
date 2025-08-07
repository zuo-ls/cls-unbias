import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from core.util import plt_clear
import datetime
import logging

class BaseHook:
    def before_train(self): pass
    def after_train(self): pass
    def before_iter(self): pass
    def after_iter(self): pass
    def before_valid(self): pass
    def after_valid(self): pass

class LogHook(BaseHook):
    """
    log and plots.

    metrics in self.trainer.train_out and self.trainer.valid_out 
    will be logged and plotted automatically.

    This hook will:
    1. create a output folder named by current time
    2. log train metrics every iter
    3. log valid metrics every valid_freq iter
    4. setup logger, log current status to the console and files every valid_freq iter
    5. save plots of the logged metrics to the output folder after training
    """
    def __init__(self,trail_name=None,do_console_output=False,do_file_output=True):     
        self.trial_name = trail_name   
        root_dir = os.path.abspath(os.path.join(__file__,'../../'))
        self.root_dir = os.path.join(root_dir,'output')
        self.create_output_dir()
        os.makedirs(self.out_pth,exist_ok=True)
        self.do_console_output = do_console_output
        self.do_file_output = do_file_output
        self.init_logger() 

    
    def init_logger(self):
        log_path = os.path.join(self.out_pth,'log.txt')
        level = logging.INFO
        format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = logging.Formatter(format)

        logger = logging.getLogger('root')
        logger.handlers.clear()
        logger.setLevel(level)

        # file handler
        if self.do_file_output:
            fh = logging.FileHandler(log_path,mode = 'w')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # console handler
        if self.do_console_output:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        self.logger = logger
    
    def create_output_dir(self):
        now = datetime.datetime.now()
        now_str = now.strftime("%Y_%m_%d_%H_%M")
        if self.trial_name is not None:
            now_str = self.trial_name + '_' + now_str
        self.out_pth = os.path.join(self.root_dir,now_str)

    def after_iter(self):
        # register metrics at the first iter
        if self.trainer.current_iter == 0:
            self.register_metrics()
        
        # log train metrics every iter
        self.log_train_out()
        
        if self.trainer.current_iter % self.trainer.valid_freq == 0:
            self.log_valid_out()
            self.log_current_status()
    
    def log_train_out(self):
        # log metrics every iter
        for each in self.train_record_names:
            self.trainer.train_recorder.log(each,self.trainer.train_out['_'.join(each.split('_')[1:])].item())
        # calculate metrics every valid_freq iter
        if self.trainer.current_iter % self.trainer.valid_freq == 0:
            self.trainer.train_recorder.calculate_metrics()
    
    def log_valid_out(self):
        for each in self.valid_record_names:
            self.trainer.valid_recorder.log_manual_summary_metric(each,self.trainer.valid_out['_'.join(each.split('_')[1:])].item())

    def register_metrics(self,):
        self.train_record_names = self.get_record_names(
                    metrics_dict=self.trainer.train_out,
                    prefix='train',
            )
        self.valid_record_names = self.get_record_names(
                    metrics_dict=self.trainer.valid_out,
                    prefix='valid',
            )
        train_register_kwargs = dict(
            names = self.train_record_names,
            manual_summary_metrics=None,
            manual_metrics=None,
        )
        valid_register_kwargs = dict(
            names = None,
            manual_summary_metrics=self.valid_record_names,
            manual_metrics=None,
        )
        self.trainer.train_recorder.register(**train_register_kwargs)
        self.trainer.valid_recorder.register(**valid_register_kwargs)

    def after_train(self):
        # save plots of metrics at the end of training
        self.plot_dicts_in_recorders()
        plt_clear()

    def get_record_names(self,metrics_dict,prefix):
        record_names = list(metrics_dict.keys())
        record_names = [prefix+'_'+name for name in record_names]
        return record_names

    def plot_dicts_in_recorders(self):
        self.trainer.train_recorder.plot_summary_metrics(
            path = self.out_pth,
            caption='Plots of training summary metrics.',
            subplot_size=3,
            caption_size=10,
            )
        self.trainer.valid_recorder.plot_summary_metrics(
                path = self.out_pth,
                caption='Plots of validation summary metrics.',
                subplot_size=3,
                caption_size=10,
                )
    
    def log_current_status(self):
        info = f'Epoch_{self.trainer.current_epoch}_Step_{self.trainer.current_iter}:\t'+'\t'.join(
                    [f'{k}={v[-1]:f}' for k,v in self.trainer.train_recorder.summary_metrics_dict.items()]
                    )+ '\t' + '\t'.join(
                    [f'{k}={v[-1]:f}' for k,v in self.trainer.valid_recorder.summary_metrics_dict.items()]
                    )+ '\t' + '\t'.join(
                    [f'{k}={v[-1]:f}' for k,v in self.trainer.valid_recorder.manual_summary_metrics_dict.items()]
                    )
        self.logger.info(info)