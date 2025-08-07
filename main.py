import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
from core.base import BaseSimpleTrainer, get_hook
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from core.Hooks import LogHook
from gen_data import get_train_two_feats_gau, get_test_two_feats_gau
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.util import seed_everything
from core.util import Config_feeder
from sklearn.metrics import confusion_matrix
from plot_bound import plot_bound
import logging


seed = 42

class BaseDataset(Dataset):
    def __init__(self, x, y, transform=None, **kwargs):
        self.x = x
        self.y = y
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        out = {'x': x, 'y': y}

        for k,v in self.__dict__.items():
            if k not in ['x','y','transform']:
                out[k] = v
        return out

def plot_data_dist(train_x, train_y, test_x, test_y, save_path=None):
    train_x = train_x + 1e-7*np.random.randn(*train_x.shape)
    test_x = test_x + 1e-7*np.random.randn(*test_x.shape)

    fig, ax = plt.subplots(2, 2, figsize=(6,5))
    sns.kdeplot(train_x[train_y==0,0], label='pos', ax=ax[0,0])
    sns.kdeplot(train_x[train_y==1,0], label='neg', ax=ax[0,0])
    sns.kdeplot(test_x[test_y==0,0], label='pos', ax=ax[0,1])
    sns.kdeplot(test_x[test_y==1,0], label='neg', ax=ax[0,1])

    sns.kdeplot(train_x[train_y==0,1], label='pos', ax=ax[1,0])
    sns.kdeplot(train_x[train_y==1,1], label='neg', ax=ax[1,0])
    sns.kdeplot(test_x[test_y==0,1], label='pos', ax=ax[1,1])
    sns.kdeplot(test_x[test_y==1,1], label='neg', ax=ax[1,1])

    ax[0,0].set_title('train_feature1')
    ax[0,1].set_title('test_feature1')
    ax[1,0].set_title('train_feature2')
    ax[1,1].set_title('test_feature2')

    for i in range(2):
        for j in range(2):
            ax[i,j].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'data_feat_dist.png'))
    plt.cla()
    plt.clf()
    plt.close()

class myTrainer(BaseSimpleTrainer):
    def __init__(self,model,max_epoch,train_loader,valid_loader,test_loader,valid_freq=1,method_name='cls_unbias',patience=20,min_improvement=1e-3,loss_ratio_threshold=2.5,min_epochs=100,*args,**kwargs):
        super().__init__(model=model,max_epoch=max_epoch,train_loader=train_loader,valid_loader=valid_loader,test_loader=test_loader,valid_freq=valid_freq,*args,**kwargs)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01)
        
        self.metrics_kwargs = dict(num_classes=2, task='multiclass',average='none')
        self.metrics = torchmetrics.MetricCollection(
            [Accuracy(**self.metrics_kwargs), Precision(**self.metrics_kwargs), Recall(**self.metrics_kwargs), F1Score(**self.metrics_kwargs)]
            )
        
        self.method_name = method_name

        self.eta = 1
        self.cls_eq_loss_weight = torch.linspace(1,2,self.max_epoch)
        self.eta_weight = torch.linspace(10,1,self.max_epoch)

        # Early stopping parameters
        self.patience = patience
        self.min_improvement = min_improvement
        self.loss_ratio_threshold = loss_ratio_threshold
        self.min_epochs = min_epochs  # Minimum epochs before early stopping can trigger
        self.best_valid_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        
        # Moving averages for losses
        self.train_loss_history = []
        self.valid_loss_history = []
        self.window_size = 10  # Increased moving average window
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_one_iter(self, batch):
        state = 'train'

        x, y = batch['x'], batch['y']
        logit = self.model(x).squeeze()
        pred = (nn.Sigmoid()(logit) > 0.5).float()
        loss_dict = self.cal_losses(logit, y, state)

        self.metrics.reset()
        metrics_dict = self.metrics(pred, y)
        metrics_dict = self.reformatted_metrics(metrics_dict)
        
        self.opt.zero_grad()
        loss_dict['loss'].backward()
        self.opt.step()

        self.train_out = dict(**loss_dict, **metrics_dict)
        self.current_train_loss = loss_dict['loss'].item()
        self.train_loss_history.append(self.current_train_loss)
    
    def get_moving_average(self, history):
        if len(history) < self.window_size:
            return sum(history) / len(history)
        return sum(history[-self.window_size:]) / self.window_size
    
    def check_consistent_overfitting(self):
        """Check if overfitting is consistent over the moving average window"""
        if len(self.train_loss_history) < self.window_size or len(self.valid_loss_history) < self.window_size:
            return False
            
        # Check if training loss is consistently decreasing and validation loss consistently increasing
        train_trend = all(self.train_loss_history[i] > self.train_loss_history[i+1] 
                         for i in range(-self.window_size, -1))
        valid_trend = all(self.valid_loss_history[i] < self.valid_loss_history[i+1] 
                         for i in range(-self.window_size, -1))
        
        return train_trend and valid_trend
    
    def valid(self):
        state = 'val'

        batch = next(iter(self.valid_loader))
        x, y = batch['x'], batch['y']

        logit = self.model(x).squeeze()
        pred = (nn.Sigmoid()(logit) > 0.5).float()
        loss_dict = self.cal_losses(logit, y, state)

        self.metrics.reset()
        metrics_dict = self.metrics(pred, y)
        metrics_dict = self.reformatted_metrics(metrics_dict)

        self.valid_out = dict(**loss_dict, **metrics_dict)
        current_valid_loss = loss_dict['loss'].item()
        self.valid_loss_history.append(current_valid_loss)

        # Skip early stopping checks if we haven't reached minimum epochs
        if self.current_epoch < self.min_epochs:
            self.logger.info(f'Epoch {self.current_epoch} - Still in minimum epoch period ({self.min_epochs} epochs)')
            return

        # Get moving averages
        avg_train_loss = self.get_moving_average(self.train_loss_history)
        avg_valid_loss = self.get_moving_average(self.valid_loss_history)
        
        # Calculate loss ratio using moving averages
        loss_ratio = avg_valid_loss / avg_train_loss

        # Check for extreme overfitting
        is_overfitting = False
        if len(self.train_loss_history) >= self.window_size and len(self.valid_loss_history) >= self.window_size:
            consistent_overfit = self.check_consistent_overfitting()
            ratio_too_high = loss_ratio > self.loss_ratio_threshold
            is_overfitting = consistent_overfit and ratio_too_high

        # Early stopping check
        if not hasattr(self, 'best_loss_ratio'):
            self.best_loss_ratio = float('inf')
        
        if loss_ratio < self.best_loss_ratio * (1 + self.min_improvement):  # Allow some tolerance
            self.best_loss_ratio = loss_ratio
            self.best_epoch = self.current_epoch
            self.counter = 0
            # Save best model and metrics
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_train_out = {k: v.detach().clone() for k, v in self.train_out.items()}
            self.best_valid_out = {k: v.detach().clone() for k, v in self.valid_out.items()}
            # Run test and save test metrics at best validation point
            self.test()
            self.best_test_out = {k: v.detach().clone() for k, v in self.test_out.items()}
            
            self.logger.info(f'Improved loss ratio: {loss_ratio:.6f} (avg_val_loss: {avg_valid_loss:.6f}, avg_train_loss: {avg_train_loss:.6f})')
        else:
            if is_overfitting:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    self.logger.info(f'Early stopping triggered after {self.current_epoch} epochs due to consistent extreme overfitting')
                    self.logger.info(f'Best model was from epoch {self.best_epoch} with loss ratio {self.best_loss_ratio:.6f}')
            else:
                self.counter = 0
        
        # Log current metrics
        self.logger.info(f'Epoch {self.current_epoch} - Val Loss: {current_valid_loss:.6f} (avg: {avg_valid_loss:.6f}), '
                        f'Train Loss: {self.current_train_loss:.6f} (avg: {avg_train_loss:.6f}), Loss Ratio: {loss_ratio:.6f}')

    def reformatted_metrics(self, metrics_dict):
        out = {}
        for k,v in metrics_dict.items():
            if v.dim() == 1:
                out[k+'_neg'] = v[0]
                out[k+'_pos'] = v[1]
            else:
                out[k] = v
        return out
    
    def test(self):
        state = 'test'

        batch = next(iter(self.test_loader))
        x, y = batch['x'], batch['y']
        
        logit = self.model(x).squeeze()
        pred = (nn.Sigmoid()(logit) > 0.5).float()
        loss_dict = self.cal_losses(logit, y, state)

        self.metrics.reset()
        metrics_dict = self.metrics(pred, y)
        metrics_dict = self.reformatted_metrics(metrics_dict)

        self.test_out = dict(**loss_dict, **metrics_dict)

    def cal_losses(self, logits, y, state='train'):
        loss_dict = {}

        loss = self.loss_func(logits, y)

        self.eta = self.eta_weight[self.current_epoch] # !!!

        if state == 'train':
            if self.method_name == 'cls_unbias':
                loss_pos = loss[y == 1].mean()
                loss_neg = loss[y == 0].mean()
                loss_pos_exp_ = (loss_pos.detach() * self.eta).exp()
                loss_neg_exp_ = (loss_neg.detach() * self.eta).exp()
                loss_sum = loss_pos_exp_ + loss_neg_exp_
                loss_dro = (loss_pos_exp_ / loss_sum) * loss_pos + (loss_neg_exp_ / loss_sum) * loss_neg 
                loss_dict['loss_dro'] = loss_dro

                loss_cls_eq = (loss_pos - loss_neg).abs() * self.cls_eq_loss_weight[self.current_epoch]
                loss = loss_dro + loss_cls_eq
                loss_dict['loss_cls_eq'] = loss_cls_eq
            
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()
        loss_dict['loss'] = loss
        return loss_dict


feeder = Config_feeder()
feeder.register_config(
    method_name=['cls_unbias','bl'],
    do_balance=[True,False],
    test_idx=[1,2]
)

decision_boundary = dict(
    balanced = dict(
        bl = {},
        cls_unbias = {}
    ),
    imbalanced = dict(
        bl = {},
        cls_unbias = {}
    )
)

method_name_list = ['cls_unbias','bl']

for each_config in feeder.config_generator:
    
    method_name = each_config['method_name']
    test_idx = each_config['test_idx']
    do_balance = each_config['do_balance']

    exp_name = ''
    if do_balance:
        train_x, train_y = get_train_two_feats_gau(1000,0.5,seed=seed)
        val_x, val_y = get_train_two_feats_gau(250,0.5,seed=seed+1)
        exp_name = exp_name + 'balance/'
    else:
        train_x, train_y = get_train_two_feats_gau(1000,0.1,seed=seed)
        val_x, val_y = get_train_two_feats_gau(250,0.5,seed=seed+1)
        exp_name = exp_name + 'imbalance/'
    
    if test_idx == 1:
        test_x, test_y = get_test_two_feats_gau(500,0.5,seed=seed)
        exp_name = exp_name + 'test1/'
    elif test_idx == 2:
        test_x, test_y = get_test_two_feats_gau(500,0.5,do_rand=True,seed=seed)
        exp_name = exp_name + 'test2/'
    else:
        raise ValueError(f'test_idx must be 1 or 2, but got {test_idx}')
    
    exp_name = exp_name + f'{method_name}'

    train_x = torch.tensor(train_x).float()
    train_y = torch.tensor(train_y).float()
    test_x = torch.tensor(test_x).float()
    test_y = torch.tensor(test_y).float()
    val_x = torch.tensor(val_x).float()
    val_y = torch.tensor(val_y).float()

    train_data = BaseDataset(train_x, train_y)
    test_data = BaseDataset(test_x, test_y)
    val_data = BaseDataset(val_x, val_y)
    train_loader = DataLoader(train_data, batch_size=len(train_y), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_y), shuffle=False)
    val_loader = DataLoader(val_data, batch_size=len(val_y), shuffle=False)
    
    seed_everything(seed)
    model = nn.Linear(2,1)

    trainer = myTrainer(
        model=model,
        max_epoch=1000,
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        valid_freq=5,  
        method_name=method_name,
        patience=5,  
        min_improvement=1e-4  
    )

    trainer.register_hooks([
            LogHook(exp_name)
        ])
    trainer.run()

    save_pth = get_hook(trainer, 'LogHook').out_pth
    logger = get_hook(trainer, 'LogHook').logger

    plot_data_dist(train_x.numpy(), train_y.numpy(), test_x.numpy(), test_y.numpy(), save_pth)

    # Report best metrics
    logger.info('\n\nBest metrics (at epoch {}):\n'.format(trainer.best_epoch) + ('-'*20))
    logger.info('\nBest train metrics:')
    for each in trainer.best_train_out.keys():
        logger.info(f'{each}: {trainer.best_train_out[each].item()}')
    
    logger.info('\nBest validation metrics:')
    for each in trainer.best_valid_out.keys():
        logger.info(f'{each}: {trainer.best_valid_out[each].item()}')
    
    logger.info('\nBest test metrics:')
    for each in trainer.best_test_out.keys():
        logger.info(f'{each}: {trainer.best_test_out[each].item()}')

    # Load best model for final weights and bias
    model.load_state_dict(trainer.best_model_state)
    weights = model.weight.data.numpy().squeeze()
    bias = model.bias.data.numpy().squeeze()

    logger.info(f'Best model weights: {weights}')
    logger.info(f'Best model bias: {bias}')

    # log normalized weights
    weights_norm = weights / np.linalg.norm(weights)
    logger.info(f'Normalized weights: {weights_norm}')
    
    balanced_str = 'balanced' if do_balance else 'imbalanced'
    if method_name == 'bl':
        decision_boundary[balanced_str][method_name] = dict(weight=weights, bias=bias)

    elif method_name == 'cls_unbias':
        decision_boundary[balanced_str][method_name] = dict(weight=weights, bias=bias)

    # plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(3,3))
    sns.barplot(x=['w1','w2'],y=weights)
    plt.title('Best model weights')
    plt.savefig(f'{save_pth}/weights.png')
    plt.cla()
    plt.clf()
    plt.close()

    #Get the confusion matrix for best model
    plt.figure(figsize=(3,3))
    with torch.no_grad():
        logit = model(test_x).squeeze()
        pred = (nn.Sigmoid()(logit) > 0.5).float()
    cf_matrix = confusion_matrix(test_y, pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title('Best model confusion matrix')
    plt.savefig(f'{save_pth}/confusion_matrix.png')
    plt.cla()
    plt.clf()
    plt.close()

    del trainer, model, train_loader, test_loader, train_data, test_data, train_x, train_y, test_x, test_y

for do_balance in [True, False]:
    balanced_str = 'balanced' if do_balance else 'imbalanced'
    plot_bound(decision_boundary[balanced_str], do_balance, seed)