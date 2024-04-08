import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import load_model, save_model
from matplotlib import pyplot as plt
from stuff import sufficient_sample_size
from prettytable import PrettyTable


class MeansDataset(torch.utils.data.Dataset):
    """Dataset with means of loss function on different datasets from UCI."""
    def __init__(self, lookback=5, idx=None, train_size=None, datasets=None):
        super(MeansDataset, self).__init__()
        self.features = torch.tensor(np.array([datasets[name]['sample_sizes'] for name in datasets.keys()]), dtype=torch.float32)
        self.targets = torch.tensor(np.array([datasets[name]['means'] for name in datasets.keys()]), dtype=torch.float32)
        self.features = self.transform(self.features, features=True)
        self.targets = self.transform(self.targets, features=False)
        self.X, self.y = self.create_dataset(lookback=lookback, idx=idx, train_size=train_size)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)
    
    def create_dataset(self, lookback=5, idx=None, train_size=None):
        """
        Transform a time series into a prediction dataset
        
        Args:
            ts: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
        """    
        X, y = [], []
        
        if idx is not None and train_size is not None:
            sample_sizes = self.features[idx][:train_size]
            means = self.targets[idx][:train_size]
            for i in range(len(sample_sizes) - lookback):
                feature = sample_sizes[i:i+lookback]
                target = means[i+1:i+1+lookback]
                X.append(feature)
                y.append(target)
        else:
            for sample_sizes, means in zip(self.features, self.targets):
                for i in range(len(sample_sizes) - lookback):
                    feature = sample_sizes[i:i+lookback]
                    target = means[i+1:i+1+lookback]
                    X.append(feature)
                    y.append(target)
        
        return torch.stack(X), torch.stack(y)
    
    def transform(self, x, features=True, forward=True, idx=None):
        if forward and idx is not None:
            raise NotImplementedError()
        if forward:
            if features:
                self.features_min = x.min(axis=1).values.view(-1, 1).repeat(1, self.features.shape[1])
                self.features_max = (x - self.features_min).max(axis=1).values.view(-1, 1).repeat(1, self.features.shape[1])
                min_ = self.features_min
                max_ = self.features_max
            else:
                self.targets_min = x.min(axis=1).values.view(-1, 1).repeat(1, self.features.shape[1])
                self.targets_max = (x - self.targets_min).max(axis=1).values.view(-1, 1).repeat(1, self.features.shape[1])
                min_ = self.targets_min
                max_ = self.targets_max
            return (x - min_) / max_
        else:
            if features:
                min_ = self.features_min
                max_ = self.features_max
            else:
                min_ = self.targets_min
                max_ = self.targets_max
            if idx is None:
                return min_ + max_ * x
            else:
                return min_[idx] + max_[idx] * x
            
            
class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(dim=-1))
        x = self.linear(x)
        return x.squeeze(dim=-1)
    
    
def train(n_epochs=1000, lookback=10, hidden_size=64, batch_size=4, datasets=None):
     
    means_dataset = MeansDataset(lookback=lookback, datasets=datasets)
    means_dataloader = torch.utils.data.DataLoader(means_dataset, shuffle=True, batch_size=4)
    means_model = LSTMForecaster(hidden_size=hidden_size)
    means_optimizer = optim.Adam(means_model.parameters())
    means_criterion = nn.MSELoss()

    for epoch in tqdm(range(n_epochs)):
        means_model.train()
        for X_batch, y_batch in means_dataloader:
            y_pred = means_model(X_batch)
            means_loss = means_criterion(y_pred, y_batch)
            means_optimizer.zero_grad()
            means_loss.backward()
            means_optimizer.step()
            
    return means_model


def finetune(model: torch.nn.Module,
             idx: int = 0,
             n_epochs: int = 10,
             train_size: float = 0.3,
             lookback: int = 10,
             datasets=None,
             plot=True,
             verbose=True,
             save=False,
             filename=None):
    
    train_size = int(0.3 * len(next(iter(datasets.values()))['sample_sizes']))

    means_dataset = MeansDataset(lookback=lookback, idx=idx, train_size=train_size, datasets=datasets)
    means_dataloader = torch.utils.data.DataLoader(means_dataset, shuffle=True, batch_size=4)
    means_model = load_model(model, './checkpoints', 'lstm.pth', verbose=verbose)
    means_optimizer = optim.Adam(means_model.parameters())
    criterion = nn.MSELoss()

    if verbose:
        loop = tqdm(range(n_epochs))
    else:
        loop = range(n_epochs)

    for epoch in loop:
        means_model.train()
        for X_batch, y_batch in means_dataloader:
            y_pred = means_model(X_batch)
            loss = criterion(y_pred, y_batch)
            means_optimizer.zero_grad()
            loss.backward()
            means_optimizer.step()
            
    with torch.no_grad():

        samples = means_dataset.transform(means_dataset.features[idx], features=True, forward=False, idx=idx)
        means = means_dataset.transform(means_dataset.targets[idx], features=False, forward=False, idx=idx)

        train_samples = samples[:train_size]
        train_means = means[:train_size]

        test_samples = samples[train_size:]
        test_means = means[train_size:]
        
        means_predicted = means_dataset.transform(means_model(means_dataset.features[idx]), features=False, forward=False, idx=idx)
        
        if plot:
        
            plt.plot(train_samples, train_means, label='Train')
            plt.plot(test_samples, test_means, label='Test')
            plt.plot(train_samples, means_predicted[:train_size], label='Train Predicted')
            plt.plot(test_samples, means_predicted[train_size:], label='Test Predicted')
            plt.title(list(datasets.keys())[idx])
            plt.xlabel('Available sample size')
            plt.ylabel('Mean')
            plt.legend()
            
            plt.tight_layout()
            if save:
                plt.savefig('./figs/' + filename, bbox_inches='tight')
            plt.show()
        
    return samples, means, means_predicted


def compare_sufficient(model: torch.nn.Module,
                       thresholds=np.logspace(-3, 3, 1000),
                        idx: int = 0,
                        n_epochs: int = 10,
                        train_size: float = 0.3,
                        lookback: int = 10,
                        datasets=None,
                        plot=True,
                        save=False,
                        filename=None):
    
    samples, means, means_predicted = finetune(model, idx, n_epochs, train_size, lookback, datasets, plot=False)
    
    
    sufficient = []
    sufficient_approx = []

    for eps in thresholds:
        
        sufficient.append(sufficient_sample_size(sample_sizes=samples.numpy().astype(int),
                                                        means=means.numpy().astype(float),
                                                        eps=eps,
                                                        method='rate'))
        sufficient_approx.append(sufficient_sample_size(sample_sizes=samples.numpy().astype(int),
                                                        means=means_predicted.numpy().astype(float),
                                                        eps=eps,
                                                        method='rate'))
    
    if plot:
        
        plt.plot(thresholds, sufficient, label='True')
        plt.plot(thresholds, sufficient_approx, label='Approximation')
        plt.xscale('log')
        plt.xlabel(r"$\varepsilon$")
        plt.ylabel(r"$m^*$")
        plt.title(list(datasets.keys())[idx])
        plt.legend()
        if save:
            plt.savefig('./figs/' + filename, bbox_inches='tight')
        plt.show()
        
    return sufficient, sufficient_approx


def compare_different_sufficient(model: torch.nn.Module,
                                thresholds=np.logspace(-3, 3, 7),
                                n_epochs: int = 10,
                                train_size: float = 0.3,
                                lookback: int = 10,
                                datasets=None):
    
    for eps in thresholds:
    
        sufficient = {}
        sufficient_approx = {}

        for idx, name in tqdm(enumerate(datasets.keys())):
            
            samples, means, means_predicted = finetune(model, idx, n_epochs, train_size,
                                                    lookback, datasets, plot=False, verbose=False)
            
            sufficient[name] = sufficient_sample_size(sample_sizes=samples.numpy().astype(int),
                                                        means=means.numpy().astype(float),
                                                        eps=eps,
                                                        method='rate')
            sufficient_approx[name] = sufficient_sample_size(sample_sizes=samples.numpy().astype(int),
                                                            means=means_predicted.numpy().astype(float),
                                                            eps=eps,
                                                            method='rate')
            
        table = PrettyTable()
        table.title = f'eps = {eps}'
        table.field_names = ["Dataset name", "Actual", "Predicted"]

        for name in datasets.keys():
            table.add_row([name, sufficient[name], sufficient_approx[name]])

        print(table)