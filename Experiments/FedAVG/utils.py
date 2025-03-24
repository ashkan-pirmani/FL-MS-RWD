import os.path
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import os
import sys
import wandb
import csv
import numpy as np
import datetime
import logging
from torch.utils.data import ConcatDataset, TensorDataset
from dataset import create_tensor_datasets, load_data, create_finetuning_datasets
import os.path
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import math
import ray


device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#device = "cpu"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../FL-Expriment/data")


def load_partition(idx: int, unique, min_unique):
    """Load clients of the training and test data to simulate a partition."""
    df = load_data()
    datasets = create_tensor_datasets(df, unique, min_unique)

    # Check if the provided index is valid
    assert idx in range(len(datasets)), "Invalid index"
    # Directly use the 'idx' to get the corresponding datasets
    trainset = datasets[idx]['train']
    valset = datasets[idx]['val']
    testset = datasets[idx]['test']
    weights = class_weights(trainset)

    return trainset, valset, testset, weights

def compute_metrics(predictions, targets):
    """
    Compute ROC-AUC and AUC-PR. If only one unique label is present, return -1 for the metrics.
    """
    # Convert predictions to probabilities using sigmoid
    probas = [1 / (1 + np.exp(-pred)) for pred in predictions]

    # Check if only one unique label is present
    if len(np.unique(targets)) == 1:
        roc_auc = -1
        pr_auc = -1
        precision, recall, fpr, tpr = None, None, None, None
    else:
        roc_auc = roc_auc_score(targets, probas)
        pr_auc = average_precision_score(targets, probas)
        precision, recall, _ = precision_recall_curve(targets, probas)
        fpr, tpr, _ = roc_curve(targets, probas)

    return roc_auc, pr_auc, fpr, tpr, precision, recall


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch with CUDA optimization and corrected loss.backward() call.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(data)
        
        loss = criterion(outputs, target.unsqueeze(-1).float())
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    """
    Validate the model's performance on the validation set with CUDA optimization.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            outputs = model(data)
            
            loss = criterion(outputs, target.unsqueeze(-1).float())
            
            total_loss += loss.item()
            
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, all_preds, all_targets

def train(model, train_loader, val_loader, weights=None, optimizer_name='adam', num_epochs=10, lr=1e-3, weight_decay=1e-5, patience=10,device=device):
    """
    Training function with optimized printing.
    """
    model = model.to(device)
    
    # Conditionally set weights for BCEWithLogitsLoss
    if weights is not None:
        weights = weights.clone().detach().to(device, non_blocking=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1].view(1))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Use the specified optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized!")
    
    # Set min_lr for ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience//2, verbose=False, min_lr=1e-06)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate the model
        val_loss, predictions, true_labels = validate(model, val_loader, criterion, device)
        
        # Compute ROC-AUC and AUC-PR
        roc_auc, pr_auc, _, _, _, _ = compute_metrics(predictions, true_labels)
        
        # Update the learning rate
        scheduler.step(val_loss)
        
        # Print epoch info every print_interval epochs
        print_interval=50
        #if (epoch + 1) % print_interval == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            #print(f"Epoch [{epoch+1}/{num_epochs}] => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ROC-AUC: {roc_auc}, AUC-PR: {pr_auc}")
        
        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                #print("Early stopping due to no improvement in validation loss!")
                break
    
    return model, epoch, val_loss,roc_auc,pr_auc


def test(model, test_loader, weights, device):
    """
    Test the model's performance on the test set, including loss calculation.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            outputs = model(data)

            if weights is not None:
                weights = weights.clone().detach().to(device, non_blocking=True)
                criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1].view(1))
            else:
                criterion = nn.BCEWithLogitsLoss()

            # Calculate loss
            loss = criterion(outputs, target.unsqueeze(-1).float())
            total_loss += loss.item()

            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    # Compute ROC-AUC and AUC-PR
    roc_auc, pr_auc, fpr, tpr, precision, recall = compute_metrics(all_preds, all_targets)

    #print(f"Test Results => ROC-AUC: {roc_auc}, AUC-PR: {pr_auc}")

    results = {
        'auc_score': roc_auc,
        'auc_pr': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }
    return avg_loss ,results


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class PointWiseModel(nn.Module):
    def __init__(self, hidden_dim, dropout_p, n_layers):
        super(PointWiseModel, self).__init__()
        input_dim = 42

        if n_layers == 0:
            self.pre_layers = nn.Linear(input_dim, hidden_dim)
        else:
            self.pre_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p))

        self.mid_layers = nn.ModuleList()
        for layer in range(n_layers - 1):
            self.mid_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p)))

        self.out_layer = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.pre_layers(x)
        for mod in self.mid_layers:
            x = mod(x)
        x = self.out_layer(x)
        return x

class AdaptiveDualBranchNet(nn.Module):
    def __init__(self, core_hidden_dim, extension_hidden_dim, dropout_p, n_core_layers, training_size):
        super(AdaptiveDualBranchNet, self).__init__()
        self.n_core_layers = n_core_layers
        input_dim = 42

        # Core Layers
        self.pre_layers = nn.Sequential(
            nn.Linear(input_dim, core_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p))

        self.core_layers = nn.ModuleList()
        for _ in range(n_core_layers - 1):
            self.core_layers.append(
                nn.Sequential(
                    nn.Linear(core_hidden_dim, core_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p)))

        # Extension Layers - Separate Branch
        n_extension_layers = self.calculate_extension(training_size)
        extension_hidden_dim = extension_hidden_dim

        extension_layers = []
        extension_layers.append(nn.Linear(input_dim, extension_hidden_dim))
        extension_layers.append(nn.ReLU())
        extension_layers.append(nn.Dropout(dropout_p))
        for _ in range(n_extension_layers - 1):
            extension_layers.append(nn.Linear(extension_hidden_dim, extension_hidden_dim))
            extension_layers.append(nn.ReLU())
            extension_layers.append(nn.Dropout(dropout_p))

        self.extension_branch = nn.Sequential(*extension_layers)

        # Combining Layer - Merge outputs from core and extension layers
        self.combining_layer = nn.Linear(core_hidden_dim + extension_hidden_dim, core_hidden_dim)

        # Output Layer
        self.out_layer = nn.Sequential(nn.Linear(core_hidden_dim, 1))

    def forward(self, x):
        # Core layers pathway
        core_x = self.pre_layers(x)
        for layer in self.core_layers:
            core_x = layer(core_x)

        # Extension layers pathway
        ext_x = self.extension_branch(x)

        # Combine the outputs of the core and extension pathways
        combined_x = torch.cat((core_x, ext_x), dim=1)
        combined_x = self.combining_layer(combined_x)

        # Output layer
        return self.out_layer(combined_x)

    @staticmethod
    def calculate_extension(training_size):
        if training_size > 25000:
            layers = 5
        elif training_size >= 20000:
            layers = min(4, round(math.log(training_size / 1000) * 1.5))
        elif training_size >= 15000:
            layers = min(3, round(math.log(training_size / 1000) * 1.5))
        elif training_size >= 10000:
            layers = min(2, round(math.log(training_size / 1000) * 1.5))
        elif training_size >= 2000:
            layers = min(1, round(math.log(training_size / 1000) * 2.5))
        else:
            layers = 0
        return min(layers, 5)

    def print_extension_layers(self):
        print("Extension Layers:")
        for name, module in self.extension_branch.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Dropout)):
                print(f"  - {name}: {module}")

    def count_extension_layers(self):
        count = 0
        for name, module in self.extension_branch.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Dropout)):
                count += 1
        return count/3


def class_weights(trainset):
    _, train_labels = trainset.tensors
    len(train_labels)
    # Calculate number of samples in each class
    num_class_0 = (train_labels == 0).sum().item()
    num_class_1 = (train_labels == 1).sum().item()

    if num_class_0 == 0:
        # Return None or handle accordingly
        return None

    elif num_class_1 == 0:
        # Return None or handle accordingly
        return None

    else:
        # Calculate the total number of samples
        total_samples = len(train_labels)

        # Calculate the weights for each class
        weight_for_0 = total_samples / (2 * num_class_0)
        weight_for_1 = total_samples / (2 * num_class_1)

        # Store weights in a tensor
        class_weights = torch.tensor([weight_for_0, weight_for_1])
        return class_weights
