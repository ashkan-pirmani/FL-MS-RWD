import os
from collections import OrderedDict
import datetime
from typing import Dict

import flwr as fl
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from utils import load_partition, PointWiseModel, AdaptiveDualBranchNet, train, test


class MSClients(fl.client.NumPyClient):
    def __init__(self, trainset, valset, testset, device, weights, hidden, dropout, num_layers, cid, hidden_ext, model_type):
        super().__init__()
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.weights = weights
        self.cid = cid
        self.hidden_ext = hidden_ext
        self.model_type = model_type
        self.device = device

        if model_type == "pointwise":
            self.model = PointWiseModel(hidden_dim=hidden, dropout_p=dropout, n_layers=num_layers)
        elif model_type == "AdaptiveDualBranchNet":
            self.model = AdaptiveDualBranchNet(core_hidden_dim=hidden, extension_hidden_dim=hidden_ext, dropout_p=dropout, n_core_layers=num_layers, training_size=len(trainset))
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def set_parameters(self, parameters):
        if self.model_type == "pointwise":
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        elif self.model_type == "AdaptiveDualBranchNet":
            shared_state_dict = OrderedDict()
            for name, param in self.model.state_dict().items():
                if 'extension_branch' not in name:
                    tensor = torch.Tensor(parameters.pop(0))
                    shared_state_dict[name] = tensor
            self.model.load_state_dict(shared_state_dict, strict=False)

    def get_parameters(self, config):
        if self.model_type == "pointwise":
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        elif self.model_type == "AdaptiveDualBranchNet":
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'extension_branch' not in name]

    def save_model(self, client_name, round_num, run_number):
        model_save_path = "/scratch/leuven/356/vsc35621/FL-MS-RWD/Sweep/FedAdagrad/output"
        client_folder = os.path.join(model_save_path, 'client-models', f'run_{run_number}', f'client_{self.cid}')

        # Create the directory if it doesn't exist
        if not os.path.exists(client_folder):
            os.makedirs(client_folder)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f'client_{client_name}_run-{run_number}_round-{round_num}_{timestamp}.pt'
        model_path = os.path.join(client_folder, file_name)
        torch.save(self.model.state_dict(), model_path)
        #print(f'Model saved: {model_path}')

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        epochs = config["epochs"]
        weight_decay = config["weight_decay"]
        patience = config["patience"]
        batch_size = config["batch_size"]
        run_number = config["run_number"]
        trainloader = DataLoader(self.trainset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valloader = DataLoader(self.valset, batch_size, shuffle=False, num_workers=0, pin_memory=True)

        _, _, val_loss, val_roc_auc, val_auc_pr = train(self.model, trainloader, valloader, self.weights, 'adam', epochs, lr, weight_decay, patience, device=self.device)

        # Save model after training
        self.save_model(self.cid, config["round"], run_number)

        return self.get_parameters({}), len(self.trainset), {"val_loss": val_loss, "val_size": len(self.valset), "val_roc_auc": val_roc_auc, "val_auc_pr": val_auc_pr, "client_id": self.cid}

    def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        self.set_parameters(parameters)

        testloader = DataLoader(self.testset, 64, shuffle=False, num_workers=0, pin_memory=True)
        loss, test_results = test(self.model, testloader, self.weights, self.device)
        test_roc_auc = float(test_results['auc_score'])

        metrics = {
            "auc_score": test_roc_auc,
            "auc_pr": float(test_results['auc_pr']),
            "loss": float(loss),
            "client_id": self.cid,
        }

        return float(loss), len(self.testset), metrics


def generate_client_fn(cfg: DictConfig):
    def client_function(cid):
        trainset, valset, testset, weights = load_partition(idx=int(cid), unique='country', min_unique=5)
        tr_batch_size = max(1, int((2 * len(trainset) / 100)))
        val_batch_size = max(1, int((2 * len(valset) / 100)))
        te_batch_size = max(1, int((2 * len(testset) / 100)))
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return MSClients(trainset, valset, testset, device,
                         weights,
                         cfg.model.hidden,
                         cfg.model.dropout,
                         cfg.model.num_layers,
                         cid,
                         cfg.model.hidden_ext,
                         cfg.model.model_type,
                         )

    return client_function