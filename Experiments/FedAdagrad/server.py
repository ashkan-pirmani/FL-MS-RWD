import os
from datetime import datetime
from typing import List, Tuple, Union, Dict, Optional
from omegaconf import DictConfig

import numpy as np
import wandb
import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

# Constants for paths
MODEL_SAVE_PATH = "output/best_model"
CLIENT_MODEL_PATH = "output/client-models"

# Global variables to track the best ROC-AUC
best_server_roc_auc = -1
best_round = -1
current_server_roc_auc = -1

def reset_roc_auc_metrics():
    global best_server_roc_auc, best_round, current_server_roc_auc
    best_server_roc_auc = -1
    best_round = -1
    current_server_roc_auc = -1

def get_on_fit_config(config: DictConfig, run_number):
    def fit_config_fn(server_round):
        return {
            'lr': config.training.lr,
            'epochs': config.training.epochs,
            'weight_decay': config.training.weight_decay,
            'patience': config.training.patience,
            'batch_size': config.training.batch_size,
            'round': server_round,
            'run_number': run_number,
            'best_round': best_round
        }
    return fit_config_fn

# Initialize a step counter (federated round counter)
federated_round = 0

class EarlyStoppingException(Exception):
    pass

def reset_federated_round():
    global federated_round
    federated_round = 0

def weighted_average(metrics: List[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, str]]:
    global federated_round, current_server_roc_auc
    federated_round += 1  # Increment the federated round counter

    # Calculate weights for auc_score, auc_pr, and loss
    roc_auc_weights = [m["num_examples"] * m["auc_score"] for m in metrics if "auc_score" in m]
    auc_pr_weights = [m["num_examples"] * m["auc_pr"] for m in metrics if "auc_pr" in m]
    loss_weights = [m["num_examples"] * m["loss"] for m in metrics if "loss" in m]

    # Calculate total examples for valid auc_score and auc_pr
    total_examples_roc_auc = sum(m["num_examples"] for m in metrics if "auc_score" in m)
    total_examples_auc_pr = sum(m["num_examples"] for m in metrics if "auc_pr" in m)
    total_examples = sum(m["num_examples"] for m in metrics)

    # Compute the weighted averages
    roc_auc = sum(roc_auc_weights) / total_examples_roc_auc if total_examples_roc_auc > 0 else "NOT AVAILABLE"
    auc_pr = sum(auc_pr_weights) / total_examples_auc_pr if total_examples_auc_pr > 0 else "NOT AVAILABLE"
    loss = sum(loss_weights) / total_examples if total_examples > 0 else "NOT AVAILABLE"

    print(f"Aggregated ROC-AUC: {roc_auc} | Aggregated AUC-PR: {auc_pr} | Aggregated Loss: {loss}")

    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/aggregated_metrics.txt', 'a') as f:
        f.write(f"{roc_auc},{auc_pr},{loss}\n")

    # Update the current server ROC-AUC
    current_server_roc_auc = roc_auc

    # Log individual client metrics to WandB
    for metric in metrics:
        client_id = metric.get("client_id")
        if client_id is not None:
            wandb.log({
                f"client_{client_id}_roc_auc": metric["auc_score"],
                f"client_{client_id}_auc_pr": metric["auc_pr"],
                f"client_{client_id}_loss": metric["loss"]
            }, step=federated_round)  # Use federated_round as step

    wandb.log({"server_roc_auc": roc_auc, "server_auc_pr": auc_pr, "server_loss": loss}, step=federated_round)

    return {"roc_auc": roc_auc, "auc_pr": auc_pr, "loss": loss}

class SaveModelStrategy(fl.server.strategy.FedAdagrad):
    def __init__(self, patience: int = 10, run_number: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.previous_parameters = None
        self.should_stop = False
        self.run_number = run_number

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        
        # Monitor validation loss for early stopping
        val_losses = [(fit_res.metrics["val_loss"], fit_res.metrics["val_size"]) for _, fit_res in results if
                      "val_loss" in fit_res.metrics and "val_size" in fit_res.metrics]
        val_roc_aucs = [(fit_res.metrics["val_roc_auc"], fit_res.metrics["val_size"]) for _, fit_res in results if
                        "val_roc_auc" in fit_res.metrics and "val_size" in fit_res.metrics]
        val_auc_prs = [(fit_res.metrics["val_auc_pr"], fit_res.metrics["val_size"]) for _, fit_res in results if
                       "val_auc_pr" in fit_res.metrics and "val_size" in fit_res.metrics]

        total_val_size = sum(size for _, size in val_losses)
        total_roc_auc_size = sum(size for _, size in val_roc_aucs)
        total_auc_pr_size = sum(size for _, size in val_auc_prs)

        avg_val_loss = sum(loss * size for loss, size in val_losses) / total_val_size if total_val_size > 0 else None
        avg_roc_auc = sum(roc_auc * size for roc_auc, size in val_roc_aucs) / total_roc_auc_size if total_roc_auc_size > 0 else None
        avg_auc_pr = sum(auc_pr * size for auc_pr, size in val_auc_prs) / total_auc_pr_size if total_auc_pr_size > 0 else None

        if avg_val_loss is not None and avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.epochs_without_improvement = 0
        elif avg_val_loss is not None:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            print("Early stopping triggered")
            raise EarlyStoppingException("Early stopping triggered")

        # Log the validation metrics for the current round
        wandb.log({
            "avg_val_loss_server": avg_val_loss if avg_val_loss is not None else "NOT AVAILABLE",
            "avg_val_roc_auc_server": avg_roc_auc if avg_roc_auc is not None else "NOT AVAILABLE",
            "avg_val_auc_pr_server": avg_auc_pr if avg_auc_pr is not None else "NOT AVAILABLE"
        }, step=server_round)

        # Compute the change in parameters
        if self.previous_parameters is not None and aggregated_parameters is not None:
            previous_ndarrays = parameters_to_ndarrays(self.previous_parameters)
            current_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            changes = [
                np.linalg.norm(current - previous) / np.linalg.norm(previous)
                for current, previous in zip(current_ndarrays, previous_ndarrays)
            ]
            avg_change = sum(changes) / len(changes)
            print(f"Average parameter change: {avg_change:.4%}")
            wandb.log({"avg_parameter_change": avg_change * 100}, step=server_round)
            
        self.previous_parameters = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        global best_server_roc_auc, best_round, MODEL_SAVE_PATH, current_server_roc_auc

        # Extract the loss and metrics from the evaluation results
        loss_results = [res.loss for _, res in results]
        metrics_results = [{**res.metrics, "num_examples": res.num_examples} for _, res in results]

        aggregated_loss = sum(loss_results) / len(loss_results) if loss_results else None

        # Calculate the weighted average metrics
        aggregated_metrics = weighted_average(metrics_results)
        current_server_roc_auc = aggregated_metrics["roc_auc"]

        if current_server_roc_auc != "NOT AVAILABLE" and current_server_roc_auc > best_server_roc_auc:
            best_server_roc_auc = current_server_roc_auc
            best_round = server_round
            self.save_best_model(server_round)

        return aggregated_loss, aggregated_metrics

    def save_best_model(self, server_round):
        if self.previous_parameters is not None:
            aggregated_ndarrays = parameters_to_ndarrays(self.previous_parameters)
            model_folder = os.path.join(MODEL_SAVE_PATH, f'run_{self.run_number}')
            os.makedirs(model_folder, exist_ok=True)

            # Remove old models if they exist in the folder
            for f in os.listdir(model_folder):
                os.remove(os.path.join(model_folder, f))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            file_name = f"best_model_round-{server_round}_{timestamp}.npz"
            file_path = os.path.join(model_folder, file_name)
            np.savez(file_path, *aggregated_ndarrays)
            print(f"Saved best model for round {server_round} with ROC-AUC: {current_server_roc_auc}")

            self.delete_old_client_models(best_round, self.run_number)

    def delete_old_client_models(self, best_round, run_number):
        client_model_folder = os.path.join(CLIENT_MODEL_PATH, f'run_{run_number}')
        if os.path.exists(client_model_folder):
            for client_id in os.listdir(client_model_folder):
                client_round_folder = os.path.join(client_model_folder, client_id)
                for model_file in os.listdir(client_round_folder):
                    if f"round-{best_round}" not in model_file:
                        os.remove(os.path.join(client_round_folder, model_file))
                        #print(f'Deleted old client model: {os.path.join(client_round_folder, model_file)}')

    def on_training_end(self):
        self.cleanup_models()
        #print(f"Training completed. Best round: {best_round} with ROC-AUC: {best_server_roc_auc}")

    def cleanup_models(self):
        self.delete_old_client_models(best_round, self.run_number)