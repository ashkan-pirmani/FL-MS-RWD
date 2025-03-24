import os
import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
import flwr as fl
from utils import load_partition, get_model_params, PointWiseModel
from clients import generate_client_fn
from server import get_on_fit_config, weighted_average, SaveModelStrategy, EarlyStoppingException, reset_federated_round, reset_roc_auc_metrics
from omegaconf import DictConfig, OmegaConf
import wandb
import ray

# Ensure proper Ray initialization in the main script
def initialize_ray():
    if ray.is_initialized():
        ray.shutdown()
    try:
        ray.init(address='auto', ignore_reinit_error=True)
    except Exception as e:
        print(f"Failed to initialize Ray: {e}")
        raise

initialize_ray()

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    global best_server_roc_auc, best_round, current_server_roc_auc  # Add global declaration for reset

    N_RUNS = cfg.repetition
    all_results = {
        "roc_auc": [],
        "auc_pr": [],
        "loss": []
    }

    for run in range(N_RUNS):
        print(f"Starting run {run + 1} of {N_RUNS}")

        # Reset the federated round counter and ROC-AUC metrics
        reset_federated_round()
        reset_roc_auc_metrics()

        run_number = run + 1  # Update the run_number for each run

        wandb.init(project="FL-MS-RWD-Nature", config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                   tags=["FedProx", "Adaptive","Final"])
        wandb.config.update({"run_number": run_number})

        client_fn = generate_client_fn(cfg)

        strategy = SaveModelStrategy(
            fraction_fit=1,
            min_fit_clients=cfg.federation.num_clients,
            min_available_clients=cfg.federation.num_clients,
            fraction_evaluate=1,
            min_evaluate_clients=cfg.federation.num_clients,
            on_fit_config_fn=get_on_fit_config(cfg, run_number),
            evaluate_metrics_aggregation_fn=weighted_average,
            patience=cfg.training.patience_server,
            proximal_mu=cfg.training.proximal,
            run_number=run_number,
        )

        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=cfg.federation.num_clients,
                config=fl.server.ServerConfig(num_rounds=cfg.federation.federation_rounds),
                strategy=strategy,
                client_resources={'num_cpus': cfg.main.cpu},
                ray_init_args={"address": "auto", "ignore_reinit_error": True}  # Ensure Ray initializes correctly
            )
        except EarlyStoppingException as e:
            print(e)
        except Exception as e:
            print(f"Simulation error: {e}")

        save_path = HydraConfig.get().runtime.output_dir
        results_path = Path(save_path) / 'results.pkl'
        results = {'history': history}

        with open(str(results_path), 'wb') as h:
            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

        # Ensure 'results' directory exists
        os.makedirs('results', exist_ok=True)
        aggregated_metrics_path = 'results/aggregated_metrics.txt'

        # Ensure the file exists before reading
        if not os.path.exists(aggregated_metrics_path):
            with open(aggregated_metrics_path, 'w') as f:
                f.write('roc_auc,auc_pr,loss\n')

        with open(aggregated_metrics_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip the header line
                last_line = lines[-1]
                roc_auc, auc_pr, loss = map(float, last_line.split(','))
                all_results["roc_auc"].append(roc_auc)
                all_results["auc_pr"].append(auc_pr)
                all_results["loss"].append(loss)

        wandb.log({"final_roc_auc": roc_auc, "final_auc_pr": auc_pr})
        wandb.finish()

        # Call cleanup after each run
        strategy.cleanup_models()

    means = {
        "roc_auc": sum(all_results["roc_auc"]) / N_RUNS,
        "auc_pr": sum(all_results["auc_pr"]) / N_RUNS,
        "loss": sum(all_results["loss"]) / N_RUNS
    }

    std_devs = {
        "roc_auc": (sum([(x - means["roc_auc"]) ** 2 for x in all_results["roc_auc"]]) / N_RUNS) ** 0.5,
        "auc_pr": (sum([(x - means["auc_pr"]) ** 2 for x in all_results["auc_pr"]]) / N_RUNS) ** 0.5,
        "loss": (sum([(x - means["loss"]) ** 2 for x in all_results["loss"]]) / N_RUNS) ** 0.5
    }

    with open('results/final_aggregated_metrics.txt', 'w') as f:
        f.write(f"roc_auc_mean: {means['roc_auc']}, roc_auc_std_dev: {std_devs['roc_auc']}\n")
        f.write(f"auc_pr_mean: {means['auc_pr']}, auc_pr_std_dev: {std_devs['auc_pr']}\n")
        f.write(f"loss_mean: {means['loss']}, loss_std_dev: {std_devs['loss']}\n")

if __name__ == "__main__":
    main()