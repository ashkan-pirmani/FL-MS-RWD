import torch.utils.data
import torch
from utils import load_center_partition
from utils import train, test, PointWiseModel, AdaptiveDualBranchNet
import numpy as np
import wandb
import os
import torch
import wandb


trainset, valset, testset = load_center_partition()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#"values": ["pointwise", "AdaptiveDualBranchNet"]


config_dict = {
    "model": {
        "model_type": "pointwise",  # No need for extra curly braces or "values" here
        "hidden": 512,
        "dropout": 0.1,
        "num_layers": 5,
        "hidden_ext": 64},
    "training": {
        "lr": 0.0001,
        "epochs": 150,
        "patience": 150,
        "batch_size": 1024,
    },
}





def save_weights(model, path):
    weights_dict = {}

    weights_dict['arr_0'] = model.pre_layers[0].weight.data.cpu().numpy()
    weights_dict['arr_1'] = model.pre_layers[0].bias.data.cpu().numpy()

    for i in range(len(model.mid_layers)):
        weight_key = f'arr_{2 * i + 2}'
        bias_key = f'arr_{2 * i + 3}'
        weights_dict[weight_key] = model.mid_layers[i][0].weight.data.cpu().numpy()
        weights_dict[bias_key] = model.mid_layers[i][0].bias.data.cpu().numpy()

    weights_dict['arr_10'] = model.out_layer[0].weight.data.cpu().numpy()
    weights_dict['arr_11'] = model.out_layer[0].bias.data.cpu().numpy()

    np.savez(path, **weights_dict)


def run_experiments(n_experiments, train_dataset, val_dataset, test_dataset, num_epochs=150,
                    lr=0.0001, weight_decay=0.00005):
    for i in range(n_experiments):
        print(f"Running experiment {i + 1}/{n_experiments}")
        wandb.init(project="FL_RWD", tags=["Centralized", "PointWise"], config=config_dict)

        # Model configuration
        model_type = wandb.config["model"]["model_type"]
        hidden = wandb.config["model"]["hidden"]
        dropout_p = wandb.config["model"]["dropout"]
        n_layers = wandb.config["model"]["num_layers"]
        hidden_ext = wandb.config["model"]["hidden_ext"]

        # Data loaders
        trainloader = torch.utils.data.DataLoader(train_dataset, wandb.config["training"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        valloader = torch.utils.data.DataLoader(val_dataset, wandb.config["training"]["batch_size"], num_workers=0, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_dataset, wandb.config["training"]["batch_size"], num_workers=0, pin_memory=True)

        # Model instantiation
        if model_type == "pointwise":
            model = PointWiseModel(hidden_dim=hidden, dropout_p=dropout_p, n_layers=n_layers)
        elif model_type == "AdaptiveDualBranchNet":
            model = AdaptiveDualBranchNet(core_hidden_dim=hidden, extension_hidden_dim=hidden_ext, dropout_p=dropout_p, n_core_layers=n_layers, training_size=len(train_dataset))

        model = model.to(device, non_blocking=True)

        # Training loop with epoch-level logging and saving
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train and validate for one epoch
            model, current_epoch, val_loss, roc_auc, pr_auc = train(model, trainloader, valloader, weights=None, optimizer_name='adam', num_epochs=1, lr=lr, weight_decay=weight_decay, patience=150, device=device)

            # Log server ROC-AUC and AUC-PR at each epoch
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "server_roc_auc": roc_auc,
                "server_auc_pr": pr_auc
            })

            # Save the model at each epoch in the experiment-specific directory
            model_save_dir = f"./saved_models/experiment_{i + 1}"
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_save_path)

        # Perform final testing and logging
        test_loss, test_results = test(model, testloader, weights=None, device=device)
        wandb.log({
            "final_test_roc_auc": test_results["auc_score"],
            "final_test_auc_pr": test_results["auc_pr"]
        })

        # Close the WandB run
        wandb.finish()



run_experiments(10, trainset, valset, testset, 150, 0.0001, 0.00005)