import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pickle

from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), "../../")), "./Data")

def create_tensor_datasets(df: pd.DataFrame, unique_col: str, min_unique: int) -> List[torch.utils.data.TensorDataset]:
    df = df[~df[unique_col].isin(df[unique_col].value_counts()[df[unique_col].value_counts() < min_unique].index)]
    dfs = [df.loc[df[unique_col] == val].reset_index(drop=True) for val in df[unique_col].unique()]
    _, cov_cols, _ = get_msbase_static_dataset(no_dmt=False, minimal_example=False, num_visits=3, include_clinic=True,
                                               include_country=True)
    cov_cols = [col for col in cov_cols if
                col not in ["slice_id", "Time", "clinic", "country", "clinic_i", "country_i", "label"]]

    datasets = []
    for i, df_i in enumerate(dfs):
        data_dfs = df_i[cov_cols].values
        label_dfs = df_i['label'].astype(int).values

        # First split: 80% for combined training and validation sets, 20% for test set
        data_train_val, data_test, label_train_val, label_test = train_test_split(
            data_dfs, label_dfs, test_size=0.2, random_state=42)

        # Second split: Divide the 80% into 60% for training and 20% for validation
        data_train, data_val, label_train, label_val = train_test_split(
            data_train_val, label_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

        # Normalization
        scaler = StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_val = scaler.transform(data_val)
        data_test = scaler.transform(data_test)

        # Create PyTorch Datasets
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_train), torch.LongTensor(label_train))
        val_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_val), torch.LongTensor(label_val))
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_test), torch.LongTensor(label_test))

        datasets.append({
            'name': f"{unique_col}_{df_i[unique_col].iloc[0]}",
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        })

    return datasets

def create_finetuning_datasets(df: pd.DataFrame, unique_col: str, min_unique: int, partition: str) -> List[Dict[str, torch.utils.data.TensorDataset]]:
    # Debugging: Print initial number of samples

    df = df[~df[unique_col].isin(df[unique_col].value_counts()[df[unique_col].value_counts() < min_unique].index)]

    # Initial split into 60% and 40% for the entire dataset
    df_60, df_40 = train_test_split(df, test_size=0.3, random_state=42)

    if partition == 'fed':
        dfs = [df_60.loc[df[unique_col] == val].reset_index(drop=True) for val in df_60[unique_col].unique()]
        _, cov_cols, _ = get_msbase_static_dataset(no_dmt=False, minimal_example=False, num_visits=3, include_clinic=True,
                                                   include_country=True)
        cov_cols = [col for col in cov_cols if
                    col not in ["slice_id", "Time", "clinic", "country", "clinic_i", "country_i", "label"]]

        datasets = []
        for i, df_i in enumerate(dfs):
            data_dfs = df_i[cov_cols].values
            label_dfs = df_i['label'].astype(int).values

            # First split: 80% for initial training set, 20% for combined test and validation sets
            data_train_i, data_test_val, label_train_i, label_test_val = train_test_split(
                data_dfs, label_dfs, test_size=0.2, random_state=42)

            # Second split: Divide the 20% into 10% for test and 10% for validation
            data_test_i, data_val_i, label_test_i, label_val_i = train_test_split(
                data_test_val, label_test_val, test_size=0.5, random_state=42)

            # Adjust the initial training set to be 60% of the original dataset
            data_train_i, _, label_train_i, _ = train_test_split(
                data_train_i, label_train_i, test_size=0.25, random_state=42)

            # Normalization
            scaler = StandardScaler().fit(data_train_i)
            data_train_i = scaler.transform(data_train_i)
            data_val_i = scaler.transform(data_val_i)
            data_test_i = scaler.transform(data_test_i)

            train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_train_i), torch.LongTensor(label_train_i))
            val_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_val_i), torch.LongTensor(label_val_i))
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_test_i), torch.LongTensor(label_test_i))

            datasets.append(
                {'name': f"{unique_col}_{df_i[unique_col].iloc[0]}", 'train': train_dataset, 'val': val_dataset,
                 'test': test_dataset})

    if partition == 'finetune':
        dfs = [df_40.loc[df[unique_col] == val].reset_index(drop=True) for val in df_40[unique_col].unique()]
        _, cov_cols, _ = get_msbase_static_dataset(no_dmt=False, minimal_example=False, num_visits=3,
                                                   include_clinic=True,
                                                   include_country=True)
        cov_cols = [col for col in cov_cols if
                    col not in ["slice_id", "Time", "clinic", "country", "clinic_i", "country_i", "label"]]

        datasets = []
        for i, df_i in enumerate(dfs):
            data_dfs = df_i[cov_cols].values
            label_dfs = df_i['label'].astype(int).values

            # First split: 80% for initial training set, 20% for combined test and validation sets
            data_train_i, data_test_val, label_train_i, label_test_val = train_test_split(
                data_dfs, label_dfs, test_size=0.2, random_state=42)

            # Second split: Divide the 20% into 10% for test and 10% for validation
            data_test_i, data_val_i, label_test_i, label_val_i = train_test_split(
                data_test_val, label_test_val, test_size=0.5, random_state=42)

            # Adjust the initial training set to be 60% of the original dataset
            data_train_i, _, label_train_i, _ = train_test_split(
                data_train_i, label_train_i, test_size=0.25, random_state=42)

            # Normalization
            scaler = StandardScaler().fit(data_train_i)
            data_train_i = scaler.transform(data_train_i)
            data_val_i = scaler.transform(data_val_i)
            data_test_i = scaler.transform(data_test_i)

            train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_train_i), torch.LongTensor(label_train_i))
            val_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_val_i), torch.LongTensor(label_val_i))
            test_dataset = torch.utils.data.TensorDataset(torch.Tensor(data_test_i), torch.LongTensor(label_test_i))

            datasets.append(
                {'name': f"{unique_col}_{df_i[unique_col].iloc[0]}", 'train': train_dataset, 'val': val_dataset,
                 'test': test_dataset})

    datasets = sorted(datasets, key=lambda x: x['name'])
    return datasets

def get_msbase_static_dataset(no_dmt=False, include_country=False, include_clinic=False, minimal_example=False,
                              num_visits=3):
    directory = DATA_DIR
    df = pd.read_pickle(directory + "/df.pkl")
    with open(directory + "/cov_cols.pkl", "rb") as fp:  # Unpickling
        cov_cols = pickle.load(fp)
    with open(directory + "/continuous_cols.pkl", "rb") as fp:  # Unpickling
        continuous_cols = pickle.load(fp)

    if not no_dmt:
        dmt_groups = ["Mild", "Moderate", "High"]
        dmt_cols = ["DMT_AT_0_" + dmt for dmt in dmt_groups]

        dmt_groups_ind = ["High_Induction"]
        dmt_cols_ind = ["DMT_IND_AT_0_" + dmt for dmt in dmt_groups_ind]

    else:
        cov_cols = [c for c in cov_cols if c not in dmt_cols]
        cov_cols = [c for c in cov_cols if c not in dmt_cols_ind]

    if not include_country:
        cov_cols = [c for c in cov_cols if c != "country"]
        cov_cols = [c for c in cov_cols if c != "country_i"]
    if not include_clinic:
        cov_cols = [c for c in cov_cols if c != "clinic"]
        cov_cols = [c for c in cov_cols if c != "clinic_i"]

    if minimal_example:
        cov_cols = ["EDSS_at_0"]
        continuous_cols = ["EDSS_at_0"]

    cols = cov_cols + ["PATIENT_ID", "slice_id", "label"]
    return df[cols], cov_cols, continuous_cols

def load_data():


    def get_msbase_static_dataset(no_dmt=False, include_country=False, include_clinic=False, minimal_example=False,
                                  num_visits=3):
        directory = DATA_DIR
        df = pd.read_pickle(directory + "/df.pkl")
        with open(directory + "/cov_cols.pkl", "rb") as fp:  # Unpickling
            cov_cols = pickle.load(fp)
        with open(directory + "/continuous_cols.pkl", "rb") as fp:  # Unpickling
            continuous_cols = pickle.load(fp)

        if not no_dmt:
            dmt_groups = ["Mild", "Moderate", "High"]
            dmt_cols = ["DMT_AT_0_" + dmt for dmt in dmt_groups]

            dmt_groups_ind = ["High_Induction"]
            dmt_cols_ind = ["DMT_IND_AT_0_" + dmt for dmt in dmt_groups_ind]

        else:
            cov_cols = [c for c in cov_cols if c not in dmt_cols]
            cov_cols = [c for c in cov_cols if c not in dmt_cols_ind]

        if not include_country:
            cov_cols = [c for c in cov_cols if c != "country"]
            cov_cols = [c for c in cov_cols if c != "country_i"]
        if not include_clinic:
            cov_cols = [c for c in cov_cols if c != "clinic"]
            cov_cols = [c for c in cov_cols if c != "clinic_i"]

        if minimal_example:
            cov_cols = ["EDSS_at_0"]
            continuous_cols = ["EDSS_at_0"]

        cols = cov_cols + ["PATIENT_ID", "slice_id", "label"]
        return df[cols], cov_cols, continuous_cols

    datatype = "MSBase2020"
    num_visits = 3
    fold = 0
    # %% md
    ### Loading the dataframe
    # %%
    df_processed, cov_cols, continuous_cols = get_msbase_static_dataset(no_dmt=False, minimal_example=False,
                                                                        num_visits=num_visits, include_clinic=True,
                                                                        include_country=True)
    return df_processed
