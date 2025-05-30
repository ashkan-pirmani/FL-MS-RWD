# Personalized Federated Learning for Predicting Disability Progression in Multiple Sclerosis Using Real-World Routine Clinical Data

## 📖 Overview

The fragmented nature of Real-World Data (RWD) presents significant challenges in researching low-prevalence diseases like Multiple Sclerosis (MS). This study leverages **Federated Learning (FL)** to enable collaborative model training without centralizing sensitive patient data, addressing variations across data providers through **Personalized Federated Learning (PFL)**.

We evaluate standard FL alongside two **personalization strategies** for predicting confirmed MS disability progression over two years using data from **26,000+ patients in the MSBase registry**:

1. **AdaptiveDualBranchNet** – A novel architecture that selectively exchanges key model parameters, enabling nuanced adaptation across diverse clinical centers.
2. **Fine-Tuning** – Adjusting the global FL model to better fit local site data.

### 🔍 Key Findings

- **Personalized FL outperforms standard FL**: The adaptive and fine-tuned versions of **FedProx** and **FedAVG** achieved the highest **ROC–AUC scores**:
  - **FedProx**: 0.8398 ± 0.0019 (adaptive) and 0.8375 ± 0.0019 (fine-tuned)
  - **FedAVG**: 0.8384 ± 0.0014 (adaptive) and 0.8370 ± 0.0016 (fine-tuned)
- **Standard FL falls behind**: Among non-personalized FL methods, **FedAdam** (0.7919 ± 0.0031) and **FedYogi** (0.7910 ± 0.0028) performed best.
- **Personalization is essential**: This study establishes that personalization is not a luxury but a necessity for unlocking FL’s full predictive potential in clinical decision-making.

---

## 📂 Repository Structure

```
FL-MS-RWD/
│── Data/                        # (Data sources, if applicable)
│── Experiments/                 # FL and PFL training setups
│   │── BestModels/              # Best performing models Configuration
│   │── Centralized/             # Centralized baseline models
│   │── FedAdagrad/              # Federated Adagrad model
│   │── FedAdam/                 # Federated Adam model
│   │── FedAVG/                  # Federated Averaging (FedAVG)
│   │── FedProx/                 # Federated Proximal (FedProx)
│   │── FedYogi/                 # Federated Yogi model
│── .gitignore                   # Ignore unnecessary files
│── README.md                    # This document
```

## ⚙️ Hyperparameter Configurations

The **best performing hyperparameter configurations** used in this study are stored in the [`Experiments/BestModels`](https://github.com/ashkan-pirmani/FL-MS-RWD/tree/paper/Experiments/BestModels) directory.  
- These files contain the exact configurations that led to the reported results.
- You can directly reuse them as **base configurations** to replicate or extend experiments.

### Hyperparameters Overview

- **training**
  - `batch_size`: Number of samples per client update step.
  - `lr`: Learning rate for the optimizer.
  - `epochs`: Number of local epochs per federated round.
  - `weight_decay`: L2 regularization applied to model weights.
  - `patience`: Early stopping patience for local training.
  - `patience_server`: Early stopping patience for global server aggregation.

- **model**
  - `hidden`: Hidden dimension size of the neural network layers.
  - `dropout`: Dropout rate for regularization.
  - `num_layers`: Number of layers in the model.
  - `hidden_ext`: Size of extra hidden dimensions (only for AdaptiveDualBranchNet).
  - `model_type`: Can be set to `"AdaptiveDualBranchNet"` (default architecture used) or a pointwise baseline.

- **federation**
  - `federation_rounds`: Total number of federated communication rounds.
  - `num_clients`: Number of participating clients in the FL simulation.

- **main**
  - `cpu` / `gpu`: Resource allocation for training.

- **other**
  - `repetition`: Number of repetitions for statistical robustness.

### Strategy-Specific Hyperparameters

Certain federated algorithms require additional hyperparameters:
- **FedProx**: uses `mu` to control the proximal term strength.
- **FedAdam**, **FedYogi**, **FedAdagrad**: include optimizer-specific parameters handled internally.

### `model_type: "AdaptiveDualBranchNet"`

- `AdaptiveDualBranchNet` is the primary model architecture for personalized federated learning in this project.
- It separates globally shared and adaptive client-specific parameters, allowing selective parameter exchange during aggregation.
- Model logic is implemented in `utils.py`, while parameter exchange mechanisms (`set_parameters` and `get_parameters`) are handled inside `clients.py`. This allows flexible control over which parameters are globally synchronized.
- Alternatively, `model_type` can be set to a simple pointwise baseline model for non-personalized training.



## 📊 Results Summary

| Model      | Personalization | ROC–AUC |
|------------|---------------|---------|
| FedProx    | ✅ Adaptive   | **0.8398 ± 0.0019** |
| FedAVG     | ✅ Adaptive   | **0.8384 ± 0.0014** |
| FedProx    | ✅ Fine-tuned | 0.8375 ± 0.0019 |
| FedAVG     | ✅ Fine-tuned | 0.8370 ± 0.0016 |
| FedAdam    | ❌ No         | 0.7919 ± 0.0031 |
| FedYogi    | ❌ No         | 0.7910 ± 0.0028 |

---

## 📌 Key Contributions

- 🏥 **First large-scale application of PFL and FL for MS prediction using real-world data**.
- 🔬 **Benchmarks FL vs. PFL models**, showing the necessity of personalized approaches.
- ⚡ **Introduces AdaptiveDualBranchNet**, a novel architecture for federated adaptation.
- 🔑 **Provides concrete guidelines** for implementing PFL in clinical research.

---

## Environment Setup

To reproduce the environment for this project, please follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd FL-MS-RWD
   ```

2. **Set Up the Conda Environment:**

   Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed.

   Create the environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment:**

   ```bash
   conda activate fl
   ```

4. **Verify the Environment:**

   To check that all required packages are installed, run:

   ```bash
   conda list
   ```

5. **Run the Application:**

   Launch the application using:

   ```bash
   python main.py
   ```

   (Replace `main.py` with the appropriate entry point if necessary.)

### Alternative: Using Docker

If you prefer using Docker, follow these steps:

- **Build the Docker Image:**

  ```bash
  docker build -t fl-ms-rwd .
  ```

- **Run the Docker Container:**

  ```bash
  docker run --rm -it fl-ms-rwd
  ```
---

## 🏆 Citation

If you use this repository in your research, please cite:

```
@article{PFL-Pirmani2025,
  title={Personalized Federated Learning for Predicting Disability Progression in Multiple Sclerosis Using Real-World Routine Clinical Data},
  author={Pirmani et al.},
  journal={npj Digital Medicine},
  year={2025}
}
```

---

## 🤝 Acknowledgments

This research was made possible by data from **MSBase** and contributions from multiple institutions. We acknowledge the importance of federated and personalized learning in tackling real-world medical challenges.

