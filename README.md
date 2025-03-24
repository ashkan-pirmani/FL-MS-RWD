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

