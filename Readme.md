# Electricity Price Forecasting: Bridging Linear Models, Neural Networks and Online Learning
**Author:** Btissame El Mahtout (btissame.el@tu-dortmund.de) & Florian Ziel

**Date of assembly:** March 2026 | **Last updated:** September 2026

This repository contains the code and data required to reproduce the results of the project, including **online model execution**, **benchmark evaluation**, **robustness analyses**, and the generation of the tables and figures reported in the manuscript.

## 1 Project Overview

The repository is organized into the following main folders:

- `Data/` → Contains the data required for training, validation, and testing.
- `Functions/` → Contains all functions required by the Python and R scripts.
- `Online/` → Contains the validation and test scripts for the proposed online models.
- `Benchmark/` → Contains the validation and test scripts for LEAR, DNN, and GAM.
- `No_online/` → Contains the scripts used for the experiments without online updating.
- `Crisis/` →Contains the scripts for the proposed model used in the crisis-period ablation analysis.
- `Figures/` →Contains the code used to generate the tables and figures reported in the manuscript.
---

## 2 Environment Setup

Before running the experiments, create **two separate virtual environments**, since the `Benchmark/` scripts use a different Python version and package configuration from the rest of the repository.

### Benchmark environment

Use this environment for all scripts in the `Benchmark/` folder.

- Python version: **3.10**
- Required packages: listed in `requirements_ben.txt`

### Online environment

Use this environment for the `Online/`, `No_Online/`, `Crisis/`, and `Figures/` folders.

- Python version: **3.12**
- Required packages: listed in `requirements_online.txt`

---

## 3. Repository Structure and Folder Descriptions

### 3.1 📂 Online 

The `Online/` folder contains four main scripts:

- `Online_validation.py` → Hyperparameter tuning and model validation
- `Online_test.py` → Evaluation of the selected models over the test period
- `BOA_validation.py` → Validation of the **BOA all** model
- `BOA_test.py` → Evaluation of the **BOA all** model over the test period


#### 3.1.1 Running the Online Models

Run:

```python
Online_validation.py
```

The user only needs to specify the model type and country:

The `country` can be either: "Germany" or "Spain"

The `model_type` determines which model specification is estimated:

| `model_type` | Models |
|---|---|
| 1 | RLin & RLin (BOA) |
| 2 | MLP & MLP (BOA) |
| 3 | MLP with RLin & MLP with RLin (BOA) |
| 4 | FLin & FLin (BOA) |
| 5 | MLP with FLin & MLP with FLin (BOA) |
| 7 | RLin with OLS & RLin with OLS (BOA) |
| 8 | MLP with RLin and OLS & MLP with RLin and OLS (BOA) |

For example:

```python
model_type = 1
country = "Germany"
```

runs the RLin and RLin (BOA) models for the German-Luxembourg market.


After completing the validation step for all models and countries, run:

```python
Online_test.py
```

Use the same model_type and country settings as in the corresponding validation run. The script then loads the selected model configuration obtained during validation and generates forecasts for the test period.


#### 3.1.2 Running the BOA all

Before running the BOA scripts, the corresponding Online models must first be run. Then run:

```python
BOA_validation.py
```

Only the country needs to be specified: "Germany"  # or "Spain"

After completing BOA validation, run:

```python
BOA_test.py
```

Again, only specify the country: "Germany"  # or "Spain"

The script generates BOA all forecasts for the test period.


### 3.2 📂 Benchmark Models

The `Benchmark/` folder contains the scripts used to estimate and evaluate the benchmark models: **LEAR**, **DNN** and **GAM Online**.

#### 3.2.1 Running LEAR and DNN


Run the benchmark validation script.

```python
benchmark_validation.py
```

Only the country needs to be specified: "Germany"  # or "Spain"

The validation script performs the model-selection and tuning procedure for DNN.

After completing the validation step, run:

```python
benchmark_test.py
```
Again, only specify the country: "Germany"  # or "Spain"


The script then generates forecasts for both the LEAR and DNN models over the test period.



#### 3.2.2 Running GAM Online

The GAM Online validation and test scripts are also located in the `Benchmark/` folder and are written in R.

Run the GAM validation script after selecting the country:

```r
GAM_validation.R
```

The validation script performs the hyperparameter-selection procedure for GAM Online.

After completing validation, run the corresponding GAM test script for each country:

```r
GAM_test.R
```

The script then generates forecasts for the GAM Online model over the test period.


#### 3.2.3 Crisis-Period Analysis

The `Benchmark/` folder also contains scripts with the `_crisis` suffix.

These scripts reproduce the experiments conducted for the electricity-price crisis period.

The `_crisis` scripts should be run **without any modification**, since the required dates, settings, and model configurations are already defined in the scripts.


### 3.3 📂 No_Online 

The `No_Online/` folder contains the scripts used to reproduce the experiments in which online learning is removed.

For these scripts, the `num_epochs` parameter must be specified as one of the following:

```python
num_epochs = 10  # 60 or None
```
No changes should be made to the predefined **country** or **model_type** settings.

### 3.4 📂 Crisis 

The `Crisis/` folder contains the scripts used to evaluate the proposed model during the crisis period.

The scripts in this folder should be run **without any modification**, as all required settings are already predefined.


### 3.5 📂 Figures

After running all the scripts described above, run the scripts in the `Figures/` folder to generate the tables and figures reported in the manuscript.

---
## 4 Recommended Execution Order

For the complete set of experiments, the recommended execution order is:

1. Ensure that the required datasets are available in the `Data/` folder.
2. Run `Online_validation.py` for all required `model_type` specifications and for both Germany and Spain.
3. Run `Online_test.py` for the corresponding models and countries.
4. Run `BOA_validation.py` for both countries.
5. Run `BOA_test.py` for both countries.
6. Run `benchmark_validation.py` for both countries to perform the DNN model-selection and tuning procedure.
7. Run `benchmark_test.py` for both countries to generate forecasts for LEAR and DNN.
8. Run `GAM_validation.R` for both countries.
9. Run `GAM_test.R` for both countries to generate the GAM Online forecasts.
10. Run the scripts in the `No_Online/` folder for each required number of epochs: `10`, `60`, and `None`. 
11. Run the `Benchmark/_crisis_` without modification.
12. Run the scripts in the `Crisis/` folder without modification.
13. After all experiments have been completed, run the scripts in the `Figures/` folder to generate the tables and figures reported in the manuscript.

**Note:** For each analysis, run the corresponding validation script (`_validation`) first, followed by the associated test script (`_test`).    

---

## 5. Summary

| Folder / Component | Purpose | Required Inputs / Instructions | Approximate Runtime |
|--------------------|---------|--------------------------------|---------------------|
| `Online/Online_` | Validation and testing of the online models | `model_type`, `country` | Validation: approximately 10–20 hours per model and country; Test: from a few seconds to a few minutes |
| `Online/BOA_` | Validation and testing of the BOA all combination | `country` | Validation: around 10 hours per country; Test: a few minutes per country |
| `Benchmark/benchmark_` | Validation and testing of LEAR and DNN | `country` | Validation: approximately 4–6 hours per country; Test: approximately 7–10 hours per country |
| `Benchmark/GAM_` | Validation and testing of GAM Online | `country` | Validation: approximately 4–6 hours per country; Test: a few minutes per country |
| `No_Online/` | Experiments without online learning | `num_epochs` | Validation: from a few hours to a few days per `num_epochs`; Test: a few minutes per `num_epochs` |
| `Benchmark/_crisis_` scripts | LEAR/DNN evaluation during the crisis period | Run without modification | Validation: approximately 4–6 hours; Test: approximately 7–10 hours |
| `Benchmark/GAM_crisis_` scripts | GAM Online evaluation during the crisis period | Run without modification | Validation: approximately 4–6 hours; Test: a few minutes |
| `Crisis/` | Crisis-period analysis | Run without modification | Validation: approximately 10–20 hours; Test: from a few seconds to a few minutes |
| `Figures/` | Generates the tables and figures reported in the manuscript | Run after completing all experiments | — |

> **Computational environment:** The validation scripts for the `Online/`, `No_Online/`, and `Crisis/` experiments were run on a machine equipped with a GPU, 32 CPU cores, and 64 GB of RAM. All `Benchmark/` scripts, as well as the test scripts for the other experiments, were run on a MacBook Pro (2020) equipped with an Apple M1 processor (8 CPU cores) and 16 GB of unified memory.

---

## 6. License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the `LICENSE` file for details.

The main parts of `benchmark_validation.py`, `benchmark_validation_crisis.py`, `benchmark_test.py`, and `benchmark_test_crisis.py` are adapted from the [epftoolbox repository](https://github.com/jeslago/epftoolbox/tree/master), which is distributed under the AGPL-3.0 license. These files have been modified for the experiments conducted in this project.
