# %%

# Select Number of epochs: 10,60, None
num_epochs = 10

# select germany
country = "Germany"
model_type = 3



# packages
import re
import torch
import numpy as np
import pandas as pd
from calendar import day_abbr
import locale
import os
import sys
sys.path.append("../Functions")
from my_functions import DST_trafo
from my_functions import reg_matrix
from my_functions import forecast_MLP_rolling_no_online
from my_functions import stepwise_selection
import matplotlib.pyplot as plt
import torch.nn as nn

import joblib


import optuna
import time
import random
import matplotlib.colors as mcolors


# Set CuBLAS deterministic behavior to enforce deterministic behavior for CuBLAS operations when using optuna
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# for cuda
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False



#------------------------------------------
#                data Preprocessing
#----------------------------------------------
#  set language setting
locale.getlocale()

#  check the working directory
os.getcwd()

# read the data
data = pd.read_csv(f"../Data/{country}.csv")


# select the price and time
id_select = 1
price = data.iloc[:, id_select]

time_utc = pd.to_datetime(data["time_utc"], utc=True, format="%Y-%m-%d %H:%M:%S")
local_time_zone = "CET"
time_lt = time_utc.dt.tz_convert(local_time_zone)

output_dim = 24

#  Save the start and end-time
start_end_time_S = time_lt.iloc[[0, -1]].dt.tz_localize(None).dt.tz_localize("UTC")

# creating 'fake' local time
start_end_time_S_num = pd.to_numeric(start_end_time_S)
time_S_numeric = np.arange(
    start=start_end_time_S_num.iloc[0],
    stop=start_end_time_S_num.iloc[1] + 24 * 60 * 60 * 10**9 / output_dim,
    step=24 * 60 * 60 * 10**9 / output_dim,
)

#  'fake' local time
time_S = pd.Series(pd.to_datetime(time_S_numeric, utc=True))
dates_S = pd.Series(time_S.dt.date.unique())

# import DST_trafo function and use it on data
data_array = DST_trafo(X=data.iloc[:, 1:], Xtime=time_utc, tz=local_time_zone)




#---------------------------------------------------
#          Define the variable of interest
#-------------------------------------------------------

# Save the variable names
reg_names = data.columns[1:]

# Specify the weekday dummies: Mon, Sat, and Sun
wd = [1, 6, 7]

# Specify the lags: lag 1 , lag 2 and lag 7
price_s_lags = [1, 2, 7]

# Specify DA and its lags: predictions of tomorrow 
da_lag = [0]

fuel_lags = [2]

# batch size
batch_size = 32



# Keep the last 2 years for test
N = 2 * 365
dat_eval = data_array[:-N, :, :]
days_eval = pd.to_datetime(dates_S)[:-N]

#--------------------------------------------------
#                Regression matrix
#-------------------------------------------------------

# Extrax the matrix and the nuber of column needed for indices
regmat_eval = reg_matrix(
    dat_eval, days_eval, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[0]
columns_s = reg_matrix(
    dat_eval, days_eval, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[1]
columns_base = reg_matrix(
    dat_eval, days_eval, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[2]
columns_total = reg_matrix(
    dat_eval, days_eval, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[3]

# Remove NAs
regmat0_eval = regmat_eval.dropna()
#
# Extract the coeficients name

indices = list(range(0 * columns_s, (0 + 1) * columns_s))
non_sn_indices = list(range(columns_total - columns_base, columns_total))
coefficient = regmat0_eval.iloc[:, indices + non_sn_indices].columns[1:].tolist()
# Remove the '_s<number>' ending from elements in the list
coefficient = [re.sub(r"_s\d+$", "", col) for col in coefficient]


# Convert DataFrame to a NumPy array first, then to a tensor
regmat_tensor_eval = torch.from_numpy(regmat0_eval.values).float().to(device)

# Creat dictionary for the indices of each s
index_dict = {}
for s in range(output_dim):
    indices = list(range(s * columns_s, (s + 1) * columns_s))

    if s == output_dim - 1:
        non_sn_indices = list(range(columns_total - columns_base, (columns_total - 1)))
        index_dict[s] = indices + non_sn_indices

    else:
        non_sn_indices = list(range(columns_total - columns_base, columns_total))
        index_dict[s] = indices + non_sn_indices
# Extract the indices of  dependent varibales
dependent_index = []
for s in range(output_dim):
    dependent_index.append(index_dict[s][0])
#  Create active independent variable for each s

# Initialize an empty dictionary to store the results
active_regressor = {}

# Loop through each key in the dictionary to get regressors without depent variables
for key in index_dict:
    # Exclude the first value and assign to the new dictionary
    active_regressor[key] = index_dict[key][1:]

num_columns = (
    max(max(indices) for indices in active_regressor.values()) + 1
)  # Determine max column index

# Initialize the mask tensor with zeros
mask_in_out_red = torch.zeros((output_dim, num_columns), dtype=torch.float32, device=device)

# Populate the mask tensor that has 1 in active regressors and zero otherwise
for s, indices in active_regressor.items():
    mask_in_out_red[s, indices] = 1


# Initialize the mask tensor with zeros
mask_in_out_full = torch.ones((output_dim, num_columns), dtype=torch.float32, device=device)
mask_in_out_full[:, dependent_index] = 0


#remove the dependent variables from regression matrix 

dependent_var_eval = regmat0_eval.iloc[:, dependent_index]
dependent_var_tensor_eval = (
    torch.from_numpy(dependent_var_eval.values).float().to(device)
)
regmat_tensor_eval[:, dependent_index] = 0




#-----------------------------------------------------
#           Define the validation dataset
#--------------------------------------------------------
#  Define the evaluation period length
length_eval = int(2 * 365)  # Two years and half

# The first obdervation in the evaluation period
begin_eval = regmat0_eval.shape[0] - length_eval

length_study = length_eval




#-----------------------------------------------------
#           Hyper-parameter tuning
#--------------------------------------------------------

os.makedirs(f"{country}/Model{model_type}/", exist_ok=True)


 # Number of trials
n_trials = 500  

Forecast_trials = torch.zeros(length_study, output_dim, n_trials)
weight_tensor = {}


def objective_init(trial, num_epochs):



    # Sample hyperparameters
    if num_epochs == 10:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        D = trial.suggest_int("D", 30, 2 * 365)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-2, log=True)

    elif num_epochs == 60:
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        lambda_reg = trial.suggest_float("lambda_reg", 1e-4, 1e-2, log=True)

        D = trial.suggest_int("D", 1, 365)

    elif num_epochs is None:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        D = trial.suggest_int("D", 30, 2 * 365)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 5, 75)

    if model_type in {7, 8}:
        use_ols_weights = True
        alpha = trial.suggest_float("alpha", 0, 2)

    else:
        use_ols_weights = None
        alpha = None

    if model_type in {2, 3, 5, 6, 8}:
        number_neurons = trial.suggest_int("number_neurons", 1, 128)
    else:
        number_neurons = 0

    # Call the function and retrieve timings
    overall_mean, _, forecast = forecast_MLP_rolling_no_online(
        dat_eval,
        begin_eval,
        regmat_tensor_eval,
        dependent_var_tensor_eval,
        learning_rate,
        num_epochs,
        D,
        number_neurons,
        weight_decay,
        lambda_reg,
        length_study,
        dependent_index,
        mask_in_out_red,
        mask_in_out_full,
        batch_size,
        num_columns,
        device,
        model_type,)



    Forecast_trials[:, :, trial.number] = forecast

    # return overall_mean
    return overall_mean



study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
)
# study.optimize(objective_init, n_trials=n_trials)
study.optimize(
    lambda trial: objective_init(trial, num_epochs=num_epochs), n_trials=n_trials
)


# Get the best trial(the one  with smallest rmse)
best_trial = study.best_trial
print("Best MAE:", best_trial.value)
print("Best hyperparameters:", best_trial.params)
# number of the trial with the lowest RMSE
best_number = best_trial.number

if num_epochs in {10, 60}:
    os.makedirs(f"{num_epochs}/{country}/Model{model_type}/", exist_ok=True)
elif num_epochs == None:
    os.makedirs(f"num_epochs_tun/{country}/Model{model_type}/", exist_ok=True)




#--------------------------------------------------
#               Save dataframes
#-----------------------------------------------------

if num_epochs is None:
    file_optuna = f"num_epochs_tun/{country}/Model{model_type}/optuna_study.pkl"
    joblib.dump(study, file_optuna)

    file_optuna_forecast = (
        f"num_epochs_tun/{country}/Model{model_type}/optuna_forecast.pkl"
    )
    joblib.dump(Forecast_trials, file_optuna_forecast)





elif num_epochs in {10, 60}:
    file_optuna = f"{num_epochs}/{country}/Model{model_type}/optuna_study.pkl"
    joblib.dump(study, file_optuna)

    file_optuna_forecast = (
        f"{num_epochs}/{country}/Model{model_type}/optuna_forecast.pkl"
    )
    joblib.dump(Forecast_trials, file_optuna_forecast)


else:
    print("Error")

# %%
