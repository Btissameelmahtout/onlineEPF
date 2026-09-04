#%%

# select country: "Germany" or "Spain"
country = "Spain"



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
from my_functions import forecast_MLP_rolling
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

# Number of epochs for initial window
num_epochs_init = 60  

# Number of epoch for remaining windows
num_epochs_all = 10


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





#----------------------------------------------------
#               Load Optuna Forecast
#------------------------------------------------------

model_types = [1, 2, 3, 4, 5, 7, 8]
forecast_dict = {}

for model_type in model_types:
    file_optuna_forecast = f"{country}/Model{model_type}/optuna_forecast.pkl"
    forecast_dict[model_type] = joblib.load(file_optuna_forecast)


# Collect arrays in the same order
arrays = [forecast_dict[m] for m in model_types]

# Concatenate along last axis
combined_forecasts = np.concatenate(arrays, axis=2)

print(combined_forecasts.shape)

combined_forecasts_tensor = torch.tensor(combined_forecasts, device=device)
#--------------------------------------------------
#                 Stepwise Selection
# ------------------------------------------------------
os.makedirs(f"{country}/BOA_all/", exist_ok=True)

weight_boa_step = {}
# value of the dependent variables in the validation data
actual_eval = dependent_var_tensor_eval[-length_study:, :]


forecast_new = torch.zeros((length_study, output_dim, len(model_types) * 500 + 1), device=device)


forecast_new[:, :, 0] = actual_eval

forecast_new[:, :, 1:] = combined_forecasts_tensor

errors = forecast_new[..., 1:] - forecast_new[..., :1]
rmse = torch.sqrt((errors**2).mean(axis=(0, 1)))
mae = (torch.abs(errors)).mean(axis=(0, 1))

#

k = 500  # number of models to select
# Get indices of the 500 smallest MAE values
_, valid_idx = torch.topk(mae, k, largest=False)

# valid_idx now contains the indices of the best 500 models (smallest MAE)
filtered_forecasts = combined_forecasts_tensor[:, :, valid_idx]

#
filtered_forecasts = combined_forecasts_tensor[:, :, valid_idx]

# update metrics
mae_filtered = mae[valid_idx]
# rmse_filtered = rmse[valid_idx]

# index of smallest mae in filtered set
best_number = torch.argmin(mae_filtered).item()


weight_boa_step = {}
# value of the dependent variables in the test data
actual_eval = dependent_var_tensor_eval[-length_study:, :]



#--------------------------------------------------
#                 Stepwise Selection
# ------------------------------------------------------



weight_boa_step = {}
# value of the dependent variables in the test data
actual_eval = dependent_var_tensor_eval[-length_study:, :]

trials_df = pd.DataFrame({
    "number": range(filtered_forecasts.shape[2])
})
best_model_combo, best_rmse, rmse_progression, model_progression, weight_boa_step = stepwise_selection(
    dat_eval, 
    actual_eval,   
    trials_df,
    best_number,
    length_study,
    Forecast_trials=filtered_forecasts,
    device=device)

# save the indecies of best endsemble models for later use
file_step_models = f"{country}/BOA_all/step_models.pkl"
joblib.dump(best_model_combo, file_step_models)

file_boa_weights = f"{country}/BOA_all/step_weights.pkl"
joblib.dump(weight_boa_step, file_boa_weights)


# #indices of the best model  in filtered_forecasts
original_idx = valid_idx[best_model_combo]

import joblib

# Move to CPU first
original_idx_cpu = original_idx.detach().cpu()

# Save
file_original_models = f"{country}/BOA_all/original_idx.pkl"
joblib.dump(original_idx_cpu, file_original_models)

#

# %%
