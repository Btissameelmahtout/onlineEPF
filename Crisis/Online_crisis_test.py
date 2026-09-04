# %%
# select country
country = "Germany"


# select model type
model_type = 3


# packages
import re
import torch
import numpy as np
import pandas as pd
import locale
import os
import sys
sys.path.append("../Functions")
from my_functions import DST_trafo, forecast_MLP_rolling
from my_functions import reg_matrix
from my_functions import forecast_expert_ext
from my_functions import forecast_ensemble


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import joblib


import time
import random
import matplotlib.colors as mcolors


# Set CuBLAS deterministic behavior to enforce deterministic behavior for CuBLAS operations when using optuna
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# set the GPU
device = "cpu"

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



#-----------------------------------------------
#                data Preprocessing
#-------------------------------------------------
#  set language setting
locale.getlocale()

#  check the working directory
os.getcwd()

# read the data
data = pd.read_csv(f"../Data/{country}.csv")


# remove the last year od data
data = data.iloc[:-8760, :]


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


#---------------------------------------------------
#               Extract the Saved dataframes
#-------------------------------------------------------


file_optuna = f"{country}/Model{model_type}/optuna_study.pkl"
study = joblib.load(file_optuna)
# dataframe that contains information about each trial
trials_df = study.trials_dataframe()



file_step_models = f"{country}/Model{model_type}/step_models.pkl"
best_model_combo = joblib.load(file_step_models)



#--------------------------------------------------
#                Regression matrix
#-------------------------------------------------------


# Use all data including the test data
days_test = pd.to_datetime(dates_S)
dat_test = data_array


# Extrax the matrix and the nuber of column needed for indices
regmat_test = reg_matrix(
    dat_test, days_test, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[0]
columns_s = reg_matrix(
    dat_test, days_test, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[1]
columns_base = reg_matrix(
    dat_test, days_test, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[2]
columns_total = reg_matrix(
    dat_test, days_test, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
)[3]

# Remove NAs
regmat0_test = regmat_test.dropna()


# Convert DataFrame to a NumPy array first, then to a tensor
regmat_tensor_test = torch.from_numpy(regmat0_test.values).float().to(device)

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

dependent_var_test = regmat0_test.iloc[:, dependent_index]
dependent_var_tensor_test = (
    torch.from_numpy(dependent_var_test.values).float().to(device)
)
regmat_tensor_test[:, dependent_index] = 0





#-------------------------------------------------------
#    Forecatsing study for test data using ensemble of Models      
#--------------------------------------------------------



#  Define the test period length
length_test = int(3 * 365)

# The first obdervation in the evaluation period
begin_test = regmat0_test.shape[0] - length_test

length_study = length_test

# value of the dependent variables in the test data
actual_test = dependent_var_tensor_test[-length_study:, :]


weight_boa = {}


# create a tensor to stor the forecast of the models up to best model
forecast_tensor_test = torch.zeros((length_study, output_dim, len(best_model_combo)), device=device)



start_step = time.time()
(
    mae_ensemble,
    overall_mae_ensemble,
    unstandardized_y_ensemble,
    unstandardized_outputs_ensemble,
) = forecast_ensemble(
    dat_test,
    length_study,
    begin_test,
    regmat_tensor_test,
    dependent_var_tensor_test,
    regmat0_test,
    days_test,
    country,
    wd,
    price_s_lags,
    fuel_lags,
    da_lag,
    reg_names,
    dependent_index,
    active_regressor,
    batch_size,
    num_columns,
    device,
    best_model_combo,
    study,
    num_epochs_init,
    num_epochs_all,
    forecast_tensor_test,
    actual_test,
    weight_boa,
    mask_in_out_red,
    mask_in_out_full,
    model_type,
    model_types=None,
    studies=None,
    block_size=None
)

end_step = time.time()
execution_time_ensemble = end_step - start_step
print(f"Execution time: {execution_time_ensemble:.4f} seconds")




# It should be equal to the original y_test
unstandardized_y_ensemble = unstandardized_y_ensemble.cpu().numpy()

unstandarized_forecast_ensemble = unstandardized_outputs_ensemble.cpu().numpy()

# save for later use
file_unstandardized_y_ensemble = (
    f"{country}/Model{model_type}/unstandardized_actual_ensemble.pkl"
)
joblib.dump(unstandardized_y_ensemble, file_unstandardized_y_ensemble)

# save for later use
file_unstandarized_forecast_ensemble = (
    f"{country}/Model{model_type}/unstandarized_forecast_ensemble.pkl"
)
joblib.dump(unstandarized_forecast_ensemble, file_unstandarized_forecast_ensemble)

# Save execution time
file_execution_time_ensemble = (
    f"{country}/Model{model_type}/execution_time_ensemble.pkl"
)
joblib.dump(execution_time_ensemble, file_execution_time_ensemble)

#-------------------------------------------------------
#        Forecatsing study for test data using best model Only      
#--------------------------------------------------------

length_test = int(3 * 365)

# The first obdervation in the evaluation period
begin_test = regmat0_test.shape[0] - length_test

length_study = length_test

# value of the dependent variables in the test data
actual_test = dependent_var_tensor_test[-length_study:, :]


best_trial = study.best_trial

learning_rate_init = best_trial.params["learning_rate_init"]
D_init = best_trial.params["D_init"]
learning_rate_all = best_trial.params["learning_rate_all"]
D_all = best_trial.params["D_all"]
weight_decay_init = best_trial.params["weight_decay_init"]
weight_decay_all = best_trial.params["weight_decay_all"]
# weight_decay = best_trial.params["weight_decay"]
lambda_reg_all = best_trial.params["lambda_reg_all"]
lambda_reg_init = best_trial.params["lambda_reg_init"]

if model_type in {2, 3, 5, 6, 8}:
    number_neurons = best_trial.params["number_neurons"]
else:
    number_neurons = 0

if model_type in {7, 8}:
    use_ols_weights = True
    alpha = best_trial.params["alpha"]
else:
    use_ols_weights = None
    alpha = None


start_all = time.time()

(
    
    overall_agg_mae,
    unstandardized_y_test,
    unstandardized_outputs_test,
) = forecast_MLP_rolling(
    dat_eval=dat_test,
    begin_eval=begin_test,
    regmat_tensor_eval=regmat_tensor_test,
    dependent_var_tensor_eval=dependent_var_tensor_test,
    regmat0_eval=regmat0_test,
    days_eval=days_test,
    learning_rate_init=learning_rate_init,
    num_epochs_init=num_epochs_init,
    D_init=D_init,
    learning_rate_all=learning_rate_all,
    num_epochs_all=num_epochs_all,
    D_all=D_all,
    number_neurons=number_neurons,
    use_ols_weights=use_ols_weights,
    weight_decay_init=weight_decay_init,
    weight_decay_all=weight_decay_all,
    alpha=alpha,
    lambda_reg_init=lambda_reg_init,
    lambda_reg_all=lambda_reg_all,
    length_study=length_study,
    dependent_index=dependent_index,
    active_regressor=active_regressor,
    mask_in_out_red=mask_in_out_red,
    mask_in_out_full=mask_in_out_full,
    batch_size=batch_size,
    wd=wd,
    price_s_lags=price_s_lags,
    fuel_lags=fuel_lags,
    da_lag=da_lag,
    num_columns=num_columns,
    reg_names=reg_names,
    country=country,
    device=device,
    model_type=model_type)



end_all = time.time()

execution_time_best = end_all - start_all
print(f"Execution time: {execution_time_best:.4f} seconds")


# It should be equal to the original y_test
unstandardized_y_best = unstandardized_y_test.cpu().numpy()

unstandarized_forecast_best = unstandardized_outputs_test.cpu().numpy()

# save for later use
file_unstandardized_y_best = (
    f"{country}/Model{model_type}/unstandardized_actual_best.pkl"
)
joblib.dump(unstandardized_y_best, file_unstandardized_y_best)

# save for later use
file_unstandarized_forecast_best = (
    f"{country}/Model{model_type}/unstandarized_forecast_best.pkl"
)
joblib.dump(unstandarized_forecast_best, file_unstandarized_forecast_best)

# Save execution time
file_execution_time = (
    f"{country}/Model{model_type}/execution_time_best.pkl"
)
joblib.dump(execution_time_best, file_execution_time)

#

# %%
