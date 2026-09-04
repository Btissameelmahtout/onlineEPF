#%%
# from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sys
sys.path.append("../Functions")
from my_functions import model_names, calculate_metrics
from my_functions import normalize_values, fmt_min_colored, colored_cell
from pathlib import Path
import torch

#%%----------------------------------------------------------
#                      Table 1
#------------------------------------------------------------

rmse_germany, mae_germany, rmae_germany,_ = calculate_metrics("Germany")
rmse_spain, mae_spain, rmae_spain,_ = calculate_metrics("Spain")

# Ensure arrays are 1D
rmse = np.asarray(rmse_germany).flatten()
mae = np.asarray(mae_germany).flatten()
rMAE = np.asarray(rmae_germany).flatten()
rmse_spain = np.asarray(rmse_spain).flatten()
mae_spain = np.asarray(mae_spain).flatten()
rMAE_spain = np.asarray(rmae_spain).flatten()

# Keep vectors aligned
models = list(model_names[1:])
n = min(len(models), len(rmse), len(mae), len(rMAE), len(rmse_spain), len(mae_spain), len(rMAE_spain))
models = models[:n]

results = pd.DataFrame(
    {
        "RMSE_Germany": rmse[:n],
        "MAE_Germany": mae[:n],
        "rMAE_Germany": rMAE[:n],
        "RMSE_Spain": rmse_spain[:n],
        "MAE_Spain": mae_spain[:n],
        "rMAE_Spain": rMAE_spain[:n],
    },
    index=models,
)

# Normalize each metric column separately
norm_rmse_germany = normalize_values(results["RMSE_Germany"].values)
norm_mae_germany = normalize_values(results["MAE_Germany"].values)
norm_rmae_germany = normalize_values(results["rMAE_Germany"].values)

norm_rmse_spain = normalize_values(results["RMSE_Spain"].values)
norm_mae_spain = normalize_values(results["MAE_Spain"].values)
norm_rmae_spain = normalize_values(results["rMAE_Spain"].values)

# Best (min) and worst (max) indices per column
min_rmse_ger_idx = int(np.argmin(results["RMSE_Germany"].values))
min_mae_ger_idx = int(np.argmin(results["MAE_Germany"].values))
min_rmae_ger_idx = int(np.argmin(results["rMAE_Germany"].values))
min_rmse_spa_idx = int(np.argmin(results["RMSE_Spain"].values))
min_mae_spa_idx = int(np.argmin(results["MAE_Spain"].values))
min_rmae_spa_idx = int(np.argmin(results["rMAE_Spain"].values))

max_rmse_ger_idx = int(np.argmax(results["RMSE_Germany"].values))
max_mae_ger_idx = int(np.argmax(results["MAE_Germany"].values))
max_rmae_ger_idx = int(np.argmax(results["rMAE_Germany"].values))
max_rmse_spa_idx = int(np.argmax(results["RMSE_Spain"].values))
max_mae_spa_idx = int(np.argmax(results["MAE_Spain"].values))
max_rmae_spa_idx = int(np.argmax(results["rMAE_Spain"].values))

# Build colored cells
colored_data = []
for idx in range(len(results)):
    row = [
        colored_cell(
            results.iloc[idx, 0], norm_rmse_germany[idx],
            is_best=(idx == min_rmse_ger_idx), is_worst=(idx == max_rmse_ger_idx), as_percent=False
        ),
        colored_cell(
            results.iloc[idx, 1], norm_mae_germany[idx],
            is_best=(idx == min_mae_ger_idx), is_worst=(idx == max_mae_ger_idx), as_percent=False
        ),
        colored_cell(
            results.iloc[idx, 2], norm_rmae_germany[idx],
            is_best=(idx == min_rmae_ger_idx), is_worst=(idx == max_rmae_ger_idx), as_percent=True
        ),
        colored_cell(
            results.iloc[idx, 3], norm_rmse_spain[idx],
            is_best=(idx == min_rmse_spa_idx), is_worst=(idx == max_rmse_spa_idx), as_percent=False
        ),
        colored_cell(
            results.iloc[idx, 4], norm_mae_spain[idx],
            is_best=(idx == min_mae_spa_idx), is_worst=(idx == max_mae_spa_idx), as_percent=False
        ),
        colored_cell(
            results.iloc[idx, 5], norm_rmae_spain[idx],
            is_best=(idx == min_rmae_spa_idx), is_worst=(idx == max_rmae_spa_idx), as_percent=True
        ),
    ]
    colored_data.append(row)

colored_table = pd.DataFrame(
    colored_data,
    index=results.index,
    columns=["RMSE_Germany", "MAE_Germany", "rMAE_Germany", "RMSE_Spain", "MAE_Spain", "rMAE_Spain"],
)

# Grouped column headers
colored_table.columns = pd.MultiIndex.from_arrays(
    [
        ["Germany", "Germany", "Germany", "Spain", "Spain", "Spain"],
        ["RMSE", "MAE", "rMAE", "RMSE", "MAE", "rMAE"],
    ]
)

latex_table = colored_table.to_latex(
    escape=False,
    caption="Comparative Forecast Accuracy in terms of RMSE (EUR/MWh), MAE (EUR/MWh) and rMAE of Competing Models. For each metric, the cell colors range from red (indicating poorer performance) to blue (indicating better performance). The best performing models are highlighted in bold.",
    label="tab:model_results",
    multicolumn=True,
    multicolumn_format="c",
)

# Small lines under Germany and Spain
latex_table = latex_table.replace(
    "& RMSE & MAE & rMAE & RMSE & MAE & rMAE \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n & RMSE & MAE & rMAE & RMSE & MAE & rMAE \\\\",
)

# Separators after first 7 and next 8 models
if len(models) > 7:
    latex_table = latex_table.replace(models[7], f"\\hline\n{models[7]}", 1)
if len(models) > 15:
    latex_table = latex_table.replace(models[15], f"\\hline\n{models[15]}", 1)

print(latex_table)


#%%----------------------------------------------------------
#                      Table 2
#------------------------------------------------------------

Forcast_model3_best = "../Online/Germany/Model3/unstandarized_forecast_best.pkl"
unstandarized_forecast_best_loaded3 = joblib.load(Forcast_model3_best)


Forcast_model3_best_10 = "../No_Online/10/Germany/Model3/unstandarized_forecast_best.pkl"
unstandarized_forecast_best_loaded3_10 = joblib.load(Forcast_model3_best_10)

Forcast_model3_best_60 = "../No_Online/60/Germany/Model3/unstandarized_forecast_best.pkl"
unstandarized_forecast_best_loaded3_60 = joblib.load(Forcast_model3_best_60)

Forcast_model3_best_num_epochs_tun = (
    "../No_Online/num_epochs_tun/Germany/Model3/unstandarized_forecast_best.pkl"
)
unstandarized_forecast_best_loaded3_num_epochs_tun = joblib.load(
    Forcast_model3_best_num_epochs_tun
)

true = "../No_Online/10/Germany/Model3/unstandardized_actual_best.pkl"
true_loaded = joblib.load(true)


# Test period
length_study = 2 * 365
output_dim = 24


# Create a new array with 3 slices along the third axis
forecast_new = np.zeros((length_study, output_dim, 5))

forecast_new[:, :, 0] = true_loaded
forecast_new[:, :, 1] = unstandarized_forecast_best_loaded3
forecast_new[:, :, 2] = unstandarized_forecast_best_loaded3_10
forecast_new[:, :, 3] = unstandarized_forecast_best_loaded3_60
forecast_new[:, :, 4] = unstandarized_forecast_best_loaded3_num_epochs_tun


model_names_new = [
    "true",
    "MLP with RLin",
    "MLP with RLin (Initial)",
    "MLP with RLin (Update)",
    "MLP with RLin (Tuned)",
]


errors_no_online = forecast_new[..., 1:] - forecast_new[..., :1]



rmse_no_online = np.sqrt((errors_no_online**2).mean(axis=(0, 1)))
mae_no_online = (np.abs(errors_no_online)).mean(axis=(0, 1))



country = "Germany"
online_time = joblib.load(f"../Online/{country}/Model3/execution_time_best.pkl")
no_online_initial_time = joblib.load(f"../No_Online/10/{country}/Model3/execution_time_best.pkl")
no_online_update_time = joblib.load(f"../No_Online/60/{country}/Model3/execution_time_best.pkl")
tuned_time= joblib.load(f"../No_Online/num_epochs_tun/{country}/Model3/execution_time_best.pkl")


runtimes_NO_online = np.concatenate([
    np.atleast_1d(online_time),
    np.atleast_1d(no_online_initial_time),
    np.atleast_1d(no_online_update_time),
    np.atleast_1d(tuned_time),

])

model_names = model_names_new[1:]


df_table = pd.DataFrame(
    {
        "Model": model_names,
        "MAE": mae_no_online,
        "RMSE": rmse_no_online,
        "Runtime (sec)": runtimes_NO_online,
    }
)

# Sort by MAE (best first)
df_table = df_table.sort_values("MAE").reset_index(drop=True)

# Round values
df_table["MAE"] = df_table["MAE"].round(2)
df_table["RMSE"] = df_table["RMSE"].round(2)
df_table["Runtime (sec)"] = df_table["Runtime (sec)"].round(2)

# Normalize each metric column separately
norm_mae = normalize_values(df_table["MAE"].values)
norm_rmse = normalize_values(df_table["RMSE"].values)
norm_runtime = normalize_values(df_table["Runtime (sec)"].values)

# Find minimum values
min_mae = df_table["MAE"].min()
min_rmse = df_table["RMSE"].min()
min_runtime = df_table["Runtime (sec)"].min()

# Build colored table
colored_data = []
for idx in range(len(df_table)):
    row = [
        df_table.iloc[idx, 0],  # Model
        fmt_min_colored(df_table.iloc[idx, 1], min_mae, norm_mae[idx]),
        fmt_min_colored(df_table.iloc[idx, 2], min_rmse, norm_rmse[idx]),
        fmt_min_colored(df_table.iloc[idx, 3], min_runtime, norm_runtime[idx]),  # <-- bold min runtime too
    ]
    colored_data.append(row)


colored_df = pd.DataFrame(
    colored_data,
    columns=["Model", "MAE", "RMSE", "Runtime (sec)"],
)

latex_table = colored_df.to_latex(
    index=False,
    escape=False,
    column_format="lccc",
    caption="Model performance comparison.",
    label="tab:model_performance",
    longtable=False,
)

print(latex_table)

 
# %%--------------------------------------------------
#                      Table 3
#-----------------------------------------------------


# load forcast for model 3
Forcast_model3_best = "../Crisis/Germany/Model3/unstandarized_forecast_best.pkl"
unstandarized_forecast_best_loaded3 = joblib.load(Forcast_model3_best)


true = "../Crisis/Germany/Model3/unstandardized_actual_best.pkl"
true_loaded = joblib.load(true)


Forcast_model3_ensemble = (
    "../Crisis/Germany/Model3/unstandarized_forecast_ensemble.pkl"
)
unstandarized_forecast_ensemble_loaded3 = joblib.load(Forcast_model3_ensemble)


forecast_lear = pd.read_csv(
    "../Benchmark/crisis/experimental_files/LEAR_forecast_datmy_data_YT2_CW364.csv",
    index_col=False,
)
forecast_lear_tensor = torch.tensor(forecast_lear.iloc[:, 1:].values)


forecast_DNN = pd.read_csv(
    "../Benchmark/crisis/experimental_files/DNN_forecast_nl2_datmy_data_YT2_SFH0_CW1_1.csv",
    index_col=False,
)
forecast_DNN_tensor = torch.tensor(forecast_DNN.iloc[:, 1:].values)

forecast_gam = pd.read_csv("../Benchmark/Germany_forecast_crisis_results.csv")


# Test period
length_study = 3 * 365
output_dim = 24


# Create a new array with 3 slices along the third axis
forecast_new = np.zeros((length_study, output_dim, 6))

forecast_new[:, :, 0] = true_loaded
forecast_new[:, :, 1] = unstandarized_forecast_best_loaded3
forecast_new[:, :, 2] = unstandarized_forecast_ensemble_loaded3
forecast_new[:, :, 3] = forecast_lear_tensor
forecast_new[:, :, 4] = forecast_DNN_tensor
forecast_new[:, :, 5] = forecast_gam


model_names_new = [
    "true",
    "MLP with RLin",
    "MLP with RLin (BOA)",
    "LEAR",
    "DNN",
    "GAM Online"
]


errors_germany = forecast_new[..., 1:] - forecast_new[..., :1]
rmse = np.sqrt((errors_germany**2).mean(axis=(0, 1)))
mae = (np.abs(errors_germany)).mean(axis=(0, 1))

# 2021-01-16 to 2021-08-31
errors_germany1 = errors_germany[0:228, :, :]

# 2021-09-01 to 2023-01-15
errors_germany2 = errors_germany[228:730, :, :]

# 2023-01-16 to 2024-01-15
errors_germany3 = errors_germany[730:, :, :]

rmse1 = np.sqrt((errors_germany1**2).mean(axis=(0, 1)))
mae1 = (np.abs(errors_germany1)).mean(axis=(0, 1))

rmse2 = np.sqrt((errors_germany2**2).mean(axis=(0, 1)))
mae2 = (np.abs(errors_germany2)).mean(axis=(0, 1))

rmse3 = np.sqrt((errors_germany3**2).mean(axis=(0, 1)))
mae3 = (np.abs(errors_germany3)).mean(axis=(0, 1))


# Columns across models (rows: MLP+RLin, MLP+RLin(BOA), LEAR, DNN)
mae_pre = [mae1[0], mae1[1], mae1[2], mae1[3], mae1[4]]
mae_cri = [mae2[0], mae2[1], mae2[2], mae2[3], mae2[4]]
mae_pos = [mae3[0], mae3[1], mae3[2], mae3[3], mae3[4]]

rmse_pre = [rmse1[0], rmse1[1], rmse1[2], rmse1[3], rmse1[4]]
rmse_cri = [rmse2[0], rmse2[1], rmse2[2], rmse2[3], rmse2[4]]
rmse_pos = [rmse3[0], rmse3[1], rmse3[2], rmse3[3], rmse3[4]]

min_mae_pre, min_mae_cri, min_mae_pos = min(mae_pre), min(mae_cri), min(mae_pos)
min_rmse_pre, min_rmse_cri, min_rmse_pos = min(rmse_pre), min(rmse_cri), min(rmse_pos)

# Normalize each metric separately for coloring
norm_mae_pre = normalize_values(np.array(mae_pre))
norm_mae_cri = normalize_values(np.array(mae_cri))
norm_mae_pos = normalize_values(np.array(mae_pos))

norm_rmse_pre = normalize_values(np.array(rmse_pre))
norm_rmse_cri = normalize_values(np.array(rmse_cri))
norm_rmse_pos = normalize_values(np.array(rmse_pos))

latex_table = r"""
\begin{table}[!ht]
\centering
\caption{RMSE and MAE for the three evaluation periods in the German–Luxembourg market. Pre-crisis period spans 2021-01-16 to 2021-08-31, Crisis period covers 2021-09-01 to 2023-01-15, and Post-crisis period corresponds to 2023-01-16 to 2024-01-15.}
\label{tab:germany_period_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{MAE} & \multicolumn{3}{c}{RMSE} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Model
& Pre-crisis & Crisis & Post-crisis
& Pre-crisis & Crisis & Post-crisis \\
\midrule
MLP with RLin
& %s & %s & %s
& %s & %s & %s \\

MLP with RLin (BOA)
& %s & %s & %s
& %s & %s & %s \\

LEAR
& %s & %s & %s
& %s & %s & %s \\

DNN
& %s & %s & %s
& %s & %s & %s \\

GAM Online
& %s & %s & %s
& %s & %s & %s \\
\bottomrule
\end{tabular}
}
\end{table}
""" % (
    # Row 1 (MLP with RLin)
    fmt_min_colored(mae1[0], min_mae_pre, norm_mae_pre[0]),
    fmt_min_colored(mae2[0], min_mae_cri, norm_mae_cri[0]),
    fmt_min_colored(mae3[0], min_mae_pos, norm_mae_pos[0]),
    fmt_min_colored(rmse1[0], min_rmse_pre, norm_rmse_pre[0]),
    fmt_min_colored(rmse2[0], min_rmse_cri, norm_rmse_cri[0]),
    fmt_min_colored(rmse3[0], min_rmse_pos, norm_rmse_pos[0]),
    # Row 2 (MLP with RLin BOA)
    fmt_min_colored(mae1[1], min_mae_pre, norm_mae_pre[1]),
    fmt_min_colored(mae2[1], min_mae_cri, norm_mae_cri[1]),
    fmt_min_colored(mae3[1], min_mae_pos, norm_mae_pos[1]),
    fmt_min_colored(rmse1[1], min_rmse_pre, norm_rmse_pre[1]),
    fmt_min_colored(rmse2[1], min_rmse_cri, norm_rmse_cri[1]),
    fmt_min_colored(rmse3[1], min_rmse_pos, norm_rmse_pos[1]),
    # Row 3 (LEAR)
    fmt_min_colored(mae1[2], min_mae_pre, norm_mae_pre[2]),
    fmt_min_colored(mae2[2], min_mae_cri, norm_mae_cri[2]),
    fmt_min_colored(mae3[2], min_mae_pos, norm_mae_pos[2]),
    fmt_min_colored(rmse1[2], min_rmse_pre, norm_rmse_pre[2]),
    fmt_min_colored(rmse2[2], min_rmse_cri, norm_rmse_cri[2]),
    fmt_min_colored(rmse3[2], min_rmse_pos, norm_rmse_pos[2]),
    # Row 4 (DNN)
    fmt_min_colored(mae1[3], min_mae_pre, norm_mae_pre[3]),
    fmt_min_colored(mae2[3], min_mae_cri, norm_mae_cri[3]),
    fmt_min_colored(mae3[3], min_mae_pos, norm_mae_pos[3]),
    fmt_min_colored(rmse1[3], min_rmse_pre, norm_rmse_pre[3]),
    fmt_min_colored(rmse2[3], min_rmse_cri, norm_rmse_cri[3]),
    fmt_min_colored(rmse3[3], min_rmse_pos, norm_rmse_pos[3]),
    # Row 5
    fmt_min_colored(mae1[4], min_mae_pre, norm_mae_pre[4]),
    fmt_min_colored(mae2[4], min_mae_cri, norm_mae_cri[4]),
    fmt_min_colored(mae3[4], min_mae_pos, norm_mae_pos[4]),
    fmt_min_colored(rmse1[4], min_rmse_pre, norm_rmse_pre[4]),
    fmt_min_colored(rmse2[4], min_rmse_cri, norm_rmse_cri[4]),
    fmt_min_colored(rmse3[4], min_rmse_pos, norm_rmse_pos[4]),
)

print(latex_table)


# %%
