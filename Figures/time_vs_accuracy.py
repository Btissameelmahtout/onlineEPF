#%%------------------------------------------
#                   Figure 7
#--------------------------------------------


import sys
sys.path.append("../Functions")
from my_functions import model_names, calculate_metrics, plot_pareto_germany_spain
from plotly.subplots import make_subplots
import plotly.graph_objects as go


_, maes_germany, _,_= calculate_metrics("Germany")
_, maes_spain, _,_= calculate_metrics("Spain")



import joblib
import numpy as np

country = "Germany"

best_models= np.array([
    joblib.load(
        f"../Online/{country}/Model{model_type}/execution_time_best.pkl"
    )
    for model_type in [1,2,3,4,5,7,8]
])

boa_models= np.array([
    joblib.load(
        f"../Online/{country}/Model{model_type}/execution_time_ensemble.pkl"
    )
    for model_type in [1,2,3,4,5,7,8] 
])

boa_all_time= joblib.load(f"../Online/{country}/BOA_all/execution_time_boa_all.pkl")

dnn_time= joblib.load(f"../Benchmark/{country}/execution_time_dnn.pkl")

lear_time= joblib.load(f"../Benchmark/{country}/execution_time_lear.pkl")

gam_time= joblib.load(f"../Benchmark/{country}/execution_time_GAM.pkl")
gam_time = gam_time * 60

runtimes_germany = np.concatenate([
    np.atleast_1d(best_models),
    np.atleast_1d(boa_models),
    np.atleast_1d(boa_all_time),
    np.atleast_1d(dnn_time),
    np.atleast_1d(lear_time),
    np.atleast_1d(gam_time)
])

country = "Spain"

best_models= np.array([
    joblib.load(
        f"../Online/{country}/Model{model_type}/execution_time_best.pkl"
    )
    for model_type in [1,2,3,4,5,7,8]
])

boa_models= np.array([
    joblib.load(
        f"../Online/{country}/Model{model_type}/execution_time_ensemble.pkl"
    )
    for model_type in [1,2,3,4,5,7,8] 
])

boa_all_time= joblib.load(f"../Online/{country}/BOA_all/execution_time_boa_all.pkl")

dnn_time= joblib.load(f"../Benchmark/{country}/execution_time_dnn.pkl")

lear_time= joblib.load(f"../Benchmark/{country}/execution_time_lear.pkl")

gam_time= joblib.load(f"../Benchmark/{country}/execution_time_GAM.pkl")
gam_time = gam_time * 60

runtimes_spain = np.concatenate([
    np.atleast_1d(best_models),
    np.atleast_1d(boa_models),
    np.atleast_1d(boa_all_time),
    np.atleast_1d(dnn_time),
    np.atleast_1d(lear_time),
    np.atleast_1d(gam_time)
])



fig, pareto_germany, pareto_spain = plot_pareto_germany_spain(
    maes_germany=maes_germany,
    runtimes_germany=runtimes_germany,
    maes_spain=maes_spain,
    runtimes_spain=runtimes_spain,
    model_names=model_names,
)
# %%
