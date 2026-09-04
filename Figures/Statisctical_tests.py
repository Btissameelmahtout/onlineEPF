#%%
import sys
sys.path.append("../Functions")
from my_functions import plot_adf_stationarity, calculate_metrics, plot_dm_heatmap, model_names
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pandas as pd




#%%----------------------------------------------------
#                      Figure 6
#----------------------------------------------------
country = "Germany"
_, _, _, errors = calculate_metrics(country)
loss="L1"

fig, dm_results = plot_dm_heatmap(
    loss=loss,
    errors=errors,
    model_names=model_names,
    country=country,
)

#%%--------------------------------------------------
#                      Figure 12
#----------------------------------------------------
country = "Spain"
_, _, _, errors = calculate_metrics(country)
loss="L1"

fig, dm_results = plot_dm_heatmap(
    loss=loss,
    errors=errors,
    model_names=model_names,
    country=country,
)

#%%----------------------------------------------------
#                      Figure 13
#----------------------------------------------------
country = "Germany"
_, _, _, errors = calculate_metrics(country)
loss="L2"

fig, dm_results = plot_dm_heatmap(
    loss=loss,
    errors=errors,
    model_names=model_names,
    country=country,
)



#%%----------------------------------------------------
#                      Figure 14
#----------------------------------------------------
country = "Spain"
_, _, _, errors = calculate_metrics(country)
loss="L2"

fig, dm_results = plot_dm_heatmap(
    loss=loss,
    errors=errors,
    model_names=model_names,
    country=country,
)

#%%--------------------------------------------------
#                      Figure 15
#----------------------------------------------------

country = "Germany"
_, _, _, errors = calculate_metrics(country)
loss="L1"

fig = plot_adf_stationarity(
    model_names=model_names,
    country=country,
    loss=loss,
    errors=errors,
)

#%%----------------------------------------------------
#                      Figure 16
#----------------------------------------------------

country = "Germany"
_, _, _, errors = calculate_metrics(country)
loss="L2"

fig = plot_adf_stationarity(
    model_names=model_names,
    country=country,
    loss=loss,
    errors=errors,
)


#%%----------------------------------------------------
#                      Figure 17
#----------------------------------------------------

country = "Spain"
_, _, _, errors = calculate_metrics(country)
loss="L1"

fig = plot_adf_stationarity(
    model_names=model_names,
    country=country,
    loss=loss,
    errors=errors,
)

#%%----------------------------------------------------
#                      Figure 18
#----------------------------------------------------

country = "Spain"
_, _, _, errors = calculate_metrics(country)
loss="L2"

fig = plot_adf_stationarity(
    model_names=model_names,
    country=country,
    loss=loss,
    errors=errors,
)
# %%
