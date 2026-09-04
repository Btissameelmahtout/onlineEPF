

#%% Load data
import numpy as np
import sys
sys.path.append("../Functions")
from my_functions import calculate_metrics, model_names
import plotly.graph_objects as go

targets = [
    "RLin",
    "MLP",
    "MLP with RLin",
    "FLin",
    "MLP with RLin (BOA)",
    "LEAR",
    "DNN",
]
indices_target = [model_names.index(name) for name in targets]
print(indices_target)

#------------------------------------------------------
#                     Figure 5
#------------------------------------------------------
country = "Germany"
_, _, _, errors = calculate_metrics(country)

shifted_indices = [i - 1 for i in indices_target]
RMSE_hourly = np.sqrt(np.mean(errors**2, axis=0))[:, shifted_indices]
model_names_hourly = np.array(model_names)[indices_target]
hours = np.arange(RMSE_hourly.shape[0])
# Create figure
fig = go.Figure()

# Add a line for each model
for i, model in enumerate(model_names_hourly):
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=RMSE_hourly[:, i],
            mode="lines+markers",
            name=model,
            hovertemplate=f"Model: {model}<br>Hour: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>",
        )
    )

# Customize layout
fig.update_layout(
    title="",
    xaxis_title="Hour of the Day",
    yaxis_title="Hourly RMSE (log scale)",
    yaxis_type="log",
    width=1000,
    height=550,
    template="plotly_white",
    font=dict(size=16, family="Arial"),  # 👈 global font size
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=14),  # 👈 legend font size
    ),
)


# Save as PDF (vector, best for papers)
fig.write_image("Plots/hourly_rmse.pdf")
fig.show()

#------------------------------------------------------
#                     Figure 11
#------------------------------------------------------
country = "Spain"
_, _, _, errors = calculate_metrics(country)

shifted_indices = [i - 1 for i in indices_target]
RMSE_hourly = np.sqrt(np.mean(errors**2, axis=0))[:, shifted_indices]
model_names_hourly = np.array(model_names)[indices_target]
hours = np.arange(RMSE_hourly.shape[0])
# Create figure
fig = go.Figure()

# Add a line for each model
for i, model in enumerate(model_names_hourly):
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=RMSE_hourly[:, i],
            mode="lines+markers",
            name=model,
            hovertemplate=f"Model: {model}<br>Hour: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>",
        )
    )

# Customize layout
fig.update_layout(
    title="",
    xaxis_title="Hour of the Day",
    yaxis_title="Hourly RMSE (log scale)",
    yaxis_type="log",
    width=1000,
    height=550,
    template="plotly_white",
    font=dict(size=16, family="Arial"),  # 👈 global font size
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=14),  # 👈 legend font size
    ),
)


# Save as PDF (vector, best for papers)
fig.write_image("Plots/hourly_rmse_spain.pdf")
fig.show()

# %%
