#%%
import plotly.express as px
import pandas as pd

# %%------------------------------------------
#                 Figure 1
#----------------------------------------------

# read the data
country = "Germany"
data = pd.read_csv(f"../Data/{country}.csv")
fig = px.line(
    data,
    x="time_utc",
    y="Price",
    labels={"time_utc": "Time", "Price": "Electricity Price (€/MWh)"},
)

# Thin line styling
fig.update_traces(
    line=dict(color="royalblue", width=0.8),
    hovertemplate="Time=%{x}<br>Price=%{y:.2f} €",
)

fig.update_layout(
    template="plotly_white",
    title=dict(x=0.5, xanchor="center", font=dict(size=22, color="darkblue")),
    xaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        title_font=dict(size=22),
        tickfont=dict(size=22),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        title_font=dict(size=22),
        tickfont=dict(size=22),
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    width=1200,
    height=700,
    font=dict(size=22),  # smaller general font
)

# Save as PDF (vector format)
fig.write_image("Plots/Price.pdf", width=1200, height=700)
fig.show()

# %%------------------------------------------
#                 Figure 2
#----------------------------------------------


import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd

# ---------- Prepare commodities long format ----------
name_map = {
    "Coal": "Coal (€/t)",
    "NGas": "Ngas (€/MWh)",
    "Oil": "Oil (€/bbl)",
    "EUA": "EUA (€/tCO₂)",
}
df_long = data.melt(
    id_vars="time_utc",
    value_vars=list(name_map.keys()),
    var_name="Variable",
    value_name="Value",
)
df_long["Variable"] = df_long["Variable"].map(name_map)
df_long["Value"] = df_long["Value"].clip(lower=1e-6)

# ---------- Build combined figure: 5 rows ----------
fig = sp.make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.18, 0.18, 0.18, 0.18, 0.28],
)

# 1) Fundamentals (thin lines)
fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["Load_DA"],
        name="Load_DA",
        line=dict(color="blue", width=0.8),
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["Solar_DA"],
        name="Solar_DA",
        line=dict(color="orange", width=0.8),
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["WindOn_DA"],
        name="WindOn_DA",
        line=dict(color="green", width=0.8),
        showlegend=False,
    ),
    row=3,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["WindOff_DA"],
        name="WindOff_DA",
        line=dict(color="purple", width=0.8),
        showlegend=False,
    ),
    row=4,
    col=1,
)

# 2) Commodities (log axis + legend)
color_map = {
    "Coal (€/t)": "#636EFA",
    "Ngas (€/MWh)": "#EF553B",
    "Oil (€/bbl)": "#00CC96",
    "EUA (€/tCO₂)": "#AB63FA",
}

for var, grp in df_long.groupby("Variable", sort=False):
    fig.add_trace(
        go.Scatter(
            x=grp["time_utc"],
            y=grp["Value"],
            name=var,
            mode="lines",
            line=dict(width=2, color=color_map.get(var, None)),
            showlegend=True,
            legendgroup="commodities",
        ),
        row=5,
        col=1,
    )

# ---------- Axes ----------
fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
fig.update_yaxes(title_text="Solar (MW)", row=2, col=1)
fig.update_yaxes(title_text="WindOn (MW)", row=3, col=1)
fig.update_yaxes(title_text="WindOff (MW)", row=4, col=1)
fig.update_yaxes(
    title_text="Commodity Prices (log scale)",
    row=5,
    col=1,
    type="log",
    dtick="D2",
    tickformat=".0f",
)
fig.update_xaxes(title_text="Time", row=5, col=1)

# ---------- Layout styling ----------
fig.update_layout(
    template="plotly_white",
    height=1000,
    width=1200,
    margin=dict(t=40, b=50, l=70, r=20),
    font=dict(size=14),
    hovermode="x unified",
    plot_bgcolor="white",
    # 🟩 Legend just above commodities subplot
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.135,  # adjust upward/downward to fine-tune position
        xanchor="center",
        x=0.3,
        title=None,
        bgcolor="rgba(255,255,255,0.7)",
    ),
)

# Light grids for clarity
fig.update_xaxes(showgrid=True, gridcolor="lightgray")
fig.update_yaxes(showgrid=True, gridcolor="lightgray")

# Save & show
fig.write_image(
    "Plots/combined_fundamentals_commodities.pdf", width=1200, height=1000, scale=1
)
fig.show()



# %%------------------------------------------
#                 Figure 9
#----------------------------------------------

# read the data
country = "Spain"
data = pd.read_csv(f"../Data/{country}.csv")
fig = px.line(
    data,
    x="time_utc",
    y="Price",
    labels={"time_utc": "Time", "Price": "Electricity Price (€/MWh)"},
)

# Thin line styling
fig.update_traces(
    line=dict(color="royalblue", width=0.8),
    hovertemplate="Time=%{x}<br>Price=%{y:.2f} €",
)

fig.update_layout(
    template="plotly_white",
    title=dict(x=0.5, xanchor="center", font=dict(size=22, color="darkblue")),
    xaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        title_font=dict(size=22),
        tickfont=dict(size=22),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        title_font=dict(size=22),
        tickfont=dict(size=22),
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    width=1200,
    height=700,
    font=dict(size=22),  # smaller general font
)

# Save as PDF (vector format)
fig.write_image("Plots/Price_spain.pdf", width=1200, height=700)
fig.show()

# %%------------------------------------------
#                 Figure 10
#----------------------------------------------

# ---------- Prepare commodities long format ----------
name_map = {
    "Coal": "Coal (€/t)",
    "NGas": "Ngas (€/MWh)",
    "Oil": "Oil (€/bbl)",
    "EUA": "EUA (€/tCO₂)",
}
df_long = data.melt(
    id_vars="time_utc",
    value_vars=list(name_map.keys()),
    var_name="Variable",
    value_name="Value",
)
df_long["Variable"] = df_long["Variable"].map(name_map)
df_long["Value"] = df_long["Value"].clip(lower=1e-6)

# ---------- Build combined figure: 5 rows ----------
fig = sp.make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.2, 0.2, 0.2, 0.4],
)
# and move commodities to row=4 instead of row=5


# 1) Fundamentals (thin lines)
fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["Load_DA"],
        name="Load_DA",
        line=dict(color="blue", width=0.8),
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["Solar_DA"],
        name="Solar_DA",
        line=dict(color="orange", width=0.8),
        showlegend=False,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data["time_utc"],
        y=data["WindOn_DA"],
        name="WindOn_DA",
        line=dict(color="green", width=0.8),
        showlegend=False,
    ),
    row=3,
    col=1,
)


color_map = {
    "Coal (€/t)": "#636EFA",
    "Ngas (€/MWh)": "#EF553B",
    "Oil (€/bbl)": "#00CC96",
    "EUA (€/tCO₂)": "#AB63FA",
}

for var, grp in df_long.groupby("Variable", sort=False):
    fig.add_trace(
        go.Scatter(
            x=grp["time_utc"],
            y=grp["Value"],
            name=var,
            mode="lines",
            line=dict(width=2, color=color_map.get(var, None)),
            showlegend=True,
            legendgroup="commodities",
        ),
        row=4,
        col=1,
    )

# ---------- Axes ----------
fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
fig.update_yaxes(title_text="Solar (MW)", row=2, col=1)
fig.update_yaxes(title_text="WindOn (MW)", row=3, col=1)
# fig.update_yaxes(title_text="WindOff (MW)", row=4, col=1)
fig.update_yaxes(
    title_text="Commodity Prices (log scale)",
    row=4,
    col=1,
    type="log",
    dtick="D2",
    tickformat=".0f",
)
fig.update_xaxes(title_text="Time", row=5, col=1)

# ---------- Layout styling ----------
fig.update_layout(
    template="plotly_white",
    height=1000,
    width=1200,
    margin=dict(t=40, b=50, l=70, r=20),
    font=dict(size=14),
    hovermode="x unified",
    plot_bgcolor="white",
    # 🟩 Legend just above commodities subplot
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.23,  # adjust upward/downward to fine-tune position
        xanchor="center",
        x=0.3,
        title=None,
        bgcolor="rgba(255,255,255,0.7)",
    ),
)

# Light grids for clarity
fig.update_xaxes(showgrid=True, gridcolor="lightgray")
fig.update_yaxes(showgrid=True, gridcolor="lightgray")

# Save & show
fig.write_image(
    "Plots/combined_fundamentals_commodities_spain.pdf",
    width=1200,
    height=1000,
    scale=1,
)
fig.show()


# %%
