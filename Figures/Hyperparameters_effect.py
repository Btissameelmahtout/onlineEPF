

#%% ------------------------------------------------------------
#                      Figure 8
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def pareto_front_min(df, x="wall_time_sec", y="overall_agg_mae"):
    pts = df[[x, y]].to_numpy()
    is_pareto = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        if not is_pareto[i]:
            continue
        dominated = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            is_pareto[i] = False

    return is_pareto



def plot_panel(ax, file_path, label_column, title):

    df = pd.read_csv(file_path)
    df["pareto"] = pareto_front_min(df)

    marker_size = 30

    # Non-Pareto
    ax.scatter(
        df.loc[~df["pareto"], "wall_time_sec"],
        df.loc[~df["pareto"], "overall_agg_mae"],
        s=marker_size,
        c="#e5c07b",
        alpha=0.8,
        zorder=2,
    )

    # Pareto
    ax.scatter(
        df.loc[df["pareto"], "wall_time_sec"],
        df.loc[df["pareto"], "overall_agg_mae"],
        s=marker_size,
        c="#2563eb",
        alpha=0.9,
        zorder=4,
    )

    # Labels
    for _, r in df.iterrows():
        if r["pareto"]:
            xytext = (-2, -2)
            ha, va = "right", "top"
            color = "#2563eb"
        else:
            xytext = (2, 2)
            ha, va = "left", "bottom"
            color = "#e5c07b"

        ax.annotate(
            int(r[label_column]),
            (r["wall_time_sec"], r["overall_agg_mae"]),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=12,
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel("Time in seconds", fontsize=12)
    ax.set_ylabel("Overall MAE", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)

    # Minimal look
    for spine in ax.spines.values():
        spine.set_visible(False)



model_type = 3
country = "Germany"
base_path = f"Plots"

fig, axes = plt.subplots(2, 2, figsize=(11, 8))  # ← no sharex/sharey

plot_panel(
    axes[0, 1],
    f"{base_path}/D_all_grid_results.csv",
    "D_all",
    "Window size (update)",
)

plot_panel(
    axes[0, 0],
    f"{base_path}/D_init_grid_results.csv",
    "D_init",
    "Window (initial)",
)

plot_panel(
    axes[1, 0],
    f"{base_path}/num_epochs_init_grid_results.csv",
    "num_epochs_init",
    "Epochs (initial)",
)

plot_panel(
    axes[1, 1],
    f"{base_path}/num_epochs_all_grid_results.csv",
    "num_epochs_all",
    "Epochs (update)",
)


plt.tight_layout()

plt.savefig(f"Plots/{country}_combined_grid_search.pdf", bbox_inches="tight")

plt.show()

# %%
