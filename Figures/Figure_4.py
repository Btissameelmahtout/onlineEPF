
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_forecast_experiments_with_curved_arrow2():
    """
    Rolling forecast experiments:
      – 4 rows of squares & circles (N=1..4)
      – Blue dashed arrows = transferred weights (initialization)
      – Legend with all elements
    """
    # Define rolling windows
    windows = {
        1: {"used": range(1, 11), "forecast": 11},
        2: {"used": range(7, 12), "forecast": 12},
        3: {"used": range(8, 13), "forecast": 13},
        4: {"used": range(9, 14), "forecast": 14},
    }
    # Vertical positions for rows
    row_levels = {1: 2.5, 2: 2, 3: 1.5, 4: 1}

    fig, ax = plt.subplots(figsize=(10, 3.2))

    # Plot each experiment row
    for n in windows:
        used_times = windows[n]["used"]
        forecast_time = windows[n]["forecast"]
        row_y = row_levels[n]

        # (a) Available data = green squares
        for t in range(1, forecast_time + 1):
            ax.scatter(
                t, row_y, s=500, marker="s", facecolors="lightblue", edgecolors="black"
            )

        # (b) Used training data (row 1 = red, others = orange)
        for t in used_times:
            color = "#86efac" if n in [2, 3, 4] else "red"
            ax.scatter(
                t, row_y, s=200, marker="o", facecolors=color, edgecolors="black"
            )

        # (c) Forecast point = gray square
        ax.scatter(
            forecast_time,
            row_y,
            s=500,
            marker="s",
            facecolors="gray",
            edgecolors="black",
        )

    # Add curved arrows for transferred weights
    arrows = [
        ((11, 2.5), (7, 2)),  # from N=1 → N=2
        ((12, 2), (8, 1.5)),  # from N=2 → N=3
        ((13, 1.5), (9, 1)),  # from N=3 → N=4
    ]
    for (x0, y0), (x1, y1) in arrows:
        arrow = mpatches.FancyArrowPatch(
            posA=(x0, y0),
            posB=(x1, y1),
            connectionstyle="arc3,rad=0.3",
            arrowstyle="Simple, head_width=6, head_length=12, tail_width=1",
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        ax.add_patch(arrow)

    # Axes formatting
    ax.set_xlim(0.5, 15)
    ax.set_ylim(0.5, 3.0)
    ax.set_xticks(range(1, 15))
    ax.set_xlabel("Time")
    ax.set_yticks([1, 1.5, 2, 2.5])
    ax.set_yticklabels(["N=4", "N=3", "N=2", "N=1"])

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor="lightblue", edgecolor="black", label="Available data"
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Initial training data",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=10,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Training data (updated windows)",
            markerfacecolor="#86efac",
            markeredgecolor="black",
            markersize=10,
            linestyle="None",
        ),
        mpatches.Patch(facecolor="gray", edgecolor="black", label="Data to forecast"),
        plt.Line2D(
            [0, 1],
            [0, 0],
            color="blue",
            linestyle="--",
            marker=r"$\rightarrow$",
            markersize=12,
            label="Transferred weights",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=9.5,
    )

    plt.tight_layout()
    plt.savefig("Plots/rolling_weight.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    draw_forecast_experiments_with_curved_arrow2()

# %%
