
#%%-------------------------------------------------------  
#  Figure 3
#--------------------------------------------------------

import os
import sys

from calendar import day_abbr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import t


import matplotlib.patches as patches


def save_plot4_matplotlib():
    # --------- Settings & Colors ----------
    COLOR_STD_EDGE = "#2563eb"   # deep blue
    COLOR_SKIP_EDGE = "#fb7185"  # pink
    COLOR_SPEC_EDGE = "red"      # dashed overlay
    COLOR_NODE_STROKE = "#111827"

    COLOR_INPUT = "#86efac"
    COLOR_HIDDEN = "#93c5fd"
    COLOR_OUTPUT = "#fbbf24"

    NODE_SIZE = 1.05  # reduced from 1.6 to make neurons smaller
    LINE_W_STD = 1.2
    LINE_W_NODE = 1.2
    LINE_W_SPEC = 0.7
    DASH_STYLE_SPEC = (0, (5, 4))
    TEXT_SIZE = 10

    # One-line combined style for P_{d-1,23} -> P_{d,23}
    LINE_W_COMBINED_SOLID = 1.6
    LINE_W_COMBINED_DASH = 1.0
    DASH_STYLE_COMBINED = (0, (4, 3))

    # --- Node positions ---
    positions = {
        r"$P_{d-1,0}$": (-8, 3),
        r"$P_{.,0}$": (-8, 2),
        r"$Fun_{d,0}$": (-8, 1),
        r"$P_{d-1,23}$": (-8, -1),
        r"$P_{.,23}$": (-8, -2),
        r"$Fun_{d,23}$": (-8, -3),
        r"$Com_{d-2}$": (-8, -4),
        r"$Cal_{d}$": (-8, -5),
        r"$H_{1}$": (0, -4),
        r"$H_{n}$": (0, -6),
        r"$P_{d,0}$": (8, 1),
        r"$P_{d,23}$": (8, -1),
    }

    rect_nodes = {
        r"$P_{.,0}$",
        r"$P_{.,23}$",
        r"$Fun_{d,0}$",
        r"$Fun_{d,23}$",
        r"$Com_{d-2}$",
        r"$Cal_{d}$",
    }

    top_inputs = [r"$P_{d-1,0}$", r"$P_{.,0}$", r"$Fun_{d,0}$"]
    bottom_inputs = [r"$P_{d-1,23}$", r"$P_{.,23}$", r"$Fun_{d,23}$"]
    specials = [r"$Com_{d-2}$", r"$Cal_{d}$"]
    hiddens = [r"$H_{1}$", r"$H_{n}$"]
    outputs = [r"$P_{d,0}$", r"$P_{d,23}$"]

    combined_src = r"$P_{d-1,23}$"
    combined_out = r"$P_{d,23}$"

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    def draw_edge(p1, p2, color, style="-", lw=LINE_W_STD, z=1):
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            zorder=z,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linestyle=style,
                linewidth=lw,
                antialiased=True,
                capstyle="butt",
                joinstyle="miter",
            ),
        )

    # Standard: Inputs/Specials -> Hidden
    for src in top_inputs + bottom_inputs + specials:
        for h in hiddens:
            draw_edge(positions[src], positions[h], COLOR_STD_EDGE)

    # Standard: Hidden -> Output
    for h in hiddens:
        for out in outputs:
            draw_edge(positions[h], positions[out], COLOR_STD_EDGE)

    # Skip connections (exclude combined pair)
    for src in top_inputs:
        draw_edge(positions[src], positions[outputs[0]], COLOR_SKIP_EDGE)

    for src in bottom_inputs:
        if src == combined_src:
            continue
        draw_edge(positions[src], positions[outputs[1]], COLOR_SKIP_EDGE)

    # Special dashed edges (exclude combined pair)
    special_srcs = [r"$P_{d-1,23}$", r"$Com_{d-2}$", r"$Cal_{d}$"]
    for src in special_srcs:
        for out in outputs:
            if src == combined_src and out == combined_out:
                continue
            draw_edge(
                positions[src],
                positions[out],
                COLOR_SPEC_EDGE,
                style=DASH_STYLE_SPEC,
                lw=LINE_W_SPEC,
            )

    # One connection, dashed clearly on top of solid
    draw_edge(
        positions[combined_src],
        positions[combined_out],
        COLOR_SKIP_EDGE,
        style="-",
        lw=LINE_W_COMBINED_SOLID,
        z=2,
    )
    draw_edge(
        positions[combined_src],
        positions[combined_out],
        COLOR_SPEC_EDGE,
        style=DASH_STYLE_COMBINED,
        lw=LINE_W_COMBINED_DASH,
        z=3,
    )

    # --- Draw Nodes ---
    for label, (x, y) in positions.items():
        if label in outputs:
            color = COLOR_OUTPUT
        elif label in hiddens:
            color = COLOR_HIDDEN
        else:
            color = COLOR_INPUT

        if label in rect_nodes:
            ax.add_patch(
                patches.Rectangle(
                    (x - NODE_SIZE / 2, y - NODE_SIZE / 2),
                    NODE_SIZE,
                    NODE_SIZE,
                    facecolor=color,
                    edgecolor=COLOR_NODE_STROKE,
                    linewidth=LINE_W_NODE,
                    zorder=5,
                )
            )
        else:
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    NODE_SIZE / 2,
                    facecolor=color,
                    edgecolor=COLOR_NODE_STROKE,
                    linewidth=LINE_W_NODE,
                    zorder=5,
                )
            )

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=TEXT_SIZE,
            zorder=6,
            fontfamily="sans-serif",
        )

    # Vertical dots
    for dot_pos in [(-8, 0), (0, -5), (8, 0)]:
        ax.text(dot_pos[0], dot_pos[1], r"$\vdots$", fontsize=20, ha="center", va="center", zorder=6)

    # Layer labels
    ax.text(-8, 4, "Input layer", ha="center", fontsize=14, fontweight="medium")
    ax.text(0, -3.0, "Hidden layer", ha="center", fontsize=14, fontweight="medium")
    ax.text(8, 2.0, "Output layer", ha="center", fontsize=14, fontweight="medium")

    # Layout
    ax.set_xlim(-9, 9)
    ax.set_ylim(-8, 5)
    ax.axis("off")
    plt.tight_layout()

    # Save
    outdir = "Plots"
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, "plot4_mlp_skip_full.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.show()


if __name__ == "__main__":
    save_plot4_matplotlib()

# %%
