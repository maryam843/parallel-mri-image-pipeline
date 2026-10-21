"""Shared chart style for the README figures and the Part B figures (charts only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
MARKERS = ["o", "s", "D"]


def style(ax: plt.Axes, labelsize: int = 9, grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=labelsize)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def new_figure(width: float, height: float, ncols: int = 1, labelsize: int = 9, grid: bool = True):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        style(ax, labelsize, grid)
    return fig, axes[0]


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
