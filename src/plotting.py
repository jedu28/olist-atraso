"""Estilo único dos gráficos (notebooks e figuras da documentação).

Cada cor tem um papel fixo no projeto inteiro: versão do modelo (V1–V4),
classe real (no prazo / atrasado) e métrica (precisão / recall / F1). As cores
categóricas seguem uma ordem validada para daltonismo — não reordene.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Paleta categórica, na ordem validada (slots 1–8).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
DEEMPHASIS = "#c3c2b7"

VERSION_COLORS = {"V1": BLUE, "V2": ORANGE, "V3": AQUA, "V4": YELLOW}
CLASS_COLORS = {0: "#8a8984", 1: RED}
METRIC_COLORS = {"Precisão": VIOLET, "Recall": MAGENTA, "F1": GREEN}
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#104281"])


def apply_style(font_size: float = 10) -> None:
    """Aplica o estilo do projeto ao matplotlib (chame uma vez no início)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": font_size * 1.1,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    })
