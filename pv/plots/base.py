from matplotlib.figure import Figure
from matplotlib import rcParams

class BasePlotter:
    """Shared plotting utilities and IEEE-style defaults."""

    COLORS = {
        'pv': '#1f77b4',
        'wind': '#ff7f0e',
        'total': '#2ca02c',
        'irradiance': '#f1c40f',
        'temperature': '#d62728'
    }

    def __init__(self):
        self._apply_ieee_style()

    @staticmethod
    def _apply_ieee_style():
        rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'axes.labelweight': 'bold',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'legend.frameon': False,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    @staticmethod
    def _new_figure(figsize: tuple[float, float]) -> Figure:
        return Figure(figsize=figsize)
