import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from .base import BasePlotter

class HybridPlotter(BasePlotter):

    def plot_hybrid_overview(
        self,
        results: pd.DataFrame,
        figsize: tuple[float, float] = (7, 5)
    ) -> Figure:

        fig = self._new_figure(figsize)
        gs = GridSpec(2, 1, figure=fig)

        time = np.asarray(results['time'], float)
        pv = np.asarray(results['pv_power_W'], float)
        wind = np.asarray(results['wind_power_W'], float)
        total = pv + wind

        # Power stack
        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(time, 0, pv, label='PV', color=self.COLORS['pv'])
        ax1.fill_between(time, pv, total, label='Wind', color=self.COLORS['wind'])
        ax1.plot(time, total, color=self.COLORS['total'], linewidth=1.5)
        ax1.set_ylabel('Power (W)')
        ax1.legend()

        # Fraction
        ax2 = fig.add_subplot(gs[1])
        pv_frac = pv / np.maximum(total, 1e-6)
        ax2.plot(time, pv_frac, color=self.COLORS['pv'])
        ax2.set_ylabel('PV Fraction')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylim(0.0, 1.0)

        fig.tight_layout()
        return fig
