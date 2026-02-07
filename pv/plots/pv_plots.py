import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pv.plots.base import BasePlotter

class PVPlotter(BasePlotter):

    def plot_power_time_series(
        self,
        results: pd.DataFrame,
        figsize: tuple[float, float] = (6, 3)
    ) -> Figure:
        fig = self._new_figure(figsize)
        ax = fig.add_subplot(111)

        time = np.asarray(results['time'], float)
        power = np.asarray(results['pv_power_W'], float)

        ax.plot(time, power, color=self.COLORS['pv'], linewidth=1.5)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('PV Power (W)')
        ax.set_title('PV Power Output')

        fig.tight_layout()
        return fig
