import numpy as np
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from typing import Optional, cast
from .base import BasePlotter

class WindPlotter(BasePlotter):

    def plot_wind_rose(
        self,
        wind_speed: np.ndarray,
        wind_direction: Optional[np.ndarray] = None,
        bins: int = 16,
        figsize: tuple[float, float] = (5, 5)
        ) -> Optional[Figure]:

        if wind_direction is None:
            print("Warning: No wind direction data. Skipping wind rose plot.")
            return None

        fig = self._new_figure(figsize)
        ax = cast(PolarAxes, fig.add_subplot(111, projection='polar'))

        theta = np.deg2rad(wind_direction)
        angles = np.linspace(0, 2 * np.pi, bins + 1)
        width = 2 * np.pi / bins

        for i in range(bins):
            mask = (theta >= angles[i]) & (theta < angles[i + 1])
            if np.any(mask):
                avg_speed = float(np.mean(wind_speed[mask]))
                ax.bar(
                    angles[i],
                    avg_speed,
                    width=width,
                    color=self.COLORS['wind'],
                    alpha=0.8
                )

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('Wind Rose')

        return fig
