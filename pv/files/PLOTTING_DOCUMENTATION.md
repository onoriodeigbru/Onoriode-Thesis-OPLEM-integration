# Renewable Energy Plotting Module Documentation

## Overview

The `plotting.py` module provides comprehensive visualization capabilities for renewable energy systems including Solar PV, Wind Turbines, and Hybrid systems. It offers publication-quality plots for analysis, validation, and reporting.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Main Class: RenewableEnergyPlotter](#renewableenergyplotter)
4. [Plot Types](#plot-types)
5. [Usage Examples](#usage-examples)
6. [Customization](#customization)
7. [API Reference](#api-reference)

---

## Installation

### Required Dependencies

```python
pip install matplotlib seaborn numpy pandas --break-system-packages
```

### Import the Module

```python
from plotting import RenewableEnergyPlotter, quick_plot_pv, quick_plot_wind, quick_plot_hybrid
```

---

## Quick Start

### Basic Usage - PV System

```python
from model import SolarPVModel
from plotting import quick_plot_pv
import numpy as np

# Create model and simulate
pv_model = SolarPVModel()
irradiance = np.array([0, 200, 500, 800, 1000, 800, 500, 200, 0])
temperature = np.array([20, 22, 25, 28, 30, 28, 25, 22, 20])

results = pv_model.simulate_timeseries(irradiance, temperature)
metrics = pv_model.get_performance_metrics(results)

# Quick plot
quick_plot_pv(results, metrics)
```

### Basic Usage - Wind Turbine

```python
from model import WindTurbineModel
from plotting import quick_plot_wind

# Create model and simulate
wind_model = WindTurbineModel()
wind_speed = np.array([0, 5, 10, 15, 12, 8, 5, 3, 0])

results = wind_model.simulate_timeseries(wind_speed)

# Quick plot
quick_plot_wind(results)
```

### Basic Usage - Hybrid System

```python
from validation import RenewableEnergySystem
from plotting import quick_plot_hybrid

# Create hybrid system
system = RenewableEnergySystem(pv_model, wind_model, n_pv_panels=10, n_wind_turbines=2)

# Simulate
results = system.simulate_system(irradiance, temperature, wind_speed)

# Quick plot
quick_plot_hybrid(results)
```

---

## RenewableEnergyPlotter

Main class providing all plotting functionality.

### Initialization

```python
plotter = RenewableEnergyPlotter(style='seaborn-v0_8-darkgrid')
```

**Parameters:**
- `style` (str): Matplotlib style name. Options: 'seaborn-v0_8-darkgrid', 'ggplot', 'bmh', 'default'

### Color Scheme

The plotter uses a predefined color palette:
- **PV**: Orange-red (#FF6B35)
- **Wind**: Deep blue (#004E89)
- **Total**: Dark gray (#2D3142)
- **Irradiance**: Yellow (#F7B801)
- **Temperature**: Red (#E63946)
- **Efficiency**: Green (#06A77D)

---

## Plot Types

### 1. PV System Plots

#### A. Time Series Plot

Comprehensive 4-panel time series showing irradiance, temperature, power, and efficiency.

```python
fig = plotter.plot_pv_timeseries(
    results=pv_results,
    time_range=(0, 72),  # First 72 hours
    figsize=(14, 10)
)
plt.show()
```

**Panels:**
1. Solar irradiance (W/m²)
2. Ambient and cell temperature (°C)
3. Power output (W)
4. System efficiency (%)

#### B. Performance Summary Dashboard

Multi-panel dashboard with key performance indicators.

```python
fig = plotter.plot_pv_performance_summary(
    results=pv_results,
    metrics=pv_metrics,
    figsize=(15, 8)
)
```

**Panels:**
1. Power duration curve
2. Temperature vs efficiency scatter (colored by irradiance)
3. Hourly average generation
4. Power distribution histogram
5. Performance metrics summary box

#### C. Power Curves

Relationship between irradiance, power, and efficiency.

```python
fig = plotter.plot_pv_power_curve(
    results=pv_results,
    figsize=(12, 5)
)
```

**Panels:**
1. Power vs Irradiance (colored by cell temperature)
2. Efficiency vs Irradiance (colored by cell temperature)

---

### 2. Wind Turbine Plots

#### A. Time Series Plot

Wind speed and power output over time.

```python
fig = plotter.plot_wind_timeseries(
    results=wind_results,
    time_range=(0, 168),  # One week
    figsize=(14, 6)
)
```

**Panels:**
1. Wind speed (m/s)
2. Power output (W)

#### B. Power Curve

Theoretical and actual power curve comparison.

```python
power_curve = wind_model.get_power_curve()

fig = plotter.plot_wind_power_curve(
    power_curve_df=power_curve,
    actual_results=wind_results,
    figsize=(10, 6)
)
```

Shows:
- Theoretical power curve (line)
- Actual performance data points (scatter)
- Cut-in, rated, and cut-out speeds

#### C. Wind Rose (Optional)

Directional wind distribution (requires wind direction data).

```python
fig = plotter.plot_wind_rose(
    wind_speed=wind_speed_array,
    wind_direction=wind_direction_array,  # In degrees
    bins=16,
    figsize=(10, 10)
)
```

---

### 3. Hybrid System Plots

#### A. Hybrid System Overview

Comprehensive analysis of combined PV and Wind system.

```python
fig = plotter.plot_hybrid_system(
    results=hybrid_results,
    time_range=(0, 72),
    figsize=(15, 10)
)
```

**Panels:**
1. Stacked power generation (PV + Wind + Total)
2. Environmental conditions (Irradiance + Temperature)
3. Wind speed
4. Energy contribution pie chart
5. Power fraction over time
6. System statistics summary

#### B. Generation Comparison

Detailed comparison of PV vs Wind performance.

```python
fig = plotter.plot_generation_comparison(
    results=hybrid_results,
    figsize=(14, 8)
)
```

**Panels:**
1. Side-by-side power comparison
2. PV vs Wind correlation scatter
3. Hourly average generation comparison
4. Cumulative energy production

---

### 4. Validation Plots

#### A. Validation Metrics Display

Visual summary of validation metrics.

```python
metrics = {
    'MAE': 15.3,
    'RMSE': 22.1,
    'MAPE': 8.5,
    'R²': 0.95,
    'max_error': 45.2,
    'mean_error': 2.1,
    'n_samples': 1000
}

fig = plotter.plot_validation_metrics(
    metrics=metrics,
    figsize=(10, 6)
)
```

#### B. Measured vs Simulated Comparison

Time series and scatter plot comparison.

```python
fig = plotter.plot_measured_vs_simulated(
    measured=measured_data,
    simulated=simulated_data,
    title="Model Validation",
    figsize=(12, 5)
)
```

**Panels:**
1. Time series overlay
2. Measured vs Simulated scatter with perfect fit line

---

## Usage Examples

### Example 1: Complete PV Analysis

```python
import numpy as np
from model import SolarPVModel, PVParameters
from plotting import RenewableEnergyPlotter

# Setup
params = PVParameters(eta_ref=0.20, beta=0.0045, rated_power=300)
pv_model = SolarPVModel(params)

# Generate one week of data
hours = 168
time = np.arange(hours)
hour_of_day = time % 24

# Realistic solar pattern
irradiance = np.maximum(0, 1000 * np.sin(np.pi * (hour_of_day - 6) / 12))
temperature = 25 + 5 * np.sin(2 * np.pi * time / 24)

# Simulate
results = pv_model.simulate_timeseries(irradiance, temperature, timestamps=time)
metrics = pv_model.get_performance_metrics(results)

# Plot
plotter = RenewableEnergyPlotter()

# All PV plots
fig1 = plotter.plot_pv_timeseries(results, time_range=(0, 72))
fig2 = plotter.plot_pv_performance_summary(results, metrics)
fig3 = plotter.plot_pv_power_curve(results)

plt.show()
```

### Example 2: Hybrid System Analysis

```python
from model import SolarPVModel, WindTurbineModel
from validation import RenewableEnergySystem
from plotting import RenewableEnergyPlotter

# Create models
pv_model = SolarPVModel()
wind_model = WindTurbineModel()

# Create system (10 PV panels, 2 wind turbines)
system = RenewableEnergySystem(pv_model, wind_model, n_pv_panels=10, n_wind_turbines=2)

# Simulate (using previously generated data)
results = system.simulate_system(irradiance, temperature, wind_speed)

# Get capacity info
capacity = system.get_system_capacity()
print(f"Total Capacity: {capacity['total_capacity_W']} W")

# Plot
plotter = RenewableEnergyPlotter()
fig1 = plotter.plot_hybrid_system(results)
fig2 = plotter.plot_generation_comparison(results)

plt.show()
```

### Example 3: Validation with Real Data

```python
from data_nrel import NRELDataFetcher
from validation import PVModelValidator
from plotting import RenewableEnergyPlotter

# Fetch real data
fetcher = NRELDataFetcher(api_key='YOUR_API_KEY')
nrel_data = fetcher.fetch_data(latitude=40.0, longitude=-105.0, year=2020)

# Validate model
validator = PVModelValidator(pv_model)
validation_results = validator.validate_with_nrel(nrel_data, sample_size=1000)

# Extract results
results = validation_results['results']
metrics = validation_results['metrics']

# Plot validation
plotter = RenewableEnergyPlotter()

# If you have measured power data
measured_power = nrel_data['measured_power'][:1000]  # Example
simulated_power = results['power_W'].values

fig = plotter.plot_measured_vs_simulated(
    measured=measured_power,
    simulated=simulated_power,
    title="NREL Data Validation"
)

plt.show()
```

### Example 4: Save All Plots

```python
# Generate all plots
figures = []

# PV plots
figures.append(plotter.plot_pv_timeseries(pv_results))
figures.append(plotter.plot_pv_performance_summary(pv_results, pv_metrics))

# Wind plots
figures.append(plotter.plot_wind_timeseries(wind_results))

# Hybrid plots
figures.append(plotter.plot_hybrid_system(hybrid_results))

# Save all
plotter.save_all_plots(
    figures=figures,
    output_dir='./analysis_plots',
    prefix='renewable_energy',
    format='png'  # or 'pdf', 'svg'
)
```

---

## Customization

### Change Color Scheme

```python
plotter = RenewableEnergyPlotter()

# Modify colors
plotter.colors['pv'] = '#FF0000'  # Red PV
plotter.colors['wind'] = '#0000FF'  # Blue Wind

# Use in plots
fig = plotter.plot_hybrid_system(results)
```

### Custom Figure Size

```python
# All plot functions accept figsize parameter
fig = plotter.plot_pv_timeseries(
    results=pv_results,
    figsize=(20, 12)  # Width x Height in inches
)
```

### Custom Time Range

```python
# Plot only specific time period
fig = plotter.plot_pv_timeseries(
    results=pv_results,
    time_range=(48, 96)  # Hours 48-96 (day 3)
)
```

### Custom Style

```python
import matplotlib.pyplot as plt

# Available styles
print(plt.style.available)

# Use different style
plotter = RenewableEnergyPlotter(style='ggplot')
# or
plotter = RenewableEnergyPlotter(style='bmh')
```

---

## API Reference

### RenewableEnergyPlotter Methods

#### PV Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `plot_pv_timeseries()` | 4-panel time series analysis | matplotlib.Figure |
| `plot_pv_performance_summary()` | Performance dashboard | matplotlib.Figure |
| `plot_pv_power_curve()` | Power vs irradiance curves | matplotlib.Figure |

#### Wind Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `plot_wind_timeseries()` | Wind speed and power time series | matplotlib.Figure |
| `plot_wind_power_curve()` | Theoretical and actual power curve | matplotlib.Figure |
| `plot_wind_rose()` | Directional wind distribution | matplotlib.Figure |

#### Hybrid Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `plot_hybrid_system()` | Complete hybrid system analysis | matplotlib.Figure |
| `plot_generation_comparison()` | PV vs Wind comparison | matplotlib.Figure |

#### Validation Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `plot_validation_metrics()` | Display validation metrics | matplotlib.Figure |
| `plot_measured_vs_simulated()` | Comparison plots | matplotlib.Figure |

#### Utility Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `save_all_plots()` | Save multiple figures | None |
| `create_report()` | Generate complete report | None |

---

## Tips and Best Practices

### 1. Data Preparation

- Ensure your data is properly formatted as pandas DataFrames
- Use consistent time steps
- Handle missing data before plotting

### 2. Performance

- For large datasets (>10,000 points), consider:
  - Using `time_range` to plot subsets
  - Downsampling data for overview plots
  - Saving plots instead of displaying all at once

### 3. Publication Quality

```python
# High DPI for publications
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 600

# Save as vector format
fig.savefig('plot.pdf', format='pdf', bbox_inches='tight')
```

### 4. Interactive Analysis

```python
# Use interactive backend
%matplotlib widget  # In Jupyter

# Or
import matplotlib
matplotlib.use('TkAgg')
```

---

## Troubleshooting

### Issue: Plots not showing

```python
import matplotlib.pyplot as plt
plt.show()  # Add this after creating plots
```

### Issue: Font warnings

```python
import warnings
warnings.filterwarnings('ignore')
```

### Issue: Memory issues with many plots

```python
# Close figures after saving
fig.savefig('plot.png')
plt.close(fig)
```

---

## Examples of Complete Workflows

See `plotting_demo.py` for complete working examples including:
- PV system analysis
- Wind turbine analysis
- Hybrid system analysis
- Validation workflows
- Batch plot generation

---

## Support

For issues or questions:
1. Check the demo script: `plotting_demo.py`
2. Review this documentation
3. Examine the source code docstrings

---

## Version History

- **v1.0** (2024): Initial release with PV, Wind, and Hybrid plotting capabilities

---

## License

This module is part of the Renewable Energy Modeling Suite.
