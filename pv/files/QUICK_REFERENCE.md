# Plotting Module - Quick Reference Guide

## 🚀 Quick Start

```python
from plotting import RenewableEnergyPlotter

plotter = RenewableEnergyPlotter()
```

## 📊 Common Plot Types

### PV System Plots

```python
# Time series (4 panels: irradiance, temp, power, efficiency)
fig = plotter.plot_pv_timeseries(pv_results)

# Performance dashboard (6 panels with metrics)
fig = plotter.plot_pv_performance_summary(pv_results, pv_metrics)

# Power curves (power & efficiency vs irradiance)
fig = plotter.plot_pv_power_curve(pv_results)
```

### Wind Turbine Plots

```python
# Time series (wind speed & power)
fig = plotter.plot_wind_timeseries(wind_results)

# Power curve (theoretical vs actual)
power_curve = wind_model.get_power_curve()
fig = plotter.plot_wind_power_curve(power_curve, wind_results)
```

### Hybrid System Plots

```python
# Complete hybrid analysis (6 panels)
fig = plotter.plot_hybrid_system(hybrid_results)

# PV vs Wind comparison (4 panels)
fig = plotter.plot_generation_comparison(hybrid_results)
```

### Validation Plots

```python
# Metrics display
fig = plotter.plot_validation_metrics(metrics)

# Measured vs Simulated
fig = plotter.plot_measured_vs_simulated(measured, simulated)
```

## 🎨 Quick Functions

```python
from plotting import quick_plot_pv, quick_plot_wind, quick_plot_hybrid

# One-line plotting
quick_plot_pv(pv_results, pv_metrics)
quick_plot_wind(wind_results)
quick_plot_hybrid(hybrid_results)
```

## ⚙️ Common Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `results` | DataFrame | Simulation results | `pv_results` |
| `metrics` | Dict | Performance metrics | `{'total_energy_kWh': 100}` |
| `time_range` | Tuple[int, int] | Indices to plot | `(0, 72)` |
| `figsize` | Tuple[int, int] | Figure size | `(14, 10)` |

## 💾 Saving Plots

```python
# Save single plot
fig.savefig('plot.png', dpi=300, bbox_inches='tight')

# Save multiple plots
figures = [fig1, fig2, fig3]
plotter.save_all_plots(
    figures=figures,
    output_dir='./plots',
    prefix='analysis',
    format='png'
)
```

## 🎯 Common Use Cases

### 1. Quick Analysis
```python
plotter = RenewableEnergyPlotter()
quick_plot_pv(results, metrics)
plt.show()
```

### 2. Detailed Report
```python
plotter = RenewableEnergyPlotter()
fig1 = plotter.plot_pv_timeseries(results, time_range=(0, 72))
fig2 = plotter.plot_pv_performance_summary(results, metrics)
fig3 = plotter.plot_pv_power_curve(results)
plt.show()
```

### 3. Batch Processing
```python
figures = []
for dataset in datasets:
    fig = plotter.plot_pv_timeseries(dataset)
    figures.append(fig)

plotter.save_all_plots(figures, output_dir='./batch_plots')
```

## 🔧 Customization

### Colors
```python
plotter.colors['pv'] = '#FF0000'  # Change PV color to red
plotter.colors['wind'] = '#00FF00'  # Change Wind color to green
```

### Style
```python
plotter = RenewableEnergyPlotter(style='ggplot')
# Options: 'seaborn-v0_8-darkgrid', 'ggplot', 'bmh', 'default'
```

### Figure Size
```python
fig = plotter.plot_pv_timeseries(results, figsize=(20, 12))
```

## 📋 Required Data Formats

### PV Results DataFrame
```
Columns: time, irradiance_W_m2, ambient_temp_C, cell_temp_C, power_W, efficiency
```

### Wind Results DataFrame
```
Columns: time, wind_speed_m_s, power_W
```

### Hybrid Results DataFrame
```
Columns: time, irradiance_W_m2, ambient_temp_C, wind_speed_m_s, 
         pv_power_W, wind_power_W, total_power_W, pv_fraction
```

### Metrics Dictionary
```python
{
    'total_energy_kWh': float,
    'peak_power_W': float,
    'average_power_W': float,
    'capacity_factor': float,
    'max_cell_temp_C': float,  # For PV only
    'avg_efficiency': float,    # For PV only
    'min_efficiency': float     # For PV only
}
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Plots not showing | Add `plt.show()` at the end |
| Empty plots | Check data format and column names |
| Memory issues | Close figures with `plt.close(fig)` |
| Font warnings | Add `warnings.filterwarnings('ignore')` |

## 📚 Full Documentation

See `PLOTTING_DOCUMENTATION.md` for complete API reference and examples.

## 🎓 Examples

See `plotting_demo.py` for complete working examples.

---

**Color Scheme:**
- 🟠 PV: Orange-red (#FF6B35)
- 🔵 Wind: Deep blue (#004E89)
- ⚫ Total: Dark gray (#2D3142)
- 🟡 Irradiance: Yellow (#F7B801)
- 🔴 Temperature: Red (#E63946)
- 🟢 Efficiency: Green (#06A77D)
