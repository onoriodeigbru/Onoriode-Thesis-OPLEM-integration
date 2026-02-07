# Renewable Energy Visualization Module

A comprehensive Python visualization suite for Solar PV, Wind Turbine, and Hybrid renewable energy systems.

## 📦 Files Included

1. **plotting.py** - Main plotting module with all visualization functions
2. **plotting_demo.py** - Interactive demonstration script with examples
3. **PLOTTING_DOCUMENTATION.md** - Complete API reference and usage guide
4. **QUICK_REFERENCE.md** - Quick reference for common tasks
5. **README.md** - This file

## 🚀 Quick Start

### Installation

```bash
pip install matplotlib seaborn numpy pandas --break-system-packages
```

### Basic Usage

```python
from plotting import RenewableEnergyPlotter, quick_plot_pv
from model import SolarPVModel
import numpy as np

# Create and simulate PV system
pv_model = SolarPVModel()
irradiance = np.array([0, 200, 500, 800, 1000, 800, 500, 200, 0])
temperature = np.array([20, 22, 25, 28, 30, 28, 25, 22, 20])

results = pv_model.simulate_timeseries(irradiance, temperature)
metrics = pv_model.get_performance_metrics(results)

# Quick plot
quick_plot_pv(results, metrics)
```

## 📊 Features

### PV System Visualizations
- ✅ Time series analysis (irradiance, temperature, power, efficiency)
- ✅ Performance dashboard with key metrics
- ✅ Power curves (power & efficiency vs irradiance)
- ✅ Duration curves
- ✅ Hourly patterns
- ✅ Temperature-efficiency relationships

### Wind Turbine Visualizations
- ✅ Time series (wind speed & power)
- ✅ Power curves (theoretical vs actual)
- ✅ Wind rose diagrams (with direction data)
- ✅ Performance analysis

### Hybrid System Visualizations
- ✅ Combined PV + Wind analysis
- ✅ Stacked generation plots
- ✅ Energy contribution breakdown
- ✅ PV vs Wind comparison
- ✅ Correlation analysis
- ✅ Cumulative energy tracking

### Validation & Analysis
- ✅ Measured vs Simulated comparison
- ✅ Validation metrics display (MAE, RMSE, MAPE, R²)
- ✅ Error analysis
- ✅ Performance benchmarking

## 🎨 Plot Types Overview

| Plot Type | Function | Use Case |
|-----------|----------|----------|
| PV Time Series | `plot_pv_timeseries()` | Detailed temporal analysis |
| PV Dashboard | `plot_pv_performance_summary()` | Overview of system performance |
| Power Curves | `plot_pv_power_curve()` | Irradiance-power relationship |
| Wind Time Series | `plot_wind_timeseries()` | Wind generation analysis |
| Wind Power Curve | `plot_wind_power_curve()` | Turbine characteristics |
| Hybrid Overview | `plot_hybrid_system()` | Complete system analysis |
| Generation Comparison | `plot_generation_comparison()` | PV vs Wind comparison |
| Validation | `plot_validation_metrics()` | Model accuracy assessment |

## 📚 Documentation

### Quick Reference
See **QUICK_REFERENCE.md** for:
- Common code snippets
- Parameter reference
- Troubleshooting tips

### Full Documentation
See **PLOTTING_DOCUMENTATION.md** for:
- Complete API reference
- Detailed usage examples
- Customization options
- Best practices

### Demo Script
Run **plotting_demo.py** for:
- Interactive demonstrations
- Complete workflow examples
- Sample data generation

## 💡 Examples

### Example 1: PV Analysis
```python
from plotting import RenewableEnergyPlotter

plotter = RenewableEnergyPlotter()

# Time series plot
fig1 = plotter.plot_pv_timeseries(pv_results, time_range=(0, 72))

# Performance dashboard
fig2 = plotter.plot_pv_performance_summary(pv_results, pv_metrics)

import matplotlib.pyplot as plt
plt.show()
```

### Example 2: Hybrid System
```python
from validation import RenewableEnergySystem

# Create hybrid system
system = RenewableEnergySystem(pv_model, wind_model, 
                               n_pv_panels=10, 
                               n_wind_turbines=2)

# Simulate
results = system.simulate_system(irradiance, temperature, wind_speed)

# Plot
plotter = RenewableEnergyPlotter()
fig = plotter.plot_hybrid_system(results)
plt.show()
```

### Example 3: Save Multiple Plots
```python
# Generate multiple plots
figures = []
figures.append(plotter.plot_pv_timeseries(pv_results))
figures.append(plotter.plot_wind_timeseries(wind_results))
figures.append(plotter.plot_hybrid_system(hybrid_results))

# Save all at once
plotter.save_all_plots(
    figures=figures,
    output_dir='./analysis_plots',
    prefix='renewable_energy',
    format='png'
)
```

## 🎯 Use Cases

### 1. Research & Development
- Analyze renewable energy system performance
- Compare different configurations
- Validate simulation models
- Generate publication-quality figures

### 2. System Design
- Evaluate PV and wind resources
- Optimize hybrid system sizing
- Assess seasonal variations
- Identify performance bottlenecks

### 3. Education & Training
- Visualize renewable energy concepts
- Demonstrate system behavior
- Create teaching materials
- Interactive demonstrations

### 4. Reporting & Monitoring
- Create performance reports
- Track energy generation
- Compare actual vs predicted
- Stakeholder presentations

## 🔧 Customization

### Change Colors
```python
plotter = RenewableEnergyPlotter()
plotter.colors['pv'] = '#FF0000'      # Red
plotter.colors['wind'] = '#0000FF'    # Blue
plotter.colors['total'] = '#00FF00'   # Green
```

### Modify Styles
```python
# Available: 'seaborn-v0_8-darkgrid', 'ggplot', 'bmh', 'default'
plotter = RenewableEnergyPlotter(style='ggplot')
```

### Custom Figure Sizes
```python
fig = plotter.plot_pv_timeseries(results, figsize=(20, 12))
```

## 📋 Data Format Requirements

### PV Results DataFrame
```python
columns = ['time', 'irradiance_W_m2', 'ambient_temp_C', 
           'cell_temp_C', 'power_W', 'efficiency']
```

### Wind Results DataFrame
```python
columns = ['time', 'wind_speed_m_s', 'power_W']
```

### Hybrid Results DataFrame
```python
columns = ['time', 'irradiance_W_m2', 'ambient_temp_C', 'wind_speed_m_s',
           'pv_power_W', 'wind_power_W', 'total_power_W', 'pv_fraction']
```

## 🐛 Troubleshooting

### Plots Not Showing
```python
import matplotlib.pyplot as plt
plt.show()  # Add at the end
```

### Font Warnings
```python
import warnings
warnings.filterwarnings('ignore')
```

### Memory Issues
```python
import matplotlib.pyplot as plt
plt.close('all')  # Close all figures
```

### Import Errors
Make sure all files are in the same directory or add to path:
```python
import sys
sys.path.insert(0, '/path/to/modules')
```

## 🎓 Running the Demo

```bash
# Interactive demo with all plot types
python plotting_demo.py

# Or in Python:
from plotting_demo import demo_complete_workflow
figures = demo_complete_workflow()

import matplotlib.pyplot as plt
plt.show()
```

## 📊 Output Examples

The plotting module generates:
- **High-resolution figures** (300+ DPI)
- **Publication-quality plots**
- **Customizable layouts**
- **Professional color schemes**
- **Clear labeling and legends**

## 🔗 Integration

Works seamlessly with:
- ✅ NREL data (via `data_nrel.py`)
- ✅ PVGIS data (via `data_pvgis.py`)
- ✅ Custom simulation models (via `model.py`)
- ✅ Validation framework (via `validation.py`)
- ✅ OPLEM platform (via `OPLEM.py`)

## 💾 Saving Options

```python
# Single plot - PNG
fig.savefig('plot.png', dpi=300, bbox_inches='tight')

# Single plot - PDF (vector)
fig.savefig('plot.pdf', format='pdf', bbox_inches='tight')

# Multiple plots
plotter.save_all_plots(figures, output_dir='./plots', format='png')
```

## 🎨 Color Scheme

- **PV**: Orange-red (#FF6B35)
- **Wind**: Deep blue (#004E89)
- **Total**: Dark gray (#2D3142)
- **Irradiance**: Yellow (#F7B801)
- **Temperature**: Red (#E63946)
- **Efficiency**: Green (#06A77D)

## 📈 Performance Tips

1. **Large Datasets**: Use `time_range` parameter to plot subsets
2. **Memory**: Close figures after saving with `plt.close(fig)`
3. **Speed**: Use Agg backend for batch processing: `matplotlib.use('Agg')`
4. **Quality**: Set DPI to 600 for publications: `plt.rcParams['savefig.dpi'] = 600`

## 🆘 Support

For help:
1. Check **QUICK_REFERENCE.md** for common tasks
2. Review **PLOTTING_DOCUMENTATION.md** for detailed API
3. Run **plotting_demo.py** for working examples
4. Examine source code docstrings in **plotting.py**

## ✅ Testing

All plot types have been tested and verified. Run the test:
```python
# See plotting_demo.py for test examples
```

## 📝 Version

**Version 1.0** - Initial release with complete PV, Wind, and Hybrid visualization suite

## 🙏 Credits

Built for renewable energy system analysis and OPLEM platform integration.

---

**Happy Plotting! 📊🌞💨**
