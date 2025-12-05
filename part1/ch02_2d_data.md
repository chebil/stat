# 2.1 Plotting 2D Data

Many datasets contain **paired measurements** - two variables measured together. Understanding the relationship between these variables requires specialized visualization techniques.

This chapter covers:
- Plotting categorical 2D data
- Time series visualization
- Spatial data and scatter plots
- Exposing relationships between variables

## 2.1.1 Categorical Data, Counts, and Charts

When both variables are categorical, we need to show how combinations of categories occur.

### Contingency Tables

A **contingency table** (or cross-tabulation) shows frequencies for combinations of two categorical variables.

#### Example: Gender and Goals

From the Chase and Danner study:

| | Sports | Grades | Popular | **Total** |
|---------|--------|--------|---------|-------|
| **Boy** | 117 | 60 | 60 | 237 |
| **Girl** | 23 | 130 | 88 | 241 |
| **Total** | 140 | 190 | 148 | 478 |

**Observations**:
- More boys choose sports (117 vs 23)
- More girls choose grades (130 vs 60)
- The variables appear related!

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create contingency table
data = {
    'Sports': [117, 23],
    'Grades': [60, 130],
    'Popular': [60, 88]
}
df = pd.DataFrame(data, index=['Boy', 'Girl'])

print("Contingency Table:")
print(df)
print(f"\nTotal: {df.sum().sum()}")
```

### Stacked Bar Charts

**Stacked bar charts** show how one categorical variable is distributed across another.

```python
# Stacked bar chart
df.T.plot(kind='bar', stacked=True, figsize=(10, 6), 
          color=['steelblue', 'pink'], edgecolor='black')
plt.xlabel('Goal')
plt.ylabel('Number of Children')
plt.title('Goals by Gender (Stacked)')
plt.legend(title='Gender')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Grouped Bar Charts

**Grouped bar charts** place bars side-by-side for easier comparison.

```python
# Grouped bar chart
df.T.plot(kind='bar', figsize=(10, 6), 
          color=['steelblue', 'pink'], edgecolor='black')
plt.xlabel('Goal')
plt.ylabel('Number of Children')
plt.title('Goals by Gender (Grouped)')
plt.legend(title='Gender')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Heatmaps

**Heatmaps** use color intensity to show frequencies in a contingency table.

```python
# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Count'}, linewidths=1)
plt.xlabel('Goal')
plt.ylabel('Gender')
plt.title('Goals by Gender (Heatmap)')
plt.tight_layout()
plt.show()
```

**Advantages of heatmaps**:
- Easy to spot high/low values
- Compact representation
- Works well with many categories

### Mosaic Plots

**Mosaic plots** show proportions using area - both width and height convey information.

```python
from statsmodels.graphics.mosaicplot import mosaic

# Prepare data for mosaic plot
data_list = []
for gender in ['Boy', 'Girl']:
    for goal in ['Sports', 'Grades', 'Popular']:
        count = df.loc[gender, goal]
        data_list.extend([(gender, goal)] * count)

# Create mosaic plot
fig, ax = plt.subplots(figsize=(10, 6))
mosaic(data_list, ax=ax, title='Goals by Gender (Mosaic Plot)')
plt.tight_layout()
plt.show()
```

## 2.1.2 Series

A **series** is a sequence of measurements taken over time. Time series data requires special visualization to show trends, cycles, and patterns.

### Line Plots

The standard visualization for time series is a **line plot** connecting consecutive measurements.

#### Example: Temperature Over Time

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulated temperature data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')

# Temperature with seasonal pattern
day_of_year = np.arange(365)
temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + \
              np.random.normal(0, 2, 365)

# Create DataFrame
df_temp = pd.DataFrame({'Date': dates, 'Temperature': temperature})

# Line plot
plt.figure(figsize=(12, 5))
plt.plot(df_temp['Date'], df_temp['Temperature'], 
         linewidth=1, alpha=0.7, label='Daily Temperature')

# Add smoothed trend
window = 30
df_temp['Smoothed'] = df_temp['Temperature'].rolling(window=window).mean()
plt.plot(df_temp['Date'], df_temp['Smoothed'], 
         linewidth=2, color='red', label=f'{window}-Day Moving Average')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Temperature Over One Year')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Multiple Series

Comparing multiple time series on the same plot reveals relationships.

```python
# Multiple series example
np.random.seed(42)
times = np.arange(100)

series1 = np.cumsum(np.random.randn(100)) + 50
series2 = np.cumsum(np.random.randn(100)) + 55
series3 = np.cumsum(np.random.randn(100)) + 45

plt.figure(figsize=(12, 5))
plt.plot(times, series1, label='Series A', linewidth=2)
plt.plot(times, series2, label='Series B', linewidth=2)
plt.plot(times, series3, label='Series C', linewidth=2)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Multiple Time Series Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Seasonal Decomposition

Time series often contain:
- **Trend**: Long-term increase/decrease
- **Seasonality**: Regular patterns that repeat
- **Noise**: Random fluctuation

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Create time series with trend and seasonality
np.random.seed(42)
t = np.arange(365)
trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)  # 30-day cycle
noise = np.random.normal(0, 2, 365)
ts = trend + seasonality + noise + 50

# Create time series object
ts_series = pd.Series(ts, index=pd.date_range('2024-01-01', periods=365))

# Decompose
decomposition = seasonal_decompose(ts_series, model='additive', period=30)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
decomposition.resid.plot(ax=axes[3], title='Residual (Noise)')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 2.1.3 Scatter Plots for Spatial Data

**Spatial data** consists of measurements with geographic coordinates (latitude, longitude).

### Basic Scatter Plot

```python
# Simulated spatial data (e.g., earthquake locations)
np.random.seed(42)
n_points = 200

# Coordinates (roughly California-like)
longitudes = np.random.uniform(-124, -114, n_points)
latitudes = np.random.uniform(32, 42, n_points)

# Magnitudes
magnitudes = np.random.exponential(2, n_points) + 1

plt.figure(figsize=(10, 8))
plt.scatter(longitudes, latitudes, s=magnitudes**2.5, 
            c=magnitudes, cmap='YlOrRd', alpha=0.6, 
            edgecolors='black', linewidths=0.5)

plt.colorbar(label='Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Earthquakes')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Point Density Maps

When points overlap heavily, show density instead.

```python
from scipy.stats import gaussian_kde

# Calculate point density
xy = np.vstack([longitudes, latitudes])
z = gaussian_kde(xy)(xy)

# Sort by density (plot high density on top)
idx = z.argsort()
longitudes_sorted = longitudes[idx]
latitudes_sorted = latitudes[idx]
z_sorted = z[idx]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(longitudes_sorted, latitudes_sorted, 
                     c=z_sorted, s=50, cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Point Density')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Density Map')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 2.1.4 Exposing Relationships with Scatter Plots

**Scatter plots** are the fundamental tool for visualizing relationships between two continuous variables.

### Basic Scatter Plot

```python
# Generate correlated data
np.random.seed(42)
n = 100

x = np.random.normal(170, 10, n)  # Heights
y = 0.5 * x + np.random.normal(0, 5, n)  # Weights (correlated)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, s=50, edgecolors='black')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Relationship Between Height and Weight')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Adding Trend Lines

```python
# Add regression line
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(x, y)
line = slope * x + intercept

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, s=50, edgecolors='black', label='Data')
plt.plot(x, line, 'r-', linewidth=2, 
         label=f'y = {slope:.2f}x + {intercept:.2f}\n$r^2$ = {r_value**2:.3f}')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight with Regression Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Color-Coded Scatter Plots

Add a third variable using color.

```python
# Add categorical variable (e.g., gender)
np.random.seed(42)
gender = np.random.choice(['Male', 'Female'], n)

colors = ['steelblue' if g == 'Male' else 'pink' for g in gender]

plt.figure(figsize=(8, 6))
for g, color, label in [('Male', 'steelblue', 'Male'), 
                         ('Female', 'pink', 'Female')]:
    mask = gender == g
    plt.scatter(x[mask], y[mask], alpha=0.6, s=50, 
                c=color, edgecolors='black', label=label)

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight by Gender')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Scatter Plot Patterns

Different patterns reveal different relationships:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
np.random.seed(42)
n = 100

# 1. Strong positive correlation
x1 = np.random.randn(n)
y1 = 2 * x1 + np.random.randn(n) * 0.5
axes[0, 0].scatter(x1, y1, alpha=0.6)
axes[0, 0].set_title(f'Strong Positive (r={np.corrcoef(x1, y1)[0,1]:.2f})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Weak positive correlation
x2 = np.random.randn(n)
y2 = 0.3 * x2 + np.random.randn(n)
axes[0, 1].scatter(x2, y2, alpha=0.6, color='orange')
axes[0, 1].set_title(f'Weak Positive (r={np.corrcoef(x2, y2)[0,1]:.2f})')
axes[0, 1].grid(True, alpha=0.3)

# 3. Strong negative correlation
x3 = np.random.randn(n)
y3 = -1.5 * x3 + np.random.randn(n) * 0.5
axes[1, 0].scatter(x3, y3, alpha=0.6, color='red')
axes[1, 0].set_title(f'Strong Negative (r={np.corrcoef(x3, y3)[0,1]:.2f})')
axes[1, 0].grid(True, alpha=0.3)

# 4. No correlation
x4 = np.random.randn(n)
y4 = np.random.randn(n)
axes[1, 1].scatter(x4, y4, alpha=0.6, color='green')
axes[1, 1].set_title(f'No Correlation (r={np.corrcoef(x4, y4)[0,1]:.2f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Bubble Charts

A **bubble chart** is a scatter plot where point size represents a third variable.

```python
# Bubble chart example
np.random.seed(42)
n = 50

heights = np.random.normal(170, 10, n)
weights = 0.5 * heights + np.random.normal(0, 5, n)
ages = np.random.randint(18, 65, n)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(heights, weights, s=ages*10, 
                     c=ages, cmap='viridis', alpha=0.6, 
                     edgecolors='black', linewidths=1)
plt.colorbar(scatter, label='Age (years)')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight vs Age (bubble size = age)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Summary

### Visualization Techniques

| Data Type | Best Visualization | Use For |
|-----------|-------------------|----------|
| **2 Categorical** | Stacked/Grouped bars, Heatmap | Comparing category combinations |
| **Time Series** | Line plot | Trends, seasonality, patterns over time |
| **Spatial (2D continuous)** | Scatter plot | Geographic patterns, distributions |
| **2 Continuous** | Scatter plot | Relationships, correlations |
| **2 Continuous + 1 Categorical** | Color-coded scatter | Group comparisons |
| **3 Continuous** | Bubble chart, 3D scatter | Multiple relationships |

### Key Takeaways

1. **Choose visualization based on data types**
2. **Scatter plots reveal relationships** between continuous variables
3. **Line plots show temporal patterns** and trends
4. **Heatmaps efficiently display** categorical combinations
5. **Color, size, and shape** can encode additional dimensions
6. **Always check for patterns** - linear, non-linear, clusters, outliers

## Practice Problems

1. Create a grouped bar chart comparing test scores across 3 classes and 4 subjects
2. Generate a time series plot showing temperature, humidity, and pressure over 1 month
3. Make a scatter plot of study hours vs exam scores with point size representing attendance
4. Create a heatmap of correlation coefficients for a multi-variable dataset

## Next Steps

→ Continue to [2.2 Correlation](ch02_correlation.md) to quantify relationships

→ Return to [Chapter 1](chapter01.md) for review of 1D visualization