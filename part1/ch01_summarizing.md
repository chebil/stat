# 1.3 Summarizing 1D Data

After visualizing data, we need numerical summaries to describe datasets quantitatively. This section covers fundamental descriptive statistics that allow us to characterize location (where the data is centered) and scale (how spread out the data is).

## 1.3.1 The Mean

The **mean** (or average) is the most common measure of central tendency. It represents the "center of mass" of the data.

### Definition

For a dataset $\{x\}$ of $N$ data items $x_1, \ldots, x_N$, the mean is:

$$\text{mean}(\{x\}) = \bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

### Properties of the Mean

The mean has several important mathematical properties:

**Property 1.1: Translation and Scaling**
- $\text{mean}(\{x + c\}) = \text{mean}(\{x\}) + c$
- $\text{mean}(\{kx\}) = k \cdot \text{mean}(\{x\})$

**Property: Sum of Deviations**
- The sum of deviations from the mean equals zero: $\sum_{i=1}^{N} (x_i - \bar{x}) = 0$

### Worked Example: Bar Net Worth

Ten people in a bar have the following net worths (in dollars):
100,360, 109,770, 96,860, 97,860, 108,930, 124,330, 101,300, 112,710, 106,740, 120,170

The mean net worth is:
$$\bar{x} = \frac{1,079,030}{10} = 107,903 \text{ dollars}$$

This represents a typical net worth in this group. However, as we'll see, the mean can be heavily influenced by outliers.

```python
import numpy as np

data = np.array([100360, 109770, 96860, 97860, 108930, 
                 124330, 101300, 112710, 106740, 120170])
mean = np.mean(data)
print(f"Mean net worth: ${mean:,.2f}")
# Output: Mean net worth: $107,903.00
```

## 1.3.2 Standard Deviation and Variance

### Standard Deviation

The **standard deviation** measures how spread out the data is around the mean. It tells us the typical distance of data points from the mean.

**Definition:**

$$\text{std}(\{x\}) = \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$

For the net worth data, the standard deviation is approximately $9,265.

### Properties of Standard Deviation

**Property: Translation and Scaling**
- $\text{std}(\{x + c\}) = \text{std}(\{x\})$ (adding a constant doesn't change spread)
- $\text{std}(\{kx\}) = |k| \cdot \text{std}(\{x\})$ (scaling changes spread)

**Property 1.2: Chebyshev's Bound**

For any dataset with standard deviation $\sigma$, at most $\frac{1}{k^2}$ of data points lie $k$ or more standard deviations away from the mean.

This means:
- At most 100% of data is ≥1 std dev from mean
- At most 25% of data is ≥2 std devs from mean  
- At most 11% of data is ≥3 std devs from mean

For normal data (which we'll discuss later), these bounds are much tighter:
- About 68% within 1 std dev
- About 95% within 2 std devs
- About 99% within 3 std devs

**Property 1.3: At Least One Point is One Standard Deviation Away**

$$(\text{std}(\{x\}))^2 \leq \max_i (x_i - \text{mean}(\{x\}))^2$$

This guarantees that at least one data point is at least one standard deviation away from the mean.

### Variance

The **variance** is the square of the standard deviation:

$$\text{var}(\{x\}) = \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2$$

You can think of variance as the mean-square error if you replaced each data item with the mean.

**Properties of Variance:**
- $\text{var}(\{x + c\}) = \text{var}(\{x\})$
- $\text{var}(\{kx\}) = k^2 \cdot \text{var}(\{x\})$

```python
# Calculate standard deviation and variance
std = np.std(data)
var = np.var(data)

print(f"Standard Deviation: ${std:,.2f}")
print(f"Variance: ${var:,.2f}")
print(f"\nVerification: std² = {std**2:,.2f} (should equal variance)")

# Output:
# Standard Deviation: $9,265.04
# Variance: $85,841,154.10
```

### Note on Unbiased Standard Deviation

Sometimes you'll see a slightly different formula:

$$\text{std}_{\text{unbiased}}(\{x\}) = \sqrt{\frac{\sum_i (x_i - \bar{x})^2}{N-1}}$$

This uses $N-1$ instead of $N$. This version is used when estimating the population standard deviation from a sample (covered in Chapter 6). For now, we use the $N$ version for describing the data we have.

## 1.3.3 Computing Mean and Standard Deviation Online

One useful feature of means and standard deviations is that you can estimate them **online** - you can update your estimates as new data arrives without storing all previous data.

### Online Algorithm

After seeing $k$ elements, write $\hat{\mu}_k$ for the estimated mean and $\hat{\sigma}_k$ for the estimated standard deviation.

**Mean Update:**
$$\hat{\mu}_{k+1} = \frac{k \cdot \hat{\mu}_k + x_{k+1}}{k+1}$$

**Standard Deviation Update:**
$$\hat{\sigma}_{k+1} = \sqrt{\frac{k \cdot \hat{\sigma}_k^2 + (x_{k+1} - \hat{\mu}_{k+1})^2}{k+1}}$$

This is particularly useful for streaming data or when memory is limited.

```python
class OnlineStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.var = 0.0
    
    def update(self, x):
        self.count += 1
        k = self.count
        
        # Update mean
        new_mean = ((k-1) * self.mean + x) / k
        
        # Update variance
        if k > 1:
            new_var = ((k-1) * self.var + (x - new_mean)**2) / k
            self.var = new_var
        
        self.mean = new_mean
    
    @property
    def std(self):
        return np.sqrt(self.var)

# Example usage
stats = OnlineStats()
for value in data:
    stats.update(value)

print(f"Online Mean: ${stats.mean:,.2f}")
print(f"Online Std: ${stats.std:,.2f}")
```

## 1.3.4 The Median

The **median** is an alternative measure of central tendency that is more robust to outliers than the mean.

### Definition

The median is obtained by:
1. Sorting the data points
2. Finding the point halfway along the list
3. If the list has even length, averaging the two middle values

**Examples:**
- median({3, 5, 7}) = 5
- median({3, 4, 5, 6, 7}) = 5  
- median({3, 4, 5, 6}) = (4+5)/2 = 4.5

### Median vs Mean with Outliers

Consider our bar example again. Now a billionaire walks in with net worth $1,000,000,000.

**With billionaire:**
- Mean = $91,007,184 (dramatically increased!)
- Median = $108,930 (barely changed)

The median remains an effective summary because it's not affected by extreme values. For the original data:
- Median = $107,835

The small change in median ($107,835 → $108,930$) compared to the huge change in mean shows the median's robustness.

### Properties of the Median

- $\text{median}(\{x + c\}) = \text{median}(\{x\}) + c$
- $\text{median}(\{kx\}) = k \cdot \text{median}(\{x\})$

```python
median = np.median(data)
print(f"Median net worth: ${median:,.2f}")

# Add billionaire
data_with_billionaire = np.append(data, 1000000000)
mean_with = np.mean(data_with_billionaire)
median_with = np.median(data_with_billionaire)

print(f"\nWith billionaire:")
print(f"Mean: ${mean_with:,.2f}")
print(f"Median: ${median_with:,.2f}")
print(f"\nMean changed by: ${mean_with - mean:,.2f}")
print(f"Median changed by: ${median_with - median:,.2f}")
```

## 1.3.5 Percentiles and Quartiles

### Percentiles

The **kth percentile** is the value such that k% of the data is less than or equal to that value.

We write $\text{percentile}(\{x\}, k)$ for the kth percentile.

### Quartiles

Quartiles divide the data into four equal parts:

- **First Quartile (Q1)**: 25th percentile - $\text{percentile}(\{x\}, 25)$
- **Second Quartile (Q2)**: 50th percentile (the median) - $\text{percentile}(\{x\}, 50)$
- **Third Quartile (Q3)**: 75th percentile - $\text{percentile}(\{x\}, 75)$

### Interquartile Range (IQR)

The **interquartile range** measures the spread of the middle 50% of the data:

$$\text{IQR}(\{x\}) = Q3 - Q1 = \text{percentile}(\{x\}, 75) - \text{percentile}(\{x\}, 25)$$

The IQR is robust to outliers, unlike the standard deviation.

### IQR Example with Outliers

For our net worth data:
- Without billionaire: IQR = $12,350
- With billionaire: IQR = $17,710

Compare to standard deviation:
- Without billionaire: $\sigma = 9,265$
- With billionaire: $\sigma = 301,400,000$ (completely distorted!)

### Properties of the Interquartile Range

- $\text{IQR}(\{x + c\}) = \text{IQR}(\{x\})$
- $\text{IQR}(\{kx\}) = |k| \cdot \text{IQR}(\{x\})$

```python
q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)  # median
q3 = np.percentile(data, 75)
iqr = q3 - q1

print("Quartiles:")
print(f"Q1 (25th): ${q1:,.2f}")
print(f"Q2 (50th/Median): ${q2:,.2f}")
print(f"Q3 (75th): ${q3:,.2f}")
print(f"\nInterquartile Range: ${iqr:,.2f}")

# Outlier detection bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"\nOutlier bounds: [${lower_bound:,.2f}, ${upper_bound:,.2f}]")
```

### Outlier Detection

A common rule: data items are considered outliers if they are:
- Less than $Q1 - 1.5 \times \text{IQR}$, or
- Greater than $Q3 + 1.5 \times \text{IQR}$

This is the criterion used in box plots.

## 1.3.6 Using Summaries Sensibly

### Reporting Precision

Be careful about the number of **significant figures** you report. Statistical software produces many digits, but not all are meaningful.

**Example:** Reporting "mean pregnancy length = 32.833 weeks" implies precision to ~0.001 weeks or 10 minutes. This is unrealistic given:
- People's memories are imprecise
- Medical records have limited accuracy  
- Respondents may misreport

**Better:** "mean pregnancy length ≈ 32.8 weeks"

### Categorical vs Continuous Variables

The statement "the average US family has 2.6 children" is problematic because:
- Number of children is **categorical** (discrete values)
- No family actually has 2.6 children

**Better phrasing:** "The mean of the number of children in a US family is 2.6"

**Or better yet:** Report the median and distribution for categorical data.

### When Mean vs Median?

**Use the mean when:**
- Data is roughly symmetric
- No significant outliers
- Data is continuous

**Use the median when:**
- Data is skewed
- Outliers are present
- Data is categorical/ordinal
- You want a robust measure

**Best practice:** Look at both! If they differ significantly, investigate why.

## 1.3.7 Standard Coordinates (Z-Scores)

Standard coordinates (or z-scores) allow us to compare data from different scales.

### Definition

For data item $x_i$ with dataset mean $\bar{x}$ and standard deviation $\sigma$:

$$z_i = \frac{x_i - \bar{x}}{\sigma}$$

We write $\{\tilde{x}\}$ for a dataset in standard coordinates.

### Properties

- $\text{mean}(\{\tilde{x}\}) = 0$ (always!)
- $\text{std}(\{\tilde{x}\}) = 1$ (always!)
- Unitless - can compare across different measurements

### Interpretation

A z-score tells you how many standard deviations a value is from the mean:
- $z = 0$: at the mean
- $z = 1$: one standard deviation above the mean
- $z = -2$: two standard deviations below the mean

```python
# Calculate z-scores
z_scores = (data - mean) / std

print("Z-scores:")
for i, (value, z) in enumerate(zip(data, z_scores)):
    print(f"${value:>9,}: z = {z:6.2f}")

print(f"\nMean of z-scores: {np.mean(z_scores):.10f}")
print(f"Std of z-scores: {np.std(z_scores):.10f}")
```

## Summary

### Location Parameters (Where is the data?)

| Measure | Formula | Robust? | Use When |
|---------|---------|---------|----------|
| Mean | $\frac{1}{N}\sum x_i$ | No | Symmetric data, no outliers |
| Median | Middle value | Yes | Skewed data, outliers present |

### Scale Parameters (How spread out?)

| Measure | Formula | Robust? | Use When |
|---------|---------|---------|----------|
| Std Dev | $\sqrt{\frac{1}{N}\sum(x_i-\bar{x})^2}$ | No | Normal-ish data |
| Variance | $(\text{std})^2$ | No | Mathematical convenience |
| IQR | $Q3 - Q1$ | Yes | Outliers present |

### Key Takeaways

1. **Mean vs Median**: Use median for skewed data or when outliers are present
2. **Standard Deviation**: Measures typical deviation from mean; sensitive to outliers
3. **IQR**: Robust measure of spread; good for data with outliers
4. **Z-scores**: Standardize data for comparison across different scales
5. **Precision matters**: Don't report meaningless digits
6. **Categorical data**: Prefer median and percentiles over mean

## Practice Problems

Try calculating these statistics for:

1. **Test scores**: [85, 92, 78, 90, 88, 95, 100, 72, 86, 91]
   - Mean, median, std dev, IQR
   - Compare mean vs median

2. **Income data** (deliberately skewed): [45000, 52000, 48000, 51000, 2500000, 46000, 49000]
   - Why is median better than mean here?
   - Calculate both and compare

3. **Heights** (in cm): [165, 170, 168, 172, 175, 171, 169, 173, 166, 174]
   - Convert to z-scores
   - Which heights are more than 1 std dev from mean?

## Next Steps

→ Continue to [1.4 Plots and Summaries](ch01_plots_summaries.md) to learn about combining visualizations with these numerical summaries.

→ See [Chapter 2: Looking at Relationships](chapter02.md) for analyzing relationships between variables.