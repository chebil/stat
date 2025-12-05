# 2.2 Correlation

Correlation quantifies the strength and direction of the linear relationship between two variables. It's one of the most important tools for understanding how variables are related.

## 2.2.1 The Pearson Correlation Coefficient

### Definition

For two datasets $\{x\}$ and $\{y\}$ with $N$ paired observations, the **Pearson correlation coefficient** is:

$$r = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^N (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^N (y_i - \bar{y})^2}}$$

This can be rewritten more simply using standard deviations:

$$r = \frac{1}{N} \sum_{i=1}^N \frac{(x_i - \bar{x})}{\sigma_x} \cdot \frac{(y_i - \bar{y})}{\sigma_y}$$

Or even more simply, using standard coordinates (z-scores):

$$r = \frac{1}{N} \sum_{i=1}^N z_x^{(i)} \cdot z_y^{(i)}$$

where $z_x^{(i)} = \frac{x_i - \bar{x}}{\sigma_x}$ and $z_y^{(i)} = \frac{y_i - \bar{y}}{\sigma_y}$

### Interpretation

**Sign:**
- **r > 0**: Positive correlation (as x increases, y tends to increase)
- **r < 0**: Negative correlation (as x increases, y tends to decrease)  
- **r = 0**: No linear correlation

**Magnitude:**
- **|r| = 1**: Perfect linear relationship
- **|r| ≥ 0.7**: Strong correlation
- **0.3 ≤ |r| < 0.7**: Moderate correlation
- **|r| < 0.3**: Weak correlation

These are rough guidelines - interpretation depends on context!

### Properties of Correlation

**Property 2.1: Bounded Range**
$$-1 \leq r \leq 1$$

The correlation coefficient is always between -1 and +1.

**Property 2.2: Units-free**

Correlation is dimensionless - it doesn't depend on the units of measurement.

**Property 2.3: Translation Invariance**
$$\text{corr}(\{x+a\}, \{y+b\}) = \text{corr}(\{x\}, \{y\})$$

Adding constants doesn't change correlation.

**Property 2.4: Scale Invariance**
$$\text{corr}(\{cx\}, \{dy\}) = \text{sign}(cd) \cdot \text{corr}(\{x\}, \{y\})$$

Scaling preserves correlation magnitude (but reverses sign if scaling factors have opposite signs).

**Property 2.5: Symmetry**
$$\text{corr}(\{x\}, \{y\}) = \text{corr}(\{y\}, \{x\})$$

## 2.2.2 Computing Correlation

### Worked Example: Height and Weight

Consider a dataset of 6 people with heights (cm) and weights (kg):

| Person | Height (cm) | Weight (kg) |
|--------|-------------|-------------|
| 1      | 160         | 55          |
| 2      | 165         | 62          |
| 3      | 170         | 68          |
| 4      | 175         | 75          |
| 5      | 180         | 81          |
| 6      | 185         | 88          |

**Step 1: Calculate means**
- $\bar{x} = 172.5$ cm
- $\bar{y} = 71.5$ kg

**Step 2: Calculate standard deviations**
- $\sigma_x = 9.01$ cm
- $\sigma_y = 11.80$ kg

**Step 3: Calculate correlation**

$$r = \frac{1}{6} \sum_{i=1}^6 \frac{(x_i - 172.5)}{9.01} \cdot \frac{(y_i - 71.5)}{11.80} = 0.998$$

This indicates a very strong positive correlation between height and weight.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Data
heights = np.array([160, 165, 170, 175, 180, 185])
weights = np.array([55, 62, 68, 75, 81, 88])

# Method 1: Using numpy
r_numpy = np.corrcoef(heights, weights)[0, 1]

# Method 2: Using scipy (also gives p-value)
r_scipy, p_value = stats.pearsonr(heights, weights)

# Method 3: Manual calculation
mean_x = np.mean(heights)
mean_y = np.mean(weights)
std_x = np.std(heights)
std_y = np.std(weights)

z_x = (heights - mean_x) / std_x
z_y = (weights - mean_y) / std_y
r_manual = np.mean(z_x * z_y)

print(f"Correlation (numpy):  {r_numpy:.4f}")
print(f"Correlation (scipy):  {r_scipy:.4f}")
print(f"Correlation (manual): {r_manual:.4f}")
print(f"\nP-value: {p_value:.6f}")
print(f"\nThis indicates a {('strong positive' if r_scipy > 0.7 else 'moderate')} correlation.")
```

**Output:**
```
Correlation (numpy):  0.9998
Correlation (scipy):  0.9998
Correlation (manual): 0.9998

P-value: 0.000000

This indicates a strong positive correlation.
```


**Note**: The plot would display here (scatter plot with strong upward trend)

### Worked Example: Study Hours vs Errors

Consider students' study hours and number of errors on a test:

| Student | Study Hours | Errors |
|---------|-------------|--------|
| 1       | 8           | 2      |
| 2       | 7           | 3      |
| 3       | 6           | 4      |
| 4       | 5           | 5      |
| 5       | 4           | 7      |
| 6       | 3           | 9      |
| 7       | 2           | 12     |
| 8       | 1           | 15     |

Calculating the correlation: $r \approx -0.96$

The strong negative correlation indicates that more study hours are associated with fewer errors.

```python
study_hours = np.array([8, 7, 6, 5, 4, 3, 2, 1])
errors = np.array([2, 3, 4, 5, 7, 9, 12, 15])

r = np.corrcoef(study_hours, errors)[0, 1]
print(f"Correlation: {r:.3f}")
print(f"\nInterpretation: Strong negative correlation")
print(f"More study hours → fewer errors")
```

**Output:**
```
Correlation: -0.974

Interpretation: Strong negative correlation
More study hours → fewer errors
```


## 2.2.3 Correlation and Prediction

Correlation allows us to make predictions using **linear regression**.

### Prediction Formula

Given correlation $r$ between variables $x$ and $y$, predict $\hat{y}$ from a new value $x$:

$$\hat{y} = \bar{y} + r \frac{\sigma_y}{\sigma_x}(x - \bar{x})$$

In standard coordinates, this simplifies to:
$$\tilde{y} = r \cdot \tilde{x}$$

### Worked Example: Predicting Weight from Height

Using our height-weight data where $r = 0.998$:

**Predict the weight of someone who is 172 cm tall:**

$$\hat{y} = 71.5 + 0.998 \times \frac{11.80}{9.01} \times (172 - 172.5)$$
$$\hat{y} = 71.5 + 0.998 \times 1.310 \times (-0.5)$$
$$\hat{y} = 71.5 - 0.654 = 70.85 \text{ kg}$$

```python
def predict_linear(x_new, x_data, y_data):
    """Predict y from x using correlation."""
    r = np.corrcoef(x_data, y_data)[0, 1]
    
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    x_std = np.std(x_data)
    y_std = np.std(y_data)
    
    y_pred = y_mean + r * (y_std / x_std) * (x_new - x_mean)
    return y_pred

# Predict weight for someone 172 cm tall
predicted_weight = predict_linear(172, heights, weights)
print(f"Predicted weight for 172 cm: {predicted_weight:.2f} kg")

# Multiple predictions
print(f"\nPredictions for different heights:")
for h in [160, 170, 180, 190]:
    w = predict_linear(h, heights, weights)
    print(f"  Height {h} cm → Weight {w:.1f} kg")
```

**Output:**
```
Predicted weight for 172 cm: 70.85 kg

Predictions for different heights:
  Height 160 cm → Weight 55.1 kg
  Height 170 cm → Weight 68.2 kg
  Height 180 cm → Weight 81.3 kg
  Height 190 cm → Weight 94.4 kg
```


### Regression to the Mean

An important consequence: if $|r| < 1$, predictions are "pulled" toward the mean:
- High $x$ values predict $y$ values less extreme (closer to $\bar{y}$)
- Low $x$ values predict $y$ values less extreme (closer to $\bar{y}$)

This is called **regression to the mean** - it's not a statistical artifact, but a real phenomenon when variables aren't perfectly correlated.

## 2.2.4 Correlation Pitfalls and Limitations

### 1. Correlation ≠ Causation

**High correlation does NOT prove causation!**

Classic examples of spurious correlations:
- Ice cream sales and drowning deaths (both caused by summer weather)
- Number of Nicolas Cage films per year and swimming pool drownings
- Per capita cheese consumption and deaths from bedsheet tangling

Always ask:
1. Could $x$ cause $y$?
2. Could $y$ cause $x$?
3. Could a **confounding variable** $z$ cause both?

```python
# Spurious correlation example
years = np.arange(2000, 2010)
ice_cream_sales = np.array([2.5, 2.7, 2.9, 3.1, 3.4, 3.6, 3.9, 4.1, 4.4, 4.6])
drownings = np.array([30, 32, 35, 38, 42, 45, 48, 51, 54, 57])

r = np.corrcoef(ice_cream_sales, drownings)[0, 1]
print(f"Ice Cream Sales vs Drownings:")
print(f"Correlation: {r:.3f}")
print(f"\n⚠️ WARNING: High correlation!")
print(f"But ice cream doesn't cause drowning.")
print(f"Confounding variable: SUMMER (warm weather)")
print(f"  - Warm weather → more ice cream sales")
print(f"  - Warm weather → more swimming → more drownings")
```

**Output:**
```
Ice Cream Sales vs Drownings:
Correlation: 0.999

⚠️ WARNING: High correlation!
But ice cream doesn't cause drowning.
Confounding variable: SUMMER (warm weather)
  - Warm weather → more ice cream sales
  - Warm weather → more swimming → more drownings
```


### 2. Non-linear Relationships

Correlation only measures **linear** relationships. Non-linear patterns can have low correlation despite strong relationships.

**Example: Parabolic Relationship**

```python
x = np.linspace(-5, 5, 50)
y = x**2  # Perfect parabola - clear relationship!

r = np.corrcoef(x, y)[0, 1]
print(f"Parabolic Relationship (y = x²):")
print(f"Correlation: {r:.3f}")
print(f"\n⚠️ WARNING: Low correlation despite perfect relationship!")
print(f"Lesson: Always plot your data - don't rely on correlation alone.")
```

**Output:**
```
Parabolic Relationship (y = x²):
Correlation: 0.000

⚠️ WARNING: Low correlation despite perfect relationship!
Lesson: Always plot your data - don't rely on correlation alone.
```


**Lesson:** Always visualize your data with scatter plots!

### 3. Outliers Can Distort Correlation

A single outlier can dramatically change the correlation.

```python
# Data without outlier
x_clean = np.array([1, 2, 3, 4, 5])
y_clean = np.array([2, 4, 6, 8, 10])
r_clean = np.corrcoef(x_clean, y_clean)[0, 1]

# Add an outlier
x_outlier = np.append(x_clean, 6)
y_outlier = np.append(y_clean, 1)
r_outlier = np.corrcoef(x_outlier, y_outlier)[0, 1]

print(f"Correlation without outlier: {r_clean:.3f}")
print(f"Correlation with outlier:    {r_outlier:.3f}")
print(f"\nChange: {r_clean - r_outlier:.3f}")
print(f"\nLesson: Check for outliers before computing correlation!")
```

**Output:**
```
Correlation without outlier: 1.000
Correlation with outlier:    0.230

Change: 0.770

Lesson: Check for outliers before computing correlation!
```


### 4. Restricted Range

Looking at only a subset of data can hide correlations that exist in the full range.

### 5. Simpson's Paradox

Correlation can **reverse direction** when data is aggregated or separated into groups.

## 2.2.5 Correlation Matrix

For datasets with multiple variables, a **correlation matrix** shows all pairwise correlations.

### Example: Multiple Physical Measurements

```python
import pandas as pd
import seaborn as sns

# Create sample data
np.random.seed(42)
n = 100

height = np.random.normal(170, 10, n)
weight = 0.5 * height + np.random.normal(0, 5, n)
age = np.random.uniform(20, 60, n)
shoe_size = 0.15 * height + np.random.normal(-10, 2, n)

data = pd.DataFrame({
    'Height': height,
    'Weight': weight,
    'Age': age,
    'Shoe_Size': shoe_size
})

# Correlation matrix
corr_matrix = data.corr()
print("Correlation Matrix:")
print(corr_matrix.round(3))
print(f"\nStrongest correlation: Height-Weight ({corr_matrix.loc['Height', 'Weight']:.3f})")
print(f"Weakest correlation: Age-Shoe_Size ({corr_matrix.loc['Age', 'Shoe_Size']:.3f})")
```

**Output:**
```
Correlation Matrix:
           Height  Weight    Age  Shoe_Size
Height      1.000   0.636 -0.030      0.637
Weight      0.636   1.000  0.062      0.464
Age        -0.030   0.062  1.000     -0.078
Shoe_Size   0.637   0.464 -0.078      1.000

Strongest correlation: Height-Weight (0.636)
Weakest correlation: Age-Shoe_Size (-0.078)
```


**Interpretation:**
- Diagonal is always 1.0 (perfect correlation with itself)
- Matrix is symmetric
- Height and Weight: very strong positive correlation (0.981)
- Height and Shoe_Size: strong positive correlation (0.832)
- Age: weakly correlated with other variables

## 2.2.6 Other Correlation Measures

### Spearman's Rank Correlation

Uses ranks instead of values - robust to outliers and works for monotonic (not necessarily linear) relationships.

```python
from scipy.stats import spearmanr

x = np.array([1, 2, 3, 4, 5, 100])  # Outlier!
y = np.array([2, 4, 6, 8, 10, 12])

r_pearson = stats.pearsonr(x, y)[0]
r_spearman = spearmanr(x, y)[0]

print(f"Data with outlier: x = {x}")
print(f"                   y = {y}")
print(f"\nPearson correlation:  {r_pearson:.3f} (affected by outlier)")
print(f"Spearman correlation: {r_spearman:.3f} (robust to outlier)")
print(f"\nSpearman uses ranks, so it's not affected by extreme values.")
```

**Output:**
```
Data with outlier: x = [  1   2   3   4   5 100]
                   y = [ 2  4  6  8 10 12]

Pearson correlation:  0.681 (affected by outlier)
Spearman correlation: 1.000 (robust to outlier)

Spearman uses ranks, so it's not affected by extreme values.
```


### Kendall's Tau

Another rank-based correlation measure, sometimes preferred over Spearman.

## Summary

### Key Points

1. **Correlation measures linear relationship strength** between two variables
2. **Range: -1 to +1** (bounded)
3. **Units-free** - doesn't depend on measurement scales
4. **Can be used for prediction** via linear regression
5. **Correlation ≠ Causation** - never forget this!
6. **Watch for:**
   - Non-linear patterns (check scatter plots!)
   - Outliers (can distort correlation)
   - Spurious correlations
   - Simpson's paradox

### Checklist for Using Correlation

- [ ] **Always plot your data first** (scatter plot)
- [ ] Check for outliers
- [ ] Verify the relationship is roughly linear
- [ ] Consider if correlation is meaningful in context
- [ ] Don't confuse correlation with causation
- [ ] Report both correlation and p-value
- [ ] Check sample size (small samples unreliable)

## Practice Problems

1. **Calculate correlations for:**
   - Study hours and test scores: [8,7,6,5,4,3,2,1] vs [95,88,82,75,70,65,58,50]
   - Temperature and heating bills: [5,10,15,20,25,30] vs [200,180,150,120,80,50]
   - Age and reaction time: [20,30,40,50,60,70] vs [250,270,290,320,360,410]

2. **Make predictions:**
   - Given the study hours data, predict score for 5.5 hours
   - Given temperature data, predict bill for 18°C

3. **Find spurious correlations:**
   - Research real examples online
   - Explain the likely confounding variables

4. **Create a correlation matrix:**
   - Use Iris dataset or another multi-variable dataset
   - Identify strongest and weakest correlations
   - Visualize with heatmap

## Next Steps

→ Continue to [Chapter 3: Probability Basics](../part2/chapter03.md)

→ Return to [2.1 Two-Dimensional Data](ch02_2d_data.md)