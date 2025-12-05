# 1.1 Datasets

Before we can analyze data, we need to understand what data *is* and how we represent it.

## What is a Dataset?

A **dataset** is a collection of descriptions of different instances of the same phenomenon. These descriptions could take a variety of forms, but it is important that they are descriptions of the **same thing**.

### Examples of Datasets

1. **Rainfall measurements**: Daily rainfall in a garden over many years
2. **Height measurements**: Height of each person in a room
3. **Family sizes**: Number of children in each family on a block
4. **Preferences**: Whether 10 classmates prefer to be rich or famous
5. **Student records**: Test scores, attendance, demographics for students in a class

### Key Characteristic

All items in a dataset must be:
- Descriptions of the **same type of entity**
- Measured or recorded in a **consistent way**
- Comparable to each other

❌ **Not a dataset**: [height of person 1, weight of person 2, age of person 3]  
✅ **Valid dataset**: [height of person 1, height of person 2, height of person 3]

## Data Items and D-Tuples

A dataset consists of **$N$ data items**, where each data item is a **d-tuple** - an ordered list of $d$ elements.

### Notation Conventions

- **$N$**: Number of items in the dataset (always)
- **$d$**: Number of elements in each tuple (always the same for all tuples)
- **$\{x\}$**: The entire dataset
- **$x_i$**: The $i$-th data item
- **$x_i^{(j)}$**: The $j$-th component of the $i$-th data item

### Example: Students Dataset

Consider 5 students with (height in cm, weight in kg, age in years):

| Student | Height | Weight | Age |
|---------|--------|--------|-----|
| $x_1$ | 165 | 55 | 20 |
| $x_2$ | 170 | 62 | 21 |
| $x_3$ | 168 | 58 | 20 |
| $x_4$ | 175 | 70 | 22 |
| $x_5$ | 172 | 65 | 21 |

Here:
- $N = 5$ (five students)
- $d = 3$ (three measurements per student)
- $x_1 = (165, 55, 20)$ (first student's data)
- $x_3^{(2)} = 58$ (third student's weight)

## Types of Data

Data comes in different types, each requiring different analysis techniques.

### 1. Categorical Data

**Categorical data** consists of values from a fixed set of categories with no inherent numerical meaning.

**Examples**:
- Gender: {Male, Female, Other}
- Color: {Red, Blue, Green, Yellow}
- Country: {USA, Canada, Mexico, ...}
- Yes/No responses

**Characteristics**:
- Cannot meaningfully compute mean or standard deviation
- Can count frequencies
- Can compute mode (most common category)
- Use bar charts for visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Categorical data example
colors = ['Red', 'Blue', 'Red', 'Green', 'Blue', 'Red', 
          'Yellow', 'Blue', 'Red', 'Blue', 'Green', 'Red']

# Count frequencies
counts = Counter(colors)
print("Frequencies:", counts)
print("Mode:", counts.most_common(1)[0][0])

# Visualize
categories = list(counts.keys())
frequencies = list(counts.values())

plt.figure(figsize=(8, 5))
plt.bar(categories, frequencies, color=['red', 'blue', 'green', 'yellow'])
plt.xlabel('Color')
plt.ylabel('Frequency')
plt.title('Categorical Data: Color Distribution')
plt.show()
```

### 2. Ordinal Data

**Ordinal data** has categories with a meaningful order, but the distances between categories are not necessarily equal.

**Examples**:
- Education level: {High School, Bachelor's, Master's, PhD}
- Rating scale: {Poor, Fair, Good, Excellent}
- T-shirt sizes: {XS, S, M, L, XL, XXL}
- Grade levels: {Freshman, Sophomore, Junior, Senior}

**Characteristics**:
- Order matters (Master's > Bachelor's)
- Distances not equal (PhD - Master's ≠ Master's - Bachelor's in years)
- Can compute median
- Cannot meaningfully compute mean
- Can use ordered bar charts

```python
# Ordinal data example
ratings = ['Good', 'Excellent', 'Fair', 'Good', 'Poor', 
           'Excellent', 'Good', 'Good', 'Fair', 'Excellent']

# Define order
order = ['Poor', 'Fair', 'Good', 'Excellent']

# Map to numbers for analysis
rating_map = {rating: i for i, rating in enumerate(order)}
numeric_ratings = [rating_map[r] for r in ratings]

print(f"Median rating: {order[int(np.median(numeric_ratings))]}")
print(f"Mode rating: {Counter(ratings).most_common(1)[0][0]}")

# Visualize
rating_counts = Counter(ratings)
ordered_counts = [rating_counts[r] for r in order]

plt.figure(figsize=(8, 5))
plt.bar(order, ordered_counts, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Ordinal Data: Customer Ratings')
plt.show()
```

### 3. Continuous Data

**Continuous data** (also called quantitative or numerical data) consists of measurements on a continuous scale.

**Examples**:
- Height (cm)
- Weight (kg)
- Temperature (°C)
- Time (seconds)
- Income ($)

**Characteristics**:
- Can compute mean, median, standard deviation
- Can perform arithmetic operations
- Use histograms for visualization
- May have infinite precision (limited only by measurement)

```python
# Continuous data example
heights = np.array([165.5, 170.2, 168.7, 175.1, 172.3, 
                    169.8, 173.6, 177.4, 171.2, 166.9])

print(f"Mean height: {np.mean(heights):.2f} cm")
print(f"Median height: {np.median(heights):.2f} cm")
print(f"Std deviation: {np.std(heights):.2f} cm")

plt.figure(figsize=(8, 5))
plt.hist(heights, bins=6, edgecolor='black', alpha=0.7)
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.title('Continuous Data: Height Distribution')
plt.axvline(np.mean(heights), color='r', linestyle='--', label='Mean')
plt.legend()
plt.show()
```

### 4. Discrete Data

**Discrete data** consists of countable values, often integers.

**Examples**:
- Number of children in a family: {0, 1, 2, 3, ...}
- Number of cars owned: {0, 1, 2, 3, ...}
- Exam score (if integer): {0, 1, 2, ..., 100}
- Number of website visits

**Characteristics**:
- Countable (but potentially infinite)
- Can compute mean, median, mode
- Often use bar charts or histograms
- Gaps between possible values

```python
# Discrete data example
children_per_family = np.array([0, 1, 2, 1, 3, 2, 1, 0, 2, 1, 
                                 2, 4, 1, 2, 3, 1, 2, 0, 1, 2])

print(f"Mean: {np.mean(children_per_family):.2f} children")
print(f"Median: {np.median(children_per_family):.0f} children")
print(f"Mode: {Counter(children_per_family).most_common(1)[0][0]} children")

plt.figure(figsize=(8, 5))
values, counts = np.unique(children_per_family, return_counts=True)
plt.bar(values, counts, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Children')
plt.ylabel('Frequency')
plt.title('Discrete Data: Children per Family')
plt.xticks(values)
plt.show()
```

## Data Type Summary

| Type | Order? | Distances Meaningful? | Mean? | Median? | Example |
|------|--------|----------------------|-------|---------|----------|
| **Categorical** | No | No | ❌ | ❌ | Colors, Gender |
| **Ordinal** | Yes | No | ❌ | ✅ | Ratings, Sizes |
| **Discrete** | Yes | Yes | ✅ | ✅ | Counts, Integers |
| **Continuous** | Yes | Yes | ✅ | ✅ | Height, Weight |

## Vectors vs Tuples

Tuples differ from vectors:
- **Vectors**: Can always add and subtract
- **Tuples**: Cannot necessarily add or subtract (e.g., what is "Red" + "Blue"?)

However, we use the **same notation** for both:
- Write vectors and tuples in **bold**: $\mathbf{x}$
- Context makes it clear which is intended

Most of our data will be vectors (continuous or discrete numeric data).

## Missing Data

Sometimes we may not know the value of some elements in some tuples.

**Example**: Student data with missing test scores

| Student | Height | Weight | Test Score |
|---------|--------|--------|------------|
| 1 | 165 | 55 | 85 |
| 2 | 170 | ? | 90 |
| 3 | 168 | 58 | ? |

**Notation**: Often represented as:
- `NaN` (Not a Number) in Python/NumPy
- `NULL` in databases
- `NA` in R
- Empty cell in spreadsheets

**Handling strategies**:
1. **Delete rows** with missing data (if few)
2. **Impute** (fill in) with mean, median, or mode
3. **Use algorithms** that handle missing data
4. **Collect more data**

```python
import pandas as pd

# Dataset with missing values
data = pd.DataFrame({
    'Height': [165, 170, 168, np.nan, 172],
    'Weight': [55, 62, np.nan, 70, 65],
    'Score': [85, 90, 88, 92, np.nan]
})

print("Original data:")
print(data)
print(f"\nMissing values:\n{data.isna().sum()}")

# Strategy 1: Drop rows with any missing
data_dropped = data.dropna()
print(f"\nAfter dropping: {len(data_dropped)} rows")

# Strategy 2: Fill with mean
data_filled = data.fillna(data.mean())
print("\nAfter filling with mean:")
print(data_filled)
```

## Loading and Exploring Datasets

### Reading CSV Files

```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Basic exploration
print(df.head())  # First 5 rows
print(df.info())  # Data types and missing values
print(df.describe())  # Summary statistics
```

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [20, 21, 20, 22],
    'Score': [85, 90, 88, 92]
}
df = pd.DataFrame(data)

print(df)
print(f"\nDataset has {len(df)} items (N = {len(df)})")
print(f"Each item has {len(df.columns)} features (d = {len(df.columns)})")
```

## Common Data Sources

### 1. Surveys and Questionnaires
- Collect specific information from individuals
- Often mix categorical and continuous data
- Example: Customer satisfaction surveys

### 2. Sensors and Measurements
- Automated data collection
- Usually continuous or discrete
- Example: Temperature sensors, GPS coordinates

### 3. Databases and Records
- Existing organizational data
- Structured format
- Example: Hospital records, sales transactions

### 4. Web Scraping
- Extract data from websites
- Requires cleaning and processing
- Example: Product prices, social media data

### 5. Public Datasets
- Freely available for research and education
- Often well-documented
- Examples:
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml)
  - [Kaggle Datasets](https://www.kaggle.com/datasets)
  - [Data.gov](https://www.data.gov)

## Best Practices for Working with Datasets

### 1. Always Explore First

```python
# Check first few rows
print(df.head())

# Check data types
print(df.dtypes)

# Check for missing values
print(df.isna().sum())

# Summary statistics
print(df.describe())
```

### 2. Visualize Early

- Plot histograms for continuous variables
- Create bar charts for categorical variables
- Look for outliers and patterns

### 3. Document Your Data

- What does each column represent?
- What are the units?
- When/how was data collected?
- Are there known issues?

### 4. Clean Your Data

- Handle missing values
- Remove duplicates
- Fix data type issues
- Deal with outliers

```python
# Example cleaning pipeline
df_clean = df.copy()

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Handle missing values
df_clean = df_clean.fillna(df_clean.mean())

# Convert types if needed
df_clean['Age'] = df_clean['Age'].astype(int)

print(f"Cleaned: {len(df)} → {len(df_clean)} rows")
```

## Summary

Key takeaways:

1. **Dataset**: Collection of descriptions of the same phenomenon
2. **N data items**, each with **d elements**
3. **Four data types**: Categorical, Ordinal, Discrete, Continuous
4. **Different types require different analyses**
5. **Missing data is common** - have a strategy to handle it
6. **Always explore and visualize** before analyzing

## Next Steps

→ Continue to [1.2 What's Happening? Plotting Data](ch01_plotting.md)

→ See [1.3 Summarizing 1D Data](ch01_summarizing.md) for numerical summaries