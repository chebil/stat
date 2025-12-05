# File Expansion Summary - UPDATED

This document tracks the comprehensive expansion of existing files to include all subsections and worked examples from the source textbook.

**Last Updated**: December 5, 2025, 9:31 PM +03

---

## ✅ Completed Expansions (6/6 Priority Files)

### 1. ch01_summarizing.md (1.3 Summarizing 1D Data)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** 2.6 KB (basic outline)  
**Expanded to:** 14.4 KB (comprehensive content)

#### New Subsections Added:

##### 1.3.1 The Mean
- Complete definition with mathematical notation
- **Property 1.1**: Translation and scaling properties
- Property: Sum of deviations equals zero
- **Worked Example**: Bar net worth calculation
- Python implementation with numpy

##### 1.3.2 Standard Deviation and Variance
- Detailed formulas for both measures
- **Property 1.2**: Chebyshev's bound (at most 1/k² data k std devs away)
- **Property 1.3**: At least one point is ≥1 std dev away
- Comparison of population vs sample standard deviation
- Explanation of N vs N-1 divisor
- Python code for calculations

##### 1.3.3 Computing Mean and Standard Deviation Online
- **NEW**: Online algorithm for streaming data
- Recursive formulas for updating mean and std dev
- Complete Python class implementation
- Useful for memory-constrained applications

##### 1.3.4 The Median
- Formal definition with examples
- **Worked Example**: Bar with billionaire (outlier analysis)
  - Mean: $107,903 → $91,007,184 (huge change!)
  - Median: $107,835 → $108,930 (minimal change)
- Properties: translation and scaling
- When to use median vs mean

##### 1.3.5 Percentiles and Quartiles
- Definitions of percentiles and quartiles (Q1, Q2, Q3)
- **Interquartile Range (IQR)** formula and properties
- Outlier detection: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
- Comparison with standard deviation for outlier robustness
- Python code for quartile calculations

##### 1.3.6 Using Summaries Sensibly
- **NEW**: Reporting precision and significant figures
  - Example: "32.833 weeks" pregnancy is over-precise
- Categorical vs continuous variables
  - Why "2.6 children" is problematic
- Guidelines for when to use mean vs median
- Best practices for data reporting

##### 1.3.7 Standard Coordinates (Z-Scores)
- Complete definition and formula
- Properties: mean=0, std=1, unitless
- Interpretation guide
- Python implementation
- Use cases for standardization

---

### 2. ch02_correlation.md (2.2 Correlation)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** 3.6 KB (basic content)  
**Expanded to:** 14.5 KB (comprehensive content)

#### New Subsections Added:

##### 2.2.1 The Pearson Correlation Coefficient
- Three equivalent formulas (standard, using std devs, using z-scores)
- Detailed interpretation guide for r values
- **Property 2.1**: Bounded range (-1 ≤ r ≤ 1)
- **Property 2.2**: Units-free
- **Property 2.3**: Translation invariance
- **Property 2.4**: Scale invariance  
- **Property 2.5**: Symmetry

##### 2.2.2 Computing Correlation
- **Worked Example**: Height and Weight (6 people)
  - Step-by-step calculation: r = 0.998
  - Three methods in Python (numpy, scipy, manual)
- **Worked Example**: Study Hours vs Errors
  - 8 students, r ≈ -0.96
  - Scatter plot visualization
- Complete Python implementations

##### 2.2.3 Correlation and Prediction
- **Prediction formula** for linear regression
- **Worked Example**: Predicting weight from height
- Regression to the mean explanation
- Python prediction function
- Visualization of prediction line

##### 2.2.4 Correlation Pitfalls and Limitations
- **Pitfall 1: Correlation ≠ Causation**
- **Pitfall 2: Non-linear Relationships**
- **Pitfall 3: Outliers**
- **Pitfall 4: Restricted Range**
- **Pitfall 5: Simpson's Paradox**

##### 2.2.5 Correlation Matrix
- Definition and interpretation
- Heatmap visualization with seaborn
- Python code for pandas correlation

##### 2.2.6 Other Correlation Measures
- **Spearman's rank correlation**
- **Kendall's tau**
- When to use each measure

---

### 3. ch01_plotting.md (1.2 What's Happening? Plotting Data)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** Unknown (not tracked in original summary)  
**Expanded to:** 16.2 KB (comprehensive content)

#### Subsections Included:

##### 1.2.1 Bar Charts
- Definition and when to use
- **Example 1**: Gender Distribution
- **Example 2**: Goals of Students
- Horizontal vs Vertical Bars
- Complete Python implementations

##### 1.2.2 Histograms
- Definition and purpose
- **Example 1**: Net Worth Data (10 people)
- **Example 2**: Cheese Goodness Scores (20 cheeses)
- Distribution interpretation

##### 1.2.3 How to Make Histograms
- Histograms with Even Intervals
- Choosing the Number of Bins (Square root, Sturges', Scott's rules)
- Histograms with Uneven Intervals
- Manual histogram calculation in Python

##### 1.2.4 Conditional Histograms
- When to use
- **Example**: Body Temperature by Gender
- Overlapping Conditional Histograms
- Class-conditional analysis

**New Content**:
- Complete comparison table: Bar Charts vs Histograms
- Key takeaways section
- Checklist for good plots
- Common pitfalls section
- Practice problems

---

### 4. ch01_plots_summaries.md (1.4 Plots and Summaries)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** Unknown  
**Expanded to:** 12.1 KB (comprehensive content)

#### Subsections Included:

##### 1.4.1 Some Properties of Histograms
- Tails and Modes (unimodal, bimodal, multimodal)
- **Skewness**: Right-skewed and Left-skewed data
- Checking for Skewness (Method 1: histogram, Method 2: mean vs median)
- Python examples with visualizations

##### 1.4.2 Standard Coordinates and Normal Data
- Definition of Standard Coordinates (Z-scores)
- Properties (mean=0, std=1)
- **Normal Data** definition and standard normal curve
- **68-95-99.7 Rule** (Empirical Rule)
- Python implementation with scipy

##### 1.4.3 Box Plots
- Building a Box Plot (5 steps)
- Creating Box Plots in Python
- Interpreting Box Plots
- Comparison of multiple groups

**New Content**:
- Complete examples with output
- Visual comparisons
- Practice problems
- Integration with normal distribution

---

### 5. ch02_2d_data.md (2.1 Plotting 2D Data)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** Unknown  
**Expanded to:** 17.2 KB (comprehensive content)

#### Subsections Included:

##### 2.1.1 Categorical Data, Counts, and Charts
- Contingency Tables
- Stacked Bar Charts
- Grouped Bar Charts
- Heatmaps
- Mosaic Plots

##### 2.1.2 Series
- Line Plots for time series
- **Example**: Temperature Over Time (365 days)
- Multiple Series comparison
- Seasonal Decomposition

##### 2.1.3 Scatter Plots for Spatial Data
- Basic Scatter Plot (earthquake example)
- Point Density Maps
- Geographic visualization

##### 2.1.4 Exposing Relationships with Scatter Plots
- Basic Scatter Plot
- Adding Trend Lines (with regression)
- Color-Coded Scatter Plots
- Scatter Plot Patterns (4 types)
- Bubble Charts (3-variable visualization)

**New Content**:
- Complete visualization comparison table
- Summary of techniques by data type
- Practice problems
- Next steps section

---

### 6. ch01_australian_pizzas.md (1.5 Whose is Bigger? Investigating Australian Pizzas)

**Status**: ✅ **FULLY EXPANDED**  
**Expanded from:** Unknown  
**Expanded to:** 10.5 KB (comprehensive case study)

#### Complete Case Study Structure:

- **The Question**: EagleBoys vs Domino's claim
- **The Dataset**: Pizza diameter measurements
- **Step 1**: Compute Summary Statistics
- **Step 2**: Visualize with Histograms
- **Step 3**: Compare with Box Plots
- **Step 4**: Overlaid Histograms
- **Step 5**: Standard Coordinates Comparison
- **Step 6**: Check for Outliers (IQR method)

**Conclusions**:
- Evidence supporting the claim
- Important caveats
- Final answer with nuance

**New Content**:
- Complete worked example with real data
- 6-step analysis workflow
- Python implementation throughout
- Practice exercise
- Key lessons learned

---

## File Size Summary

| File | Current Size | Status |
|------|-------------|--------|
| ch01_summarizing.md | 14.4 KB | ✅ Complete |
| ch02_correlation.md | 14.5 KB | ✅ Complete |
| ch01_plotting.md | 16.2 KB | ✅ Complete |
| ch01_plots_summaries.md | 12.1 KB | ✅ Complete |
| ch02_2d_data.md | 17.2 KB | ✅ Complete |
| ch01_australian_pizzas.md | 10.5 KB | ✅ Complete |

**Total expanded content**: ~85 KB of comprehensive educational material

---

## Remaining Files to Expand

### Part 1 - Chapter 1

- [ ] `ch01_datasets.md` (13.7 KB) - **Check if already expanded**
- [ ] `ch01_you_should.md` (7.6 KB) - **Check if already expanded**

### Part 1 - Chapter 2

All Chapter 2 files appear to be expanded!

### Part 2 - Probability

All Part 2 files need expansion with:
- Probability axioms and theorems
- Conditional probability
- Bayes' theorem
- Random variables and distributions
- Common distributions (Binomial, Poisson, Normal, etc.)
- Central limit theorem
- Joint and marginal distributions

---

## Content Enhancements Applied

Each expanded file now includes:

### 1. **Complete Mathematical Definitions**
- Precise formulas with proper LaTeX notation
- Clear variable definitions
- Step-by-step derivations where appropriate

### 2. **Properties and Theorems**
- Numbered properties (Property X.Y)
- Formal statements
- Intuitive explanations
- Practical implications

### 3. **Worked Examples**
- Real datasets
- Step-by-step calculations
- Multiple solution approaches
- Interpretation of results

### 4. **Python Code**
- Complete, runnable examples
- Multiple implementation methods
- Comments explaining each step
- Best practices demonstrated
- Output shown for verification

### 5. **Visualizations**
- Matplotlib/seaborn plotting code
- Multiple chart types
- Proper labeling and formatting
- Interpretation guidance

### 6. **Practical Guidance**
- When to use each technique
- Common pitfalls to avoid
- Best practices
- Real-world considerations

### 7. **Practice Problems**
- Multiple difficulty levels
- Diverse datasets
- Progressively challenging
- Encourage hands-on learning

### 8. **Summary Sections**
- Key takeaways
- Comparison tables
- Checklists
- Quick reference guides

---

## Quality Standards Maintained

✅ **Mathematical Rigor**: All formulas are precise with proper notation  
✅ **Pedagogical Structure**: Concepts build logically from simple to complex  
✅ **Code Quality**: All code is tested, commented, and follows best practices  
✅ **Practical Focus**: Real datasets and applications throughout  
✅ **Accessibility**: Clear explanations suitable for computer science students  
✅ **Completeness**: No gaps in coverage of subsections from source material  
✅ **Consistency**: Uniform formatting and style across all files

---

## Next Steps

### Priority 1: Check Remaining Part 1 Files
1. **Review ch01_datasets.md** - Verify if comprehensive or needs expansion
2. **Review ch01_you_should.md** - Check learning objectives completeness

### Priority 2: Begin Part 2 Expansion
Once Part 1 is confirmed complete, begin systematic expansion of Part 2 (Probability) chapters following the same comprehensive approach.

### Priority 3: Create Interactive Components
- Jupyter notebooks for each chapter
- Solutions manual for practice problems
- Interactive visualizations
- Real-world case studies

---

## References

**Source Material**: *Probability and Statistics for Computer Science* by David Forsyth
- Chapter 1: First Tools for Looking at Data (pages 1-30)
- Chapter 2: Looking at Relationships (pages 31-50)

---

**Status**: Part 1 is essentially complete! 🎉  
**Next Goal**: Verify remaining files and move to Part 2 expansion

---

**Contributors**: Expanded from original textbook structure  
**Repository**: https://github.com/chebil/stat  
**Jupyter Book URL**: https://chebil.github.io/stat