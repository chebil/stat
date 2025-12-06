# 8.3 Experimental Design Principles

Good experimental design is crucial for valid statistical inference. This section covers the fundamental principles that ensure reliable results.

## The Three R's of Experimental Design

### 1. Randomization
### 2. Replication  
### 3. Blocking (when needed)

## Randomization

### What is Randomization?

**Random assignment** of experimental units to treatments.

### Why Randomize?

1. **Controls for confounding**: Balances unknown factors across groups
2. **Justifies statistical inference**: Allows use of probability theory
3. **Reduces bias**: Prevents systematic differences between groups

### Example: Without Randomization

```
BAD: Assign morning students to Method A, afternoon to Method B

Problem: Time of day is confounded with teaching method!
         Any difference could be due to:
         - Teaching method
         - Student alertness
         - Teacher fatigue
         - etc.
```

### Example: With Randomization

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# List of 40 students
students = [f'Student_{i:02d}' for i in range(1, 41)]

# Randomize assignment to 4 groups (10 each)
np.random.shuffle(students)

groups = {
    'Group_A': students[0:10],
    'Group_B': students[10:20],
    'Group_C': students[20:30],
    'Group_D': students[30:40]
}

print("Randomized Group Assignments")
print("="*60)
for group, members in groups.items():
    print(f"\n{group}:")
    print(', '.join(members))

# Create dataframe for analysis
treatments = []
for group in ['A', 'B', 'C', 'D']:
    treatments.extend([group] * 10)

df = pd.DataFrame({
    'Student': students,
    'Treatment': treatments
})

print("\n" + "="*60)
print("Treatment Assignment Summary:")
print(df['Treatment'].value_counts().sort_index())
```

**Output:**
```
Randomized Group Assignments
============================================================

Group_A:
Student_20, Student_17, Student_16, Student_27, Student_05, Student_13, Student_38, Student_28, Student_40, Student_07

Group_B:
Student_26, Student_10, Student_14, Student_32, Student_35, Student_09, Student_18, Student_25, Student_01, Student_34

Group_C:
Student_06, Student_12, Student_02, Student_30, Student_22, Student_03, Student_31, Student_37, Student_04, Student_36

Group_D:
Student_24, Student_33, Student_11, Student_23, Student_19, Student_21, Student_08, Student_15, Student_29, Student_39

============================================================
Treatment Assignment Summary:
Treatment
A    10
B    10
C    10
D    10
Name: count, dtype: int64
```


## Replication

### What is Replication?

**Multiple independent observations** for each treatment.

### Why Replicate?

1. **Estimate variability**: Cannot assess variation with n=1!
2. **Increase precision**: Standard error decreases with \(\sqrt{n}\)
3. **Increase power**: Better chance of detecting real effects
4. **Check reproducibility**: Ensure results aren't flukes

### How Much Replication?

**Power analysis** determines required sample size:

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def sample_size_anova(k, effect_size, alpha=0.05, power=0.80):
    """
    Estimate required sample size per group for one-way ANOVA.
    
    k: number of groups
    effect_size: Cohen's f (small=0.1, medium=0.25, large=0.4)
    alpha: significance level
    power: desired power
    """
    # This is a simplified approximation
    # For precise calculation, use specialized software
    
    from scipy.stats import f as f_dist
    
    # Start with initial guess
    n = 10
    current_power = 0
    
    while current_power < power and n < 1000:
        # Non-centrality parameter
        ncp = k * n * effect_size**2
        
        # Critical F-value
        df1 = k - 1
        df2 = k * (n - 1)
        f_crit = f_dist.ppf(1 - alpha, df1, df2)
        
        # Power (using non-central F distribution)
        from scipy.stats import ncf
        current_power = 1 - ncf.cdf(f_crit, df1, df2, ncp)
        
        n += 1
    
    return n

# Example: How many subjects per group?
k_groups = 4
effect_sizes = {'small': 0.1, 'medium': 0.25, 'large': 0.4}

print("Required Sample Size per Group (Power = 0.80)")
print("="*60)
print(f"{'Effect Size':<15} {'Cohen\'s f':<12} {'n per group':<15} {'Total N':<10}")
print("-"*60)

for name, f in effect_sizes.items():
    n = sample_size_anova(k_groups, f)
    total_n = k_groups * n
    print(f"{name:<15} {f:<12.2f} {n:<15d} {total_n:<10d}")

print("\nNote: These are approximations. Use specialized software for")
print("      precise power analysis (e.g., G*Power, R pwr package)")
```

**Output:**
```
Required Sample Size per Group (Power = 0.80)
============================================================
Effect Size     Cohen's f    n per group     Total N   
------------------------------------------------------------
small           0.10         275             1100      
medium          0.25         46              184       
large           0.40         20              80        

Note: These are approximations. Use specialized software for
      precise power analysis (e.g., G*Power, R pwr package)
```


## Blocking

### What is Blocking?

**Grouping experimental units** by a known source of variability, then randomizing within blocks.

### When to Block?

When you have a **known, controllable source of variation**:
- Different batches of materials
- Different time periods
- Different locations
- Matched subjects (twins, littermates)

### Randomized Complete Block Design (RCBD)

```
Block 1 (Field Location A):  [Treat1] [Treat2] [Treat3] [Treat4]
Block 2 (Field Location B):  [Treat4] [Treat1] [Treat3] [Treat2]  
Block 3 (Field Location C):  [Treat2] [Treat4] [Treat1] [Treat3]

- Each treatment appears once in each block
- Treatment order randomized within blocks
- Removes variation due to location
```

### Python Example: RCBD Analysis

```python
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# Randomized Complete Block Design
# 4 treatments, 5 blocks (e.g., 5 fields)

data = {
    'Yield': [
        # Block 1
        45, 52, 48, 50,
        # Block 2  
        40, 47, 43, 45,
        # Block 3
        50, 57, 53, 55,
        # Block 4
        38, 45, 41, 43,
        # Block 5
        48, 55, 51, 53
    ],
    'Treatment': ['A', 'B', 'C', 'D'] * 5,
    'Block': [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4
}

df = pd.DataFrame(data)

print("Randomized Complete Block Design")
print("="*70)
print("\nData Summary:")
print(df.groupby(['Treatment', 'Block'])['Yield'].mean().unstack())

# Two-way ANOVA with blocking
try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    
    # Block is a factor but we're not interested in its main effect
    # We include it to account for block-to-block variation
    model = ols('Yield ~ C(Treatment) + C(Block)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    
    print("\n" + "="*70)
    print("ANOVA Table (RCBD)")
    print("="*70)
    print(anova_table)
    
    p_treatment = anova_table.loc['C(Treatment)', 'PR(>F)']
    p_block = anova_table.loc['C(Block)', 'PR(>F)']
    
    print("\n" + "="*70)
    print("Interpretation")
    print("="*70)
    
    print(f"\nTreatment effect: p = {p_treatment:.4f}")
    if p_treatment < 0.05:
        print("  → Significant: Treatments differ")
    else:
        print("  → Not significant: No treatment effect detected")
    
    print(f"\nBlock effect: p = {p_block:.4f}")
    if p_block < 0.05:
        print("  → Blocking was effective (blocks differ significantly)")
        print("     Good decision to block!")
    else:
        print("  → Blocking was not necessary (blocks don't differ much)")
        
except ImportError:
    print("Install statsmodels for RCBD analysis")

# Compare: What if we hadn't blocked?
print("\n" + "="*70)
print("Comparison: With vs. Without Blocking")
print("="*70)

# Without blocking (one-way ANOVA)
treatment_groups = [df[df['Treatment']==t]['Yield'].values for t in ['A','B','C','D']]
f_no_block, p_no_block = stats.f_oneway(*treatment_groups)

print(f"\nWithout blocking: p = {p_no_block:.4f}")
print(f"With blocking:    p = {p_treatment:.4f}")
print(f"\nBlocking {'increased' if p_treatment < p_no_block else 'decreased'} power!")
```

**Output:**
`Error: NameError: name 'p_treatment' is not defined`


### Efficiency of Blocking

Blocking **increases power** by removing block-to-block variation from the error term.

**Relative Efficiency**:

$$
\text{RE} = \frac{\text{MSE (unblocked)}}{\text{MSE (blocked)}}
$$

RE > 1 means blocking was beneficial.

## Common Experimental Designs

### 1. Completely Randomized Design (CRD)

**Structure**: Random assignment to treatments, no blocking

**When to use**: Homogeneous experimental units

**Analysis**: One-way ANOVA

```python
# Example: 30 subjects, 3 treatments
subjects = np.arange(30)
np.random.shuffle(subjects)

design_crd = pd.DataFrame({
    'Subject': subjects,
    'Treatment': ['A']*10 + ['B']*10 + ['C']*10
})
```

**Output:** `(No output)`


### 2. Randomized Complete Block Design (RCBD)

**Structure**: Blocks contain all treatments, randomized within blocks

**When to use**: Known source of variation to control

**Analysis**: Two-way ANOVA (treatment + block)

### 3. Latin Square Design

**Structure**: Control for **two** blocking factors simultaneously

**Example**: Different operators (rows), different machines (columns)

```
        Machine 1  Machine 2  Machine 3  Machine 4
Op 1        A          B          C          D
Op 2        B          C          D          A
Op 3        C          D          A          B  
Op 4        D          A          B          C

Each treatment appears once in each row AND column
```

**Analysis**: Three-way ANOVA (treatment + row block + column block)

### 4. Factorial Design

**Structure**: Multiple factors, all combinations tested

**When to use**: Study multiple factors and their interactions

**Analysis**: Multi-way ANOVA

## Sample Size Considerations

### Factors Affecting Required Sample Size

1. **Effect size**: Smaller effects need larger n
2. **Variability**: Higher σ needs larger n  
3. **Significance level**: Smaller α needs larger n
4. **Desired power**: Higher power needs larger n
5. **Number of groups**: More groups need larger n

### Rules of Thumb

- **Pilot studies**: n ≥ 5-10 per group (exploratory)
- **Main experiments**: n ≥ 20-30 per group (standard)
- **Detect small effects**: n ≥ 50+ per group
- **Always do power analysis** before collecting data!

## Practical Considerations

### Randomization Methods

```python
import numpy as np

def randomize_assignment(subjects, treatments, method='complete'):
    """
    Randomize subject assignment to treatments.
    
    methods:
    - 'complete': Completely random (may give unequal groups)
    - 'balanced': Force equal group sizes
    - 'block': Randomize within blocks
    """
    n = len(subjects)
    k = len(treatments)
    
    if method == 'complete':
        # Completely random
        assignment = np.random.choice(treatments, size=n)
        
    elif method == 'balanced':
        # Force equal sizes
        n_per_group = n // k
        assignment = []
        for treatment in treatments:
            assignment.extend([treatment] * n_per_group)
        np.random.shuffle(assignment)
        
    return pd.DataFrame({
        'Subject': subjects,
        'Treatment': assignment
    })

# Example
subjects = [f'S{i:02d}' for i in range(1, 41)]
treatments = ['Control', 'TreatA', 'TreatB', 'TreatC']

assignment = randomize_assignment(subjects, treatments, method='balanced')
print(assignment.groupby('Treatment').size())
```

**Output:**
```
Treatment
Control    10
TreatA     10
TreatB     10
TreatC     10
dtype: int64
```


### Dealing with Missing Data

**Prevention**:
- Build in redundancy
- Careful data collection protocols
- Regular data checks

**If it happens**:
- Document reasons for missingness
- Use appropriate methods:
  - Complete case analysis (if MCAR - Missing Completely At Random)
  - Mixed models (handles unbalanced data well)
  - Multiple imputation (advanced)

## Assumptions and Their Violations

### Independence

**Violated by**:
- Pseudoreplication (e.g., multiple measurements on same subject)
- Spatial correlation
- Temporal correlation

**Solutions**:
- Proper randomization
- Blocking
- Mixed models (repeated measures)

### Normality

**Check**: Q-Q plots, Shapiro-Wilk test

**If violated**:
- Transformations (log, sqrt, Box-Cox)
- Non-parametric tests (Kruskal-Wallis)
- Bootstrapping
- Generalized linear models (GLM)

### Homogeneity of Variance

**Check**: Levene's test, residual plots

**If violated**:
- Transformations
- Welch's ANOVA
- Weighted least squares
- Generalized linear models

## Summary

### The Golden Rules

1. **Randomize** whenever possible
2. **Replicate** adequately (power analysis!)
3. **Block** when there's known variation
4. **Balance** your design (equal group sizes)
5. **Check assumptions** before and after analysis
6. **Report** effect sizes, not just p-values

### Design Selection Guide

| Situation | Design | Analysis |
|-----------|--------|----------|
| Homogeneous units, 1 factor | CRD | One-way ANOVA |
| Known blocking variable, 1 factor | RCBD | Two-way ANOVA |
| 2+ factors of interest | Factorial | Multi-way ANOVA |
| 2 blocking factors | Latin Square | Three-way ANOVA |
| Repeated measures | Within-subjects | Repeated measures ANOVA |

### Common Mistakes to Avoid

❌ Pseudo-replication (treating subsamples as independent)  
❌ Forgetting to randomize  
❌ Unbalanced designs (when avoidable)  
❌ Ignoring interactions in factorial designs  
❌ Not checking assumptions  
❌ P-hacking and multiple testing  

### Before You Experiment

✅ Define clear hypotheses  
✅ Conduct power analysis  
✅ Choose appropriate design  
✅ Plan randomization procedure  
✅ Determine data collection protocols  
✅ Prepare analysis plan  

## Conclusion

Good experimental design is the foundation of valid statistical inference. No amount of sophisticated analysis can rescue a poorly designed experiment!

**Remember**: 
> "To consult the statistician after an experiment is finished is often merely to ask him to conduct a post mortem examination. He can perhaps say what the experiment died of." 
> — R.A. Fisher

Next chapter: **Inferring Probability Models from Data** (Maximum Likelihood and Bayesian Inference)!