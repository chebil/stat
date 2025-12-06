# 7.1 Significance

The concept of statistical significance helps us determine whether observed patterns in data are meaningful or could have arisen by chance. This section introduces the framework for testing hypotheses about populations using samples.

## 7.1.1 Evaluating Significance

### The Null Hypothesis

Statistical testing begins with a **null hypothesis** (denoted $H_0$), which represents a "boring" or "default" state of the world. The null hypothesis typically claims that:
- There is no effect
- There is no difference between groups
- A parameter has a specific value

For example:
- $H_0$: A coin is fair (probability of heads = 0.5)
- $H_0$: A new drug has no effect (mean outcome with drug = mean outcome without drug)
- $H_0$: The population mean equals a specific value

### The Alternative Hypothesis

The **alternative hypothesis** (denoted $H_1$ or $H_A$) represents what we suspect might be true instead:
- There is an effect
- Groups differ
- A parameter differs from the null value

For example:
- $H_1$: A coin is biased (probability of heads ≠ 0.5)
- $H_1$: The new drug is effective (mean outcome with drug > mean outcome without drug)
- $H_1$: The population mean differs from the null value

### The Logic of Significance Testing

The process works as follows:

1. **Assume the null hypothesis is true**
2. **Compute how likely** your observed data (or something more extreme) would be under this assumption
3. **If the data would be very unlikely** under the null hypothesis, we have evidence against it
4. **Reject the null hypothesis** if the evidence is strong enough

This is reasoning by contradiction: we assume something, then show that assumption leads to something very improbable.

### Test Statistics

To measure how extreme our data is, we compute a **test statistic**—a single number that summarizes the evidence against the null hypothesis. Common test statistics include:

- **z-statistic**: When population standard deviation is known
- **t-statistic**: When population standard deviation is unknown
- **F-statistic**: For comparing variances
- **χ² statistic**: For testing model fit

```{admonition} Example: Coin Flipping
Suppose we flip a coin 100 times and get 60 heads. Is the coin fair?

- **Null hypothesis**: The coin is fair ($p = 0.5$)
- **Observed data**: 60 heads out of 100 flips
- **Question**: How likely is it to get 60 or more heads if the coin is truly fair?

If the coin is fair, the number of heads follows a binomial distribution with $N=100$ and $p=0.5$. We can compute that getting 60 or more heads has probability ≈ 0.028. This is fairly unlikely, suggesting the coin might be biased.
```

### One-Sided vs Two-Sided Tests

**Two-sided test**: We care about deviations in either direction
- $H_1$: Parameter ≠ null value
- Example: Testing if a coin is biased (either direction)

**One-sided test**: We care about deviations in only one direction
- $H_1$: Parameter > null value (or < null value)
- Example: Testing if a new drug improves outcomes (one direction)

The choice between one-sided and two-sided tests should be made **before** looking at the data.

## 7.1.2 P-Values

### Definition of P-Value

The **p-value** is the probability of observing data as extreme as (or more extreme than) what we actually observed, assuming the null hypothesis is true.

$$p\text{-value} = P(\text{data as extreme as observed} \mid H_0 \text{ is true})$$

```{important}
A p-value is NOT:
- The probability that the null hypothesis is true
- The probability that the alternative hypothesis is false
- The probability that results occurred by chance

It IS:
- The probability of observing such extreme data IF the null hypothesis were true
```

### Interpreting P-Values

**Small p-value** (typically < 0.05):
- Data would be unlikely if null hypothesis were true
- Evidence **against** the null hypothesis
- We might **reject** the null hypothesis

**Large p-value** (typically ≥ 0.05):
- Data is reasonably consistent with null hypothesis
- **Insufficient evidence** against the null hypothesis
- We **fail to reject** the null hypothesis (note: we don't "accept" it)

### Significance Levels

We choose a **significance level** (denoted $\alpha$) as a threshold:
- Common choices: $\alpha = 0.05$, $\alpha = 0.01$, $\alpha = 0.001$
- If $p < \alpha$: Result is "statistically significant at level $\alpha$"
- If $p \geq \alpha$: Result is "not statistically significant"

The choice of $\alpha$ should be made before analyzing data.

```{admonition} Example: Testing a Population Mean
Suppose we want to test if the average weight of a package differs from 500g.

- Sample of $N=25$ packages
- Sample mean: $\bar{x} = 485$g
- Sample standard deviation: $s = 30$g
- Null hypothesis: $H_0: \mu = 500$
- Alternative: $H_1: \mu \neq 500$ (two-sided)

**Step 1**: Compute test statistic
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{N}} = \frac{485 - 500}{30/\sqrt{25}} = \frac{-15}{6} = -2.5$$

**Step 2**: Find p-value
With $df = N-1 = 24$ degrees of freedom, using t-distribution:
$$p\text{-value} = 2 \times P(T \leq -2.5) \approx 0.019$$

(We multiply by 2 for a two-sided test)

**Step 3**: Conclusion
At $\alpha = 0.05$, we reject $H_0$ because $p < 0.05$. There is statistically significant evidence that the mean weight differs from 500g.
```

### Critical Values Approach

An equivalent approach uses **critical values**:
1. Choose significance level $\alpha$
2. Find critical value(s) from appropriate distribution
3. Reject $H_0$ if test statistic is more extreme than critical value

For example, with $\alpha = 0.05$ and two-sided test:
- For z-test: critical values are ±1.96
- For t-test with 24 df: critical values are ±2.064

### Common Misunderstandings

```{warning}
**Don't say**: "There's a 5% chance the null hypothesis is true"

**Do say**: "If the null hypothesis were true, we'd see data this extreme only 5% of the time"
```

```{warning}
**Don't say**: "The result is not significant, so the null hypothesis is true"

**Do say**: "We don't have sufficient evidence to reject the null hypothesis"
```

## Python Example: Hypothesis Testing

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Example: Is the mean of our sample significantly different from 100?
np.random.seed(42)
sample = np.random.normal(105, 15, size=30)  # True mean=105, we're testing against 100

# Null hypothesis: mu = 100
mu_0 = 100
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)

# Compute t-statistic
t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))

# Compute p-value (two-sided)
df = n - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample std: {sample_std:.2f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result is statistically significant at α=0.05")
    print("Reject H₀: Evidence that mean differs from 100")
else:
    print("Result is not statistically significant at α=0.05")
    print("Fail to reject H₀: Insufficient evidence that mean differs from 100")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left plot: Sample distribution
ax1.hist(sample, bins=10, density=True, alpha=0.7, edgecolor='black')
ax1.axvline(sample_mean, color='red', linestyle='--', linewidth=2, label=f'Sample mean: {sample_mean:.1f}')
ax1.axvline(mu_0, color='blue', linestyle='--', linewidth=2, label=f'Null hypothesis: {mu_0}')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Sample Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: t-distribution with test statistic
x = np.linspace(-4, 4, 200)
y = stats.t.pdf(x, df)
ax2.plot(x, y, 'b-', linewidth=2, label='t-distribution')

# Shade rejection regions
alpha = 0.05
t_critical = stats.t.ppf(1 - alpha/2, df)
ax2.axvline(t_stat, color='red', linestyle='--', linewidth=2, label=f't-statistic: {t_stat:.2f}')
ax2.axvline(-t_critical, color='gray', linestyle=':', alpha=0.7)
ax2.axvline(t_critical, color='gray', linestyle=':', alpha=0.7, label=f'Critical values: ±{t_critical:.2f}')

# Shade critical regions
x_left = x[x <= -t_critical]
y_left = stats.t.pdf(x_left, df)
x_right = x[x >= t_critical]
y_right = stats.t.pdf(x_right, df)
ax2.fill_between(x_left, y_left, alpha=0.3, color='red', label='Rejection regions (α=0.05)')
ax2.fill_between(x_right, y_right, alpha=0.3, color='red')

ax2.set_xlabel('t-value')
ax2.set_ylabel('Density')
ax2.set_title('T-Distribution and Test Statistic')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('significance_test_example.png', dpi=150, bbox_inches='tight')
plt.show()

# Using scipy's ttest_1samp for verification
t_stat_scipy, p_value_scipy = stats.ttest_1samp(sample, mu_0)
print(f"\nVerification using scipy.stats.ttest_1samp:")
print(f"t-statistic: {t_stat_scipy:.3f}")
print(f"p-value: {p_value_scipy:.4f}")
```

## Key Takeaways

- Statistical significance measures whether observed data is unlikely under a null hypothesis
- P-values quantify this unlikeliness, not the probability that hypotheses are true
- Small p-values (< significance level) provide evidence against the null hypothesis
- "Not significant" means "insufficient evidence," not "no effect"
- Always choose your significance level and test type before looking at data