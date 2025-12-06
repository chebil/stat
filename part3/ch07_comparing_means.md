# 7.2 Comparing the Mean of Two Populations

A common statistical question is whether two populations have the same mean. For example:
- Does a new teaching method improve test scores compared to the traditional method?
- Do men and women have different average salaries in a company?
- Is a new drug more effective than a placebo?

We answer these questions by collecting samples from each population and using hypothesis tests.

## General Setup

We have:
- **Population 1**: mean $\mu_1$, standard deviation $\sigma_1$
- **Population 2**: mean $\mu_2$, standard deviation $\sigma_2$
- **Sample 1**: size $N_1$, mean $\bar{x}_1$, standard deviation $s_1$
- **Sample 2**: size $N_2$, mean $\bar{x}_2$, standard deviation $s_2$

**Null hypothesis**: $H_0: \mu_1 = \mu_2$ (or equivalently, $\mu_1 - \mu_2 = 0$)

**Alternative hypothesis** (two-sided): $H_1: \mu_1 \neq \mu_2$

Or for one-sided tests:
- $H_1: \mu_1 > \mu_2$ (population 1 has larger mean)
- $H_1: \mu_1 < \mu_2$ (population 1 has smaller mean)

The approach we use depends on what we know about the population standard deviations.

## 7.2.1 Assuming Known Population Standard Deviations

This case is primarily theoretical—in practice, we rarely know the true population standard deviations. However, it's useful for understanding the logic.

### When to Use
- We know both $\sigma_1$ and $\sigma_2$ (rare in practice)
- OR both sample sizes are very large ($N_1, N_2 \geq 30$), so $s_1 \approx \sigma_1$ and $s_2 \approx \sigma_2$

### Test Statistic (Z-Test)

Under the null hypothesis, $\bar{x}_1 - \bar{x}_2$ is approximately normally distributed with:
- Mean: $\mu_1 - \mu_2 = 0$ (under $H_0$)
- Standard error: $\sqrt{\frac{\sigma_1^2}{N_1} + \frac{\sigma_2^2}{N_2}}$

The test statistic is:

$$z = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{\sigma_1^2}{N_1} + \frac{\sigma_2^2}{N_2}}}$$

Under $H_0$, this follows a standard normal distribution $N(0,1)$.

### Decision Rule

For significance level $\alpha$:

**Two-sided test**:
- Reject $H_0$ if $|z| > z_{\alpha/2}$ (e.g., $|z| > 1.96$ for $\alpha = 0.05$)
- Or equivalently, reject if p-value $< \alpha$, where p-value $= 2 \times P(Z > |z|)$

**One-sided test** ($H_1: \mu_1 > \mu_2$):
- Reject $H_0$ if $z > z_\alpha$ (e.g., $z > 1.645$ for $\alpha = 0.05$)
- Or equivalently, reject if p-value $< \alpha$, where p-value $= P(Z > z)$

```{admonition} Example: Comparing Exam Scores
Two classes take the same exam:
- **Class A**: $N_1 = 50$, $\bar{x}_1 = 75$, $\sigma_1 = 10$ (known from past data)
- **Class B**: $N_2 = 45$, $\bar{x}_2 = 72$, $\sigma_2 = 12$ (known from past data)

Test if class means differ at $\alpha = 0.05$.

$$z = \frac{75 - 72}{\sqrt{\frac{10^2}{50} + \frac{12^2}{45}}} = \frac{3}{\sqrt{2 + 3.2}} = \frac{3}{2.28} \approx 1.32$$

For two-sided test, critical values are $\pm 1.96$. Since $|z| = 1.32 < 1.96$, we fail to reject $H_0$.

P-value $= 2 \times P(Z > 1.32) \approx 2 \times 0.093 = 0.186$

**Conclusion**: At $\alpha = 0.05$, there is insufficient evidence that class means differ.
```

## 7.2.2 Assuming Same, Unknown Population Standard Deviation

Often we don't know $\sigma_1$ or $\sigma_2$, but we believe they are equal: $\sigma_1 = \sigma_2 = \sigma$ (common variance).

### When to Use
- Both $\sigma_1$ and $\sigma_2$ are unknown
- We believe $\sigma_1 = \sigma_2$ (can test this with F-test, see Section 7.3.1)
- This is called the "pooled variance" or "equal variance" t-test

### Pooled Variance Estimate

We combine both samples to estimate the common variance:

$$s_p^2 = \frac{(N_1 - 1)s_1^2 + (N_2 - 1)s_2^2}{N_1 + N_2 - 2}$$

This is a weighted average of the two sample variances.

### Test Statistic (Pooled T-Test)

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{N_1} + \frac{1}{N_2}}}$$

Under $H_0$, this follows a t-distribution with degrees of freedom:

$$df = N_1 + N_2 - 2$$

### Decision Rule

For significance level $\alpha$:

**Two-sided test**:
- Reject $H_0$ if $|t| > t_{\alpha/2, df}$
- Or equivalently, reject if p-value $< \alpha$

```{admonition} Example: Comparing Two Drugs
Compare recovery times for two drugs:
- **Drug A**: $N_1 = 12$, $\bar{x}_1 = 8.5$ days, $s_1 = 2.1$ days
- **Drug B**: $N_2 = 15$, $\bar{x}_2 = 9.8$ days, $s_2 = 2.3$ days

Assume equal population variances. Test at $\alpha = 0.05$.

**Step 1**: Pooled variance
$$s_p^2 = \frac{11 \times (2.1)^2 + 14 \times (2.3)^2}{12 + 15 - 2} = \frac{48.51 + 74.06}{25} = \frac{122.57}{25} = 4.90$$
$$s_p = \sqrt{4.90} \approx 2.21$$

**Step 2**: Test statistic
$$t = \frac{8.5 - 9.8}{2.21 \sqrt{\frac{1}{12} + \frac{1}{15}}} = \frac{-1.3}{2.21 \times 0.387} = \frac{-1.3}{0.855} \approx -1.52$$

**Step 3**: Critical value and decision
With $df = 25$ and $\alpha = 0.05$ (two-sided), $t_{0.025, 25} \approx 2.060$

Since $|t| = 1.52 < 2.060$, we fail to reject $H_0$.

P-value $\approx 0.14$

**Conclusion**: At $\alpha = 0.05$, there is insufficient evidence that recovery times differ between the two drugs.
```

## 7.2.3 Assuming Different, Unknown Population Standard Deviation

This is the most general case: we don't know $\sigma_1$ or $\sigma_2$, and we don't assume they're equal.

### When to Use
- Both $\sigma_1$ and $\sigma_2$ are unknown
- We DON'T believe $\sigma_1 = \sigma_2$ (unequal variances)
- This is called "Welch's t-test" or "unequal variance t-test"

### Test Statistic (Welch's T-Test)

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$

This follows approximately a t-distribution with **Welch-Satterthwaite degrees of freedom**:

$$df = \frac{\left(\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}\right)^2}{\frac{(s_1^2/N_1)^2}{N_1-1} + \frac{(s_2^2/N_2)^2}{N_2-1}}$$

This formula looks complicated but is straightforward to compute. The df is typically non-integer and is usually rounded down.

### Decision Rule

Same as pooled t-test, but using Welch's df:
- Reject $H_0$ if $|t| > t_{\alpha/2, df}$ (two-sided)
- Or equivalently, reject if p-value $< \alpha$

```{admonition} Example: Website Conversion Rates
Compare time spent on two website designs:
- **Design A**: $N_1 = 50$, $\bar{x}_1 = 45$ seconds, $s_1 = 12$ seconds
- **Design B**: $N_2 = 60$, $\bar{x}_2 = 52$ seconds, $s_2 = 20$ seconds

Variances look quite different, so use Welch's t-test at $\alpha = 0.05$.

**Step 1**: Test statistic
$$t = \frac{45 - 52}{\sqrt{\frac{144}{50} + \frac{400}{60}}} = \frac{-7}{\sqrt{2.88 + 6.67}} = \frac{-7}{\sqrt{9.55}} = \frac{-7}{3.09} \approx -2.27$$

**Step 2**: Degrees of freedom
$$df = \frac{(2.88 + 6.67)^2}{\frac{(2.88)^2}{49} + \frac{(6.67)^2}{59}} = \frac{90.60}{0.169 + 0.754} = \frac{90.60}{0.923} \approx 98.2$$

Round down to $df = 98$.

**Step 3**: Critical value and decision
With $df = 98$ and $\alpha = 0.05$ (two-sided), $t_{0.025, 98} \approx 1.98$

Since $|t| = 2.27 > 1.98$, we reject $H_0$.

P-value $\approx 0.025$

**Conclusion**: At $\alpha = 0.05$, there is significant evidence that average time spent differs between designs. Design B has significantly higher average time.
```

## Which Test Should You Use?

```{mermaid}
graph TD
    A[Do you know σ₁ and σ₂?] -->|Yes| B[Z-test Section 7.2.1]
    A -->|No| C[Are sample sizes large N₁,N₂ ≥ 30?]
    C -->|Yes| B
    C -->|No| D[Do you believe σ₁ = σ₂?]
    D -->|Yes| E[Pooled t-test Section 7.2.2]
    D -->|No or Unsure| F[Welch's t-test Section 7.2.3]
    D -->|Need to test| G[F-test Section 7.3.1 then decide]
```

**General recommendations**:
1. If population standard deviations are known (rare): use z-test
2. If samples are large (N ≥ 30): can use z-test with sample standard deviations
3. If samples are small:
   - Test for equal variances first (F-test)
   - If variances appear equal: pooled t-test
   - If variances differ or you're unsure: Welch's t-test (safer default)

## Python Example: All Three Tests

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate two samples
sample1 = np.random.normal(100, 15, size=30)  # mean=100, std=15
sample2 = np.random.normal(110, 15, size=35)  # mean=110, std=15

n1, n2 = len(sample1), len(sample2)
mean1, mean2 = np.mean(sample1), np.mean(sample2)
std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

print("Sample 1: n={}, mean={:.2f}, std={:.2f}".format(n1, mean1, std1))
print("Sample 2: n={}, mean={:.2f}, std={:.2f}".format(n2, mean2, std2))
print()

# 1. Z-test (assuming we know population std = 15)
print("=" * 50)
print("1. Z-TEST (assuming known σ₁=σ₂=15)")
print("=" * 50)
sigma1, sigma2 = 15, 15
se_z = np.sqrt(sigma1**2/n1 + sigma2**2/n2)
z_stat = (mean1 - mean2) / se_z
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value_z:.4f}")
print(f"Critical values (α=0.05): ±1.96")
if p_value_z < 0.05:
    print("Reject H₀: Means are significantly different")
else:
    print("Fail to reject H₀: Insufficient evidence of difference")
print()

# 2. Pooled t-test (assuming equal variances)
print("=" * 50)
print("2. POOLED T-TEST (assuming σ₁=σ₂)")
print("=" * 50)
t_stat_pooled, p_value_pooled = stats.ttest_ind(sample1, sample2, equal_var=True)
df_pooled = n1 + n2 - 2
t_critical_pooled = stats.t.ppf(0.975, df_pooled)

print(f"T-statistic: {t_stat_pooled:.3f}")
print(f"Degrees of freedom: {df_pooled}")
print(f"P-value: {p_value_pooled:.4f}")
print(f"Critical values (α=0.05): ±{t_critical_pooled:.3f}")
if p_value_pooled < 0.05:
    print("Reject H₀: Means are significantly different")
else:
    print("Fail to reject H₀: Insufficient evidence of difference")
print()

# 3. Welch's t-test (not assuming equal variances)
print("=" * 50)
print("3. WELCH'S T-TEST (not assuming σ₁=σ₂)")
print("=" * 50)
t_stat_welch, p_value_welch = stats.ttest_ind(sample1, sample2, equal_var=False)

# Calculate Welch-Satterthwaite df
var1, var2 = std1**2, std2**2
df_welch = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
t_critical_welch = stats.t.ppf(0.975, df_welch)

print(f"T-statistic: {t_stat_welch:.3f}")
print(f"Degrees of freedom: {df_welch:.1f}")
print(f"P-value: {p_value_welch:.4f}")
print(f"Critical values (α=0.05): ±{t_critical_welch:.3f}")
if p_value_welch < 0.05:
    print("Reject H₀: Means are significantly different")
else:
    print("Fail to reject H₀: Insufficient evidence of difference")
print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Box plots
ax = axes[0, 0]
ax.boxplot([sample1, sample2], labels=['Sample 1', 'Sample 2'])
ax.set_ylabel('Value')
ax.set_title('Box Plots of Both Samples')
ax.grid(True, alpha=0.3)
ax.axhline(mean1, color='C0', linestyle='--', alpha=0.5, label=f'Mean1={mean1:.1f}')
ax.axhline(mean2, color='C1', linestyle='--', alpha=0.5, label=f'Mean2={mean2:.1f}')
ax.legend()

# Plot 2: Histograms
ax = axes[0, 1]
ax.hist(sample1, bins=10, alpha=0.5, label='Sample 1', edgecolor='black')
ax.hist(sample2, bins=10, alpha=0.5, label='Sample 2', edgecolor='black')
ax.axvline(mean1, color='C0', linestyle='--', linewidth=2)
ax.axvline(mean2, color='C1', linestyle='--', linewidth=2)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histograms of Both Samples')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Sampling distribution of difference
ax = axes[1, 0]
se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
x = np.linspace(mean1-mean2-4*se_diff, mean1-mean2+4*se_diff, 200)
y = stats.norm.pdf(x, mean1-mean2, se_diff)
ax.plot(x, y, 'b-', linewidth=2, label='Sampling distribution')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='H₀: μ₁-μ₂=0')
ax.axvline(mean1-mean2, color='green', linestyle='--', linewidth=2, label=f'Observed: {mean1-mean2:.1f}')
ax.fill_between(x[x <= mean1-mean2-1.96*se_diff], y[x <= mean1-mean2-1.96*se_diff], 
                alpha=0.3, color='red')
ax.fill_between(x[x >= mean1-mean2+1.96*se_diff], y[x >= mean1-mean2+1.96*se_diff], 
                alpha=0.3, color='red', label='Rejection regions')
ax.set_xlabel('Difference in Means')
ax.set_ylabel('Density')
ax.set_title('Sampling Distribution of Difference')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: P-values comparison
ax = axes[1, 1]
tests = ['Z-test', 'Pooled t-test', "Welch's t-test"]
p_values = [p_value_z, p_value_pooled, p_value_welch]
colors = ['red' if p < 0.05 else 'gray' for p in p_values]
ax.bar(tests, p_values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
ax.set_ylabel('P-value')
ax.set_title('Comparison of P-values')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i, (test, p) in enumerate(zip(tests, p_values)):
    ax.text(i, p+0.01, f'{p:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('comparing_means_tests.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Key Takeaways

- Use z-test when population standard deviations are known or samples are large
- Use pooled t-test when variances are believed to be equal
- Use Welch's t-test when variances might be different (safer default for small samples)
- All tests have the same null hypothesis: $\mu_1 = \mu_2$
- Rejection of null hypothesis means we have evidence that population means differ
- Always report both the test statistic and p-value, not just "significant" or "not significant"