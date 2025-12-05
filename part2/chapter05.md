# Chapter 5: Useful Probability Distributions

Most random phenomena we encounter fall into a small number of standard patterns. Rather than deriving probabilities from first principles every time, we can use **standard probability distributions**—well-studied functions that model common situations.

Knowing these distributions is like having a toolkit:
- Recognize patterns in real problems
- Use established formulas instead of deriving from scratch
- Leverage existing theory and computational tools
- Communicate efficiently with other practitioners

## Why Study Probability Distributions?

Consider these scenarios:
1. Number of heads in 100 coin flips
2. Time until the next customer arrives
3. Number of typos on a page
4. Height of randomly selected person
5. Number of failures before first success

Each of these follows a **named probability distribution** with known properties. Once you recognize the pattern, you immediately know:
- The probability mass/density function
- The expected value and variance
- How to generate samples
- What approximations are available

## What You'll Learn

In this chapter, you will learn:

1. **Discrete Distributions** (Section 5.1)
   - Discrete Uniform
   - Bernoulli
   - Geometric
   - Binomial
   - Multinomial
   - Poisson

2. **Continuous Distributions** (Section 5.2)
   - Continuous Uniform
   - Beta
   - Gamma
   - Exponential

3. **The Normal Distribution** (Section 5.3)
   - Standard Normal
   - General Normal
   - Properties and applications
   - The 68-95-99.7 rule

4. **Normal Approximation to Binomial** (Section 5.4)
   - When to use the approximation
   - How to apply it correctly
   - Continuity correction

## Chapter Structure

```{tableofcontents}
```

## Common Distributions Overview

| Distribution | Type | Parameters | Used For |
|--------------|------|------------|----------|
| Uniform | Discrete | $n$ | Equal probabilities |
| Bernoulli | Discrete | $p$ | Single trial (success/fail) |
| Binomial | Discrete | $n, p$ | Number of successes in $n$ trials |
| Geometric | Discrete | $p$ | Trials until first success |
| Poisson | Discrete | $\lambda$ | Count of rare events |
| Uniform | Continuous | $a, b$ | Equal density over interval |
| Exponential | Continuous | $\lambda$ | Time until event |
| Normal | Continuous | $\mu, \sigma^2$ | Natural variation, sums |

## Key Insight

```{admonition} Pattern Recognition
:class: tip
Most of statistics and probability involves:
1. **Recognizing** which distribution fits your problem
2. **Identifying** the parameters from context
3. **Applying** formulas for that distribution
4. **Using** computational tools for calculations

You don't memorize—you recognize and look up!
```

## Prerequisites

Before starting this chapter, you should understand:
- Random variables (Chapter 4.1)
- Expected values and variance (Chapter 4.2)
- Discrete vs. continuous probability

Let's begin by exploring discrete probability distributions!