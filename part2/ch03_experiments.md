# 3.1 Experiments, Outcomes, and Probability

## Experiments

An **experiment** is any process with an uncertain outcome.

Examples:
- Flip a coin
- Roll a die
- Draw a card from a deck
- Measure packet arrival time
- Test software for bugs

## Sample Space

The **sample space** ($\Omega$) is the set of all possible outcomes.

### Examples

**Coin flip**: $\Omega = \{H, T\}$

**Die roll**: $\Omega = \{1, 2, 3, 4, 5, 6\}$

**Two coin flips**: $\Omega = \{HH, HT, TH, TT\}$

```python
import itertools

# Generate sample space for two dice
dice_outcomes = list(itertools.product(range(1, 7), repeat=2))
print(f"Sample space size: {len(dice_outcomes)}")
print(f"First few outcomes: {dice_outcomes[:5]}")
```

## Probability of Outcomes

For **equally likely outcomes**:

$$P(\text{outcome}) = \frac{1}{|\Omega|}$$

### Fair Coin

$P(H) = P(T) = \frac{1}{2}$

### Fair Die

$P(1) = P(2) = \cdots = P(6) = \frac{1}{6}$

## Computing Probabilities

```python
import numpy as np

def simulate_experiment(n_trials):
    """Simulate coin flips"""
    flips = np.random.choice(['H', 'T'], size=n_trials)
    return np.sum(flips == 'H') / n_trials

# Law of large numbers in action
for n in [10, 100, 1000, 10000]:
    prob_heads = simulate_experiment(n)
    print(f"n={n:5d}: P(H) = {prob_heads:.4f}")
```

## Probability Properties

1. **Non-negative**: $P(A) \geq 0$

2. **Normalized**: $P(\Omega) = 1$

3. **Additive**: For disjoint events $A$ and $B$:
   $$P(A \cup B) = P(A) + P(B)$$

4. **Complement**: $P(A^c) = 1 - P(A)$

## Example: Rolling Dice

```python
# Probability of rolling a sum of 7 with two dice
sample_space = [(i, j) for i in range(1, 7) for j in range(1, 7)]
favorable = [(i, j) for i, j in sample_space if i + j == 7]

prob = len(favorable) / len(sample_space)
print(f"P(sum=7) = {len(favorable)}/{len(sample_space)} = {prob:.4f}")
print(f"Favorable outcomes: {favorable}")
```

## Key Takeaways

1. Sample space contains all possible outcomes
2. For equally likely outcomes: count and divide
3. Probabilities sum to 1
4. Use simulation to verify calculations

→ Next: [3.2 Events](ch03_events.md)