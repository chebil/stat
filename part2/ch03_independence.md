# 3.3 Independence

## Definition

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Intuitively: knowing $B$ occurred doesn't change probability of $A$.

## Examples

### Independent Events

**Two coin flips**: Result of first flip doesn't affect second

```python
import numpy as np

# Simulate to verify independence
n_trials = 10000
flip1 = np.random.choice(['H', 'T'], n_trials)
flip2 = np.random.choice(['H', 'T'], n_trials)

prob_hh = np.sum((flip1 == 'H') & (flip2 == 'H')) / n_trials
prob_h = np.sum(flip1 == 'H') / n_trials

print(f"P(H1 and H2) = {prob_hh:.4f}")
print(f"P(H1) × P(H2) = {prob_h:.4f} × {prob_h:.4f} = {prob_h**2:.4f}")
print(f"Independent: {np.isclose(prob_hh, prob_h**2, atol=0.01)}")
```

**Output:**
```
P(H1 and H2) = 0.2457
P(H1) × P(H2) = 0.4978 × 0.4978 = 0.2478
Independent: True
```


### NOT Independent

**Drawing cards without replacement**

```python
# First card is Ace, second card is Ace (no replacement)
prob_first_ace = 4/52
prob_second_ace_given_first = 3/51

prob_both_aces = prob_first_ace * prob_second_ace_given_first
print(f"P(both Aces) = {prob_both_aces:.6f}")

# If independent (wrong!)
wrong_calc = (4/52) * (4/52)
print(f"If independent: {wrong_calc:.6f} (WRONG!)")
```

**Output:**
```
P(both Aces) = 0.004525
If independent: 0.005917 (WRONG!)
```


## Multiplication Rule

For independent events $A_1, A_2, \ldots, A_n$:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2) \cdot \ldots \cdot P(A_n)$$

## Application: Reliability

```python
# System with 3 independent components
# Each has 95% reliability

comp_reliability = 0.95
num_components = 3

# Series system: all must work
series_reliability = comp_reliability ** num_components
print(f"Series system reliability: {series_reliability:.4f}")

# Parallel system: at least one must work
parallel_reliability = 1 - (1 - comp_reliability) ** num_components
print(f"Parallel system reliability: {parallel_reliability:.4f}")
```

**Output:**
```
Series system reliability: 0.8574
Parallel system reliability: 0.9999
```


## Birthday Problem

What's the probability that in a group of $n$ people, at least two share a birthday?

```python
def birthday_probability(n):
    """Probability at least 2 people share birthday"""
    if n > 365:
        return 1.0
    
    # Probability all different
    prob_all_different = 1.0
    for i in range(n):
        prob_all_different *= (365 - i) / 365
    
    # At least one match
    return 1 - prob_all_different

# Famous result: 23 people → ~50%
for n in [10, 20, 23, 30, 50]:
    prob = birthday_probability(n)
    print(f"n={n:2d}: P(match) = {prob:.4f} = {prob:.1%}")
```

**Output:**
```
n=10: P(match) = 0.1169 = 11.7%
n=20: P(match) = 0.4114 = 41.1%
n=23: P(match) = 0.5073 = 50.7%
n=30: P(match) = 0.7063 = 70.6%
n=50: P(match) = 0.9704 = 97.0%
```


## Testing Independence

To check if $A$ and $B$ are independent:

1. Calculate $P(A \cap B)$
2. Calculate $P(A) \cdot P(B)$
3. Compare: If equal, independent

```python
def test_independence(prob_a, prob_b, prob_ab):
    """Test if events are independent"""
    expected = prob_a * prob_b
    is_independent = np.isclose(prob_ab, expected, atol=1e-6)
    
    print(f"P(A ∩ B) = {prob_ab:.6f}")
    print(f"P(A) × P(B) = {expected:.6f}")
    print(f"Independent: {is_independent}")
    
    return is_independent

# Example
test_independence(0.5, 0.5, 0.25)  # Two fair coins
```

**Output:**
```
P(A ∩ B) = 0.250000
P(A) × P(B) = 0.250000
Independent: True
```


## Key Takeaways

1. Independence: $P(A \cap B) = P(A) \cdot P(B)$
2. Multiple events: multiply probabilities
3. NOT independent: drawing without replacement
4. Check independence before multiplying

→ Next: [3.4 Conditional Probability](ch03_conditional.md)