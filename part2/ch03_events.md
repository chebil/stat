# 3.2 Events

## What is an Event?

An **event** is a subset of the sample space.

### Examples

**Die roll**: $\Omega = \{1, 2, 3, 4, 5, 6\}$

Events:
- $A$ = "roll is even" = $\{2, 4, 6\}$
- $B$ = "roll is greater than 4" = $\{5, 6\}$
- $C$ = "roll is 1" = $\{1\}$

## Computing Event Probabilities

For equally likely outcomes:

$$P(A) = \frac{|A|}{|\Omega|}$$

```python
# Example: Probability of drawing an Ace
deck_size = 52
num_aces = 4

prob_ace = num_aces / deck_size
print(f"P(Ace) = {prob_ace:.4f} = {prob_ace:.1%}")
```

**Output:**
`P(Ace) = 0.0769 = 7.7%`


## Set Operations on Events

### Union (OR)

$A \cup B$: Event that $A$ OR $B$ occurs

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

```python
# Die roll: even OR greater than 4
even = {2, 4, 6}
greater_4 = {5, 6}

union = even | greater_4  # {2, 4, 5, 6}
intersection = even & greater_4  # {6}

print(f"A ∪ B = {union}")
print(f"A ∩ B = {intersection}")
print(f"P(A ∪ B) = {len(union)/6:.4f}")
```

**Output:**
```
A ∪ B = {2, 4, 5, 6}
A ∩ B = {6}
P(A ∪ B) = 0.6667
```


### Intersection (AND)

$A \cap B$: Event that both $A$ AND $B$ occur

### Complement (NOT)

$A^c$: Event that $A$ does NOT occur

$$P(A^c) = 1 - P(A)$$

## Counting Techniques

### Multiplication Principle

If task 1 has $n_1$ outcomes and task 2 has $n_2$ outcomes:

Total outcomes = $n_1 \times n_2$

```python
# Password: 4 digits
num_digits = 10
password_length = 4
total_passwords = num_digits ** password_length
print(f"Possible 4-digit passwords: {total_passwords:,}")
```

**Output:**
`Possible 4-digit passwords: 10,000`


### Permutations

Ordering $k$ items from $n$ items:

$$P(n, k) = \frac{n!}{(n-k)!}$$

```python
from math import factorial

def permutations(n, k):
    return factorial(n) // factorial(n - k)

# Arrange 3 books from 5
print(f"P(5,3) = {permutations(5, 3)}")
```

**Output:**
`P(5,3) = 60`


### Combinations

Choosing $k$ items from $n$ items (order doesn't matter):

$$C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$$

```python
from math import comb

# Choose 2 cards from 5
print(f"C(5,2) = {comb(5, 2)}")
```

**Output:**
`C(5,2) = 10`


## Example: Poker Hand

```python
# Probability of a flush (5 cards same suit)
total_hands = comb(52, 5)

# 4 suits, choose 5 from 13 cards in each suit
flush_hands = 4 * comb(13, 5)

prob_flush = flush_hands / total_hands
print(f"P(flush) = {prob_flush:.6f} = {prob_flush:.4%}")
```

**Output:**
`P(flush) = 0.001981 = 0.1981%`


## Key Formulas

1. **Addition Rule**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

2. **Complement**: $P(A^c) = 1 - P(A)$

3. **Disjoint Events**: If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$

→ Next: [3.3 Independence](ch03_independence.md)