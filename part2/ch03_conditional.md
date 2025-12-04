# 3.4 Conditional Probability

## Definition

The **conditional probability** of $A$ given $B$ is:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Reads: "Probability of $A$ given that $B$ has occurred"

## Intuition

Conditional probability restricts the sample space to $B$.

```python
import numpy as np

# Example: Die roll
# A = "roll is even" = {2, 4, 6}
# B = "roll > 3" = {4, 5, 6}

omega = set(range(1, 7))
A = {2, 4, 6}
B = {4, 5, 6}

prob_a = len(A) / len(omega)
prob_b = len(B) / len(omega)
prob_a_and_b = len(A & B) / len(omega)

prob_a_given_b = prob_a_and_b / prob_b
print(f"P(A|B) = {prob_a_given_b:.4f}")
print(f"Given roll > 3, restricted to {B}")
print(f"Even outcomes in B: {A & B}")
print(f"So P(even | >3) = 2/3 = {2/3:.4f}")
```

## Multiplication Rule

Rearranging the definition:

$$P(A \cap B) = P(B) \cdot P(A|B)$$

Or equivalently:

$$P(A \cap B) = P(A) \cdot P(B|A)$$

## Bayes' Theorem

**Bayes' Theorem** relates $P(A|B)$ to $P(B|A)$:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Application: Medical Testing

```python
# Disease prevalence
prob_disease = 0.001  # 0.1% of population

# Test accuracy
prob_pos_given_disease = 0.99  # Sensitivity
prob_neg_given_healthy = 0.95  # Specificity

prob_pos_given_healthy = 1 - prob_neg_given_healthy
prob_healthy = 1 - prob_disease

# Total probability of positive test
prob_pos = (prob_pos_given_disease * prob_disease + 
            prob_pos_given_healthy * prob_healthy)

# Bayes' theorem: P(disease | positive)
prob_disease_given_pos = (prob_pos_given_disease * prob_disease) / prob_pos

print(f"P(disease | +test) = {prob_disease_given_pos:.4f}")
print(f"Only {prob_disease_given_pos:.1%} chance of disease even with positive test!")
```

## The Monty Hall Problem

Famous probability puzzle:

1. Three doors: one has a car, two have goats
2. You pick door 1
3. Host opens door 3, showing a goat
4. Should you switch to door 2?

**Answer**: YES! Switching doubles your probability.

```python
import random

def monty_hall_simulation(n_trials, switch=True):
    """Simulate Monty Hall problem"""
    wins = 0
    
    for _ in range(n_trials):
        # Setup
        car = random.randint(1, 3)
        choice = 1  # Always choose door 1
        
        # Host opens a door with goat
        doors = {1, 2, 3}
        doors.remove(choice)
        if car in doors:
            doors.remove(car)
        host_opens = doors.pop()
        
        # Switch or stay
        if switch:
            remaining = {1, 2, 3} - {choice, host_opens}
            choice = remaining.pop()
        
        if choice == car:
            wins += 1
    
    return wins / n_trials

# Simulate
n = 10000
prob_win_stay = monty_hall_simulation(n, switch=False)
prob_win_switch = monty_hall_simulation(n, switch=True)

print(f"Win rate (stay):   {prob_win_stay:.4f}")
print(f"Win rate (switch): {prob_win_switch:.4f}")
print(f"\nSwitching is {prob_win_switch/prob_win_stay:.1f}x better!")
```

### Explanation

```
Initial choice correct: 1/3
  - Host shows goat from other two
  - Switching loses

Initial choice wrong: 2/3
  - Host must show the other goat
  - Switching wins!

P(win | switch) = 2/3
P(win | stay) = 1/3
```

## Law of Total Probability

If $B_1, B_2, \ldots, B_n$ partition the sample space:

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

## Prosecutor's Fallacy

**Fallacy**: Confusing $P(evidence | innocent)$ with $P(innocent | evidence)$

Example:
- DNA match occurs in 1 in 1,000,000 people
- City has 8,000,000 people
- Expected 8 matches!
- Finding match doesn't mean probability of guilt is 999,999/1,000,000

## Independence and Conditioning

Events $A$ and $B$ are independent if and only if:

$$P(A|B) = P(A)$$

Knowing $B$ doesn't change probability of $A$.

## Key Takeaways

1. $P(A|B) = \frac{P(A \cap B)}{P(B)}$
2. Use Bayes' theorem to "reverse" conditionals
3. Monty Hall: counterintuitive but correct
4. Don't confuse $P(A|B)$ with $P(B|A)$!
5. Rare events: base rates matter

## Practice

1. Medical testing with different prevalence rates
2. Spam filtering using Bayes
3. More conditional probability puzzles

→ Next: [Chapter 4: Random Variables](../part2/chapter04.md)