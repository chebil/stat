# Chapter 3: Basic Ideas in Probability

## Overview

Probability theory provides the foundation for statistical inference, machine learning, and understanding uncertainty in computer science.

## Learning Objectives

- Understand experiments, outcomes, and probability
- Calculate probabilities using set theory
- Apply independence rules
- Master conditional probability
- Avoid common probability fallacies

## Why Probability?

Computer scientists use probability for:
- **Randomized algorithms**: Quicksort, hashing
- **Machine learning**: Probabilistic models
- **Cryptography**: Secure random number generation
- **Networks**: Packet loss and reliability
- **AI**: Decision making under uncertainty

## Chapter Structure

### [3.1 Experiments and Outcomes](ch03_experiments.md)
- Sample spaces
- Outcomes and events
- Probability axioms

### [3.2 Events](ch03_events.md)
- Computing event probabilities
- Set operations: union, intersection, complement
- Counting techniques

### [3.3 Independence](ch03_independence.md)
- Independent events
- Multiplication rule
- Applications

### [3.4 Conditional Probability](ch03_conditional.md)
- Definition and interpretation
- Bayes' theorem
- The Monty Hall problem
- Prosecutor's fallacy

## Key Concepts

**Experiment**: Process with uncertain outcome

**Sample Space** ($\Omega$): Set of all possible outcomes

**Event**: Subset of sample space

**Probability**: $P(A)$ where $0 \leq P(A) \leq 1$

### Axioms of Probability

1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$
3. For disjoint events: $P(A \cup B) = P(A) + P(B)$

## Getting Started

→ Begin with [3.1 Experiments and Outcomes](ch03_experiments.md)