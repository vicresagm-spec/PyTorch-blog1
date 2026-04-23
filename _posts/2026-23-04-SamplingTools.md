---
title: Sampling Tool
date: 2026-04-22 10:00:00 +0400
categories: [Sampling tools]
tags: [Python,PyTorch,Numpy,Sclearn..... ]
---




# Sampling Tools

A sampling tool is a method or piece of code that picks a subset of data from a larger dataset using some randomness.

Example: A big bowl = 10,000 marbles  
Pick some marbles → sampling.  
Tools used to pick them are called sampling tools.

## Programming Context

### Example 1: Sampling Tool is a Function that Selects Values

#### Python Sampling Tool

```python
import random

data = [1, 2, 3, 4, 5]

random.choice(data)  # only 1 element
random.sample(data, 3)  # pick 3 elements (no replacement)
random.choices(data, k=3)  # 3 elements with replacement
```

#### NumPy Sampling Tool

From dataset:

```python
import numpy as np

data = [1, 2, 3, 4, 5]

np.random.choice(data, 3)  # selects 3
```

### Example 2: Sampling from a Distribution

```python
np.random.randn(5)  # creates 5 samples
```

#### SciPy Sampling Tools

```python
from scipy.stats import norm

norm.rvs(size=5)
```

#### Pandas Sampling

```python
import pandas as pd

df.sample(n=5)  # random rows
df.sample(frac=0.2)  # 20% of data
```

#### PyTorch Sampling Tools

```python
from torch.utils.data import DataLoader

DataLoader(dataset, batch_size=32, shuffle=True)
```

#### Explicit Samplers

```python
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(weights, num_samples=100)
```

**Use when:**
- Class imbalance problem
- Controlled sampling

#### Scikit-learn Sampling Tools

```python
from sklearn.utils import resample

resample(data, replace=True, n_samples=100)
```

## The Need to Use Sampling Tools

### 1. To Simulate Randomness
- Coin toss
- Dice roll
- Probability experiment

### 2. To Train ML Models Efficiently
- Used in DataLoaders

### 3. To Create New Datasets
```python
np.random.choices().mean()
```

### 4. Data Augmentation
```python
random.choice(self.angles)
```

## Summary

A sampling tool is **any method that selects or generates data points from a larger set, often using randomness, to simulate, train, or analyze systems**.
