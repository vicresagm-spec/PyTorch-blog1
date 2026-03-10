---
title: What Makes a Layer Trainable in PyTorch?
date: 2026-03-8 10:00:00 +0400
categories: [pytorch, basics]
tags: [nn-module]
---

# Trainable Layers in Neural Networks (PyTorch)

A **trainable layer** is any layer that has parameters (numbers) that get updated during training using **backpropagation**.

If a layer contains **weights or bias that change during training → it is trainable**.  
If it **has no parameters → it is not trainable**.

---

## What does “trainable” really mean?

During training, the following steps happen:

1. The model makes a **prediction**
2. The **loss** is computed
3. **Gradients** are calculated using backpropagation
4. Some **parameters are updated**

👉 Only layers that contain **parameters** can be updated.  
👉 These layers are called **trainable layers**.

---

## The simplest test (very important)

Ask this question:

**Does this layer have weights or bias?**

- ✅ **Yes → Trainable**
- ❌ **No → Not trainable**

That’s it.

---

# Common PyTorch layers — trainable or not?
## Trainable layers (learn parameters)

| Layer | Why |
|------|-----|
| `nn.Linear` | Has weight and bias |
| `nn.Conv2d` | Has kernels (weights) |
| `nn.BatchNorm` | Has learnable scale and shift |
| `nn.Embedding` | Contains an embedding matrix |
| Custom layer with `nn.Parameter` | Explicitly defined parameters |

Example:

```python
self.fc = nn.Linear(10, 5)
Internally this layer contains:

```
weight: (5, 10)
bias: (5,)
```

These parameters are updated during training.

---

## Non-trainable layers (no learning)

| Layer | Why |
|------|-----|
| `nn.ReLU` | Just applies `max(0, x)` |
| `nn.Sigmoid` | Mathematical function |
| `nn.Dropout` | Random masking |
| `nn.MaxPool` | Selects maximum value |
| `nn.Flatten` | Changes tensor shape |

Example:

```python
x = torch.relu(x)
```

Here:

- No parameters exist  
- Nothing gets updated  
- It still plays an important role in the network  

---

## Why this matters for `forward()`

### Rule to remember

All **trainable layers must be defined in `__init__` and referenced using `self`.**

Example:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Why this is necessary

- The optimizer must **find parameters through `self`**
- Gradients must be **tracked**
- Weights must be **updated**

---

## Common beginner mistake

Defining trainable layers **inside `forward()`**.

❌ Incorrect example:

```python
def forward(self, x):
    fc = nn.Linear(2, 4)   # BAD
    return fc(x)
```

### Why this breaks training

- New weights are created every forward pass  
- The optimizer never sees them  
- The model cannot learn  

---

## Activations confuse many beginners

Many people ask:

> “ReLU is a layer — is it trainable?”

❌ **No.**

ReLU is:

- an **operation**
- not a **learning unit**

Think of it like a **rule applied to numbers**.

| Component | Role |
|----------|------|
| Trainable layers | Store learnable parameters |
| Activation functions | Apply mathematical behavior |

---

## Visual mental model

Imagine a neural network as a **factory**.

### Machines that learn (trainable)

- Adjustable knobs (**weights**)  
- Workers adjust them during training  

### Machines that process (non-trainable)

- Fixed rules  
- Never change  

Forward pass = **assembly line**  
Training = **adjusting the knobs**

---

## How to check trainable layers in code

You can inspect parameters like this:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

If it appears here → it is **trainable**.

---

## Connecting this to hidden layers

When people say:

> "One hidden layer = Linear + ReLU"

What they really mean is:

| Component | Type |
|----------|------|
| `nn.Linear` | Trainable |
| `ReLU` | Non-trainable |

Together they form **one functional hidden layer**.

---

## Final takeaway

**Trainable layers are the parts of the network that contain parameters (weights or bias) and get updated during training.**

If a component **does not contain parameters**, it is **not trainable**, even if it is called a "layer".