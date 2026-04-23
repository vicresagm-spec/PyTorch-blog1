---
title: Scriptable and Non Scriptable
date: 2026-04-22 10:00:00 +0400
categories: [torchvision.transform]
tags: [nonscriptable,scriptable ]
---





# Scriptable vs Non-Scriptable Transforms

## Comparison Table

| Scriptable | Non-Scriptable |
| --- | --- |
| In order to script the transformations, use `torch.nn.Sequential` instead of `Compose` | Uses **Python-specific things** |
| Uses only torch operations | Often depends on **PIL images** |
| Avoids Python-only logic | Cannot be converted with TorchScript |
| Can be compiled using `torch.jit.script(transform)` | For training in notebooks |
| Portable and production-ready | |
| For deployment pipelines | |

## Why Script Transforms?

When we deploy a model, we want to add the preprocessing along with it. Everything inside TorchScript (no Python dependency).

### Scriptable Example

```python
import torch
import torchvision.transforms as T

class MyTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = T.Resize((64, 64))  # works only if input is tensor
    
    def forward(self, x):
        return self.resize(x)

scripted = torch.jit.script(MyTransform())
```

### Non-Scriptable Example

```python
transforms.Compose([
    transforms.Resize((64, 64)),  # uses PIL
    transforms.ToTensor()
])
# only for notebooks
```

## Hidden Trap Beginners Miss

Even if a transform *looks* simple:

```python
transforms.RandomHorizontalFlip()
```

It may **NOT** be fully scriptable because of randomness handling.

**Safer alternative:** Use tensor operations manually.

## The Real Issue in One Line

Your **training pipeline** is usually:

```
file path → PIL image → Python transforms → tensor → model
```

But **exported pipelines** prefer:

```
tensor → tensor transforms → model
```

**That is the core difference.**

## What to Do Instead

Separate your pipeline into 2 parts:


### Training Pipeline

**Allowed:**
- PIL
- `ImageFolder`
- `Compose`
- Random flip
- Random crop
- Python convenience tools

### Export/Deployment Pipeline

**Preferred:**
- Tensor input
- `nn.Module`
- Tensor resize/normalize
- No PIL dependence
- No training augmentation

## Example: Export-Friendly Preprocessing Module

```python
import torch
import torch.nn as nn
import torchvision.transforms as T

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize((64, 64)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    def forward(self, x):
        return self.transforms(x)
```

## Transform Types

Most transformations accept both [PIL](https://pillow.readthedocs.io/) images and tensor images, while:
- Some transformations are **PIL-only**
- Some transformations are **tensor-only**

The [Conversion Transforms](https://docs.pytorch.org/vision/0.11/transforms.html) may be used to convert to and from PIL images.

### Tensor Image Format

- A **Tensor Image** is a tensor with shape `(C, H, W)`
- A **batch of Tensor Images** is a tensor with shape `(B, C, H, W)`

## Functional Transforms

A **Functional Transform** has:
- No randomness
- No probability
- We have to decide when to apply it
- Will never do anything random automatically

### Example

Instead of:

```python
transforms.RandomHorizontalFlip(p=0.5)
```

Use:

```python
import random

if random.random() < 0.5:
    img = F.hflip(img)
```

This gives you explicit control over when randomness is applied.
