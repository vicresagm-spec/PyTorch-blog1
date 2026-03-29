---
title: Loading  the Pytorch Models (pattern to remember )
date: 2026-03-29 10:00:00 +0400
categories: [pytorch, basics]
tags: [Loading and Saving ]
---


![Save and Load Together ](/assets/savload.png)

## Step 1: Rebuild the empty model shell
``` Python
loaded_model_2 = FashionMNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
)
```

This creates a blank model — same architecture as the one you saved, but with random untrained weights. Think of it as printing an empty form before filling it in.

*These values must match the saved model exactly. If you trained with hidden_units=10 but load with hidden_units=20, PyTorch will crash — the weight shapes won't fit.*


## Step 2: Pour the saved weights in

``` python
loaded_model_2.load_state_dict(
    torch.load(f=MODEL_SAVE_PATH)
)

```

Two functions working together — read them inside-out:
`torch.load(...)`
reads the `.pth` file from disk → returns a dictionary of weights
`load_state_dict(...)`
takes that dictionary and pours the weights into the empty model shell
`f=MODEL_SAVE_PATH`
the Path object pointing to your saved .pth file


## Step 3: A  quick sanity check

`loaded_model_2` 
Typing the variable alone in a notebook prints a summary of the model's layers. It's a quick sanity check — does this look like what I trained? It does NOT show you the weight values, just the architecture.