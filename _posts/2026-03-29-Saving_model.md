---
title: Saving the Pytorch Models Weights (pattern to remember )
date: 2026-03-29 10:00:00 +0400
categories: [pytorch, basics]
tags: [Loading and Saving ]
---


![Save and Load Together ](/assets/savload.png)



## Save Script 

``` python 

import pathlib
from pathlib import Path

# create model Directory Pth
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

# create model save
MODEL_NAME ="filename .pth"
MODEL_SAVE_PATH =MODEL_PATH/MODEL_NAME

# save the model state dict
print(f" Saving model to :{MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

```           


## Step 1: Importing pathlib 
``` python 
import pathlib
from pathlib import Path
```

The frist imports the whole module.The second gives you the `PATH` class 
##  Step 2: Create the Folder
```python
MODEL_PATH =Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

```
`Path("models") `creates a Path object — think of it as a smart string that knows it's a file path.

`.mkdir()` physically creates the folder on disk. The two arguments are the key:
`parents=True`
Create any missing parent folders too. e.g. models/saved/v2 — creates all three levels at once instead of crashing.
`exist_ok=True`
Don't crash if the folder already exists. Without this, running the code twice throws an error.

##  Step 3: Build the Save Path

``` python
MODEL_NAME = "filename.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
```
The` /` operator is pathlib's superpower — it joins paths the same way `os.path.join()` does, but reads like normal math. It works on all operating systems.
``` python
Path("models")+/+"model.pth"→models/model.pth

```
Why separate MODEL_NAME and MODEL_PATH? So you can change the folder or the filename independently without rewriting both. Clean, reusable pattern.

## Step 4:Save the Model
``` python 
print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)
           
```
`model_2.state_dict() `Returns only the learned weights (numbers) — not the whole model class. Lighter and the recommended way to save in PyTorch.
`obj`=
The thing being saved — the weights dictionary.
`f`=
The file destination — your Path object works here directly.

# The full mental pattern 

1. Make a Path object → Path("folder")
2. Create the folder → .mkdir(parents=True,exist_ok=True)
3. Build the file path → folder_path / filename.pth
4. Save to it → torch.save(obj=..., f=path)