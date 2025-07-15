Pretraining

Note that the top of the notebook shows you the setting for what's being run. These can be changed. The notebooks above are showing results for: 

1. PAT_Pretraining: Small Model with 25% Masking

```python
## Model size
# eg. ["small", "medium", "large", "huge"]
size = "small"

## Mask ratio
# eg. [.25, .50, .75]
mask_ratio = 0.25

## Smoothing
# eg. [True, False]
smoothing = False

## Loss Function
# eg. [True, False], meaning MSE on only the masked portion or everything in the reconstruction
mse_only_masked = False
```

2. PAT_Conv_Pretraining: Large Model with 90% masking
``` python 
## Model size
# eg. ["small", "medium", "large", "huge"]
size = "large"

## Mask ratio
# eg. [.25, .50, .75]
mask_ratio = 0.90

## Smoothing
# eg. [True, False]
smoothing = False

## Loss Function
# eg. [True, False], meaning MSE on only the masked portion or everything in the reconstruction
mse_only_masked = False
```
