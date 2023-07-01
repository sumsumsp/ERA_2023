## Target 
- Architecture  C1C2C3C40 (No MaxPooling)
- Total RF: 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use argumentation library and apply: horizontal flip, shiftScaleRotate, coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- Achieve 85% accuracy (Params to be less than 200k).

## Data Augmentation 
