# Deep Learning Visualisation

Plotting learning curves and decision boundaries.

- `train.py` contains the model and generates one PNG image per epoch
- `convert.py` takes all the images and creates a GIF file (30 fps, stays on last frame longer) 

Note: `gifsicle` is used to optimize the GIF and needs to be in your path

## XOR

![Training](xor/training_optimized.gif)