import os
import imageio

fps = 30
output = './xor/training.gif'

png_dir = 'xor/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

last = images[-1]
for i in range(fps * 2):
    images.append(last)

imageio.mimsave(output, images, fps=fps)

from pygifsicle import optimize

optimize(output, './xor/training_optimized.gif', options=['--optimize=3', '-v']) # compress gif
