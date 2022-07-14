"""
Convert .npy to .png
"""

import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# from utils.config import images_folder

file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/arrays/unprocessed_clip_arrays/Case001/clip_20.npy"

clip_array = np.load(file)
img_array = np.asarray(clip_array[0,:,:], dtype=np.float64)

plt.imshow(img_array, cmap='gray') # only get first image in clip
# plt.show()

# output_folder = os.path.join(images_folder, 'example')
# plt.savefig(os.path.join(output_folder, os.path.basename(file).replace('.mp4', '.png')))
output_folder = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/images/example/clip_20.png"
mpimg.imsave(output_folder, img_array, cmap="gray")

output_folder2 = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/images/example/clip_20_color.png"
mpimg.imsave(output_folder2, img_array)