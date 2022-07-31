"""
Run script to use sonocrop to mask static pixels in an ultrasound. Loop over all .mp4 data in patient folders.
https://github.com/davycro/sonocrop
"""

import os
from sonocrop import vid
from sonocrop.cli import mask, crop
import cv2
from matplotlib import pyplot as plt

def main():
  root = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/"

  for i in range(1,123):

    folder = root + "Case-" + str(i).zfill(3)

    if os.path.exists(folder):
      try:
        for file in os.listdir(folder):

          if file.endswith(".mp4") and not file.endswith("masked.mp4"):
            input_file = os.path.join(folder, file)
            # output_file1 = input_file.replace('.mp4', '_cropped.mp4')
            output_file = os.path.join(folder, file.replace('.mp4', '_masked.mp4'))

            v, fps, f, height, width = vid.loadvideo(input_file)
            # print(f)
            thresh = 5/f+0.0001
            
            # crop(input_file, output_file1, thresh=thresh)
            mask(input_file, output_file, thresh=thresh, save_mask=True)
            print(f'{input_file} -> {output_file}')

          else:
            continue
      except:
        print(file + "does not exist.")

#-------------------------------------- testing threshold --------------------------------------

def get_mask(input_file):

    v, fps, f, height, width = vid.loadvideo(input_file)
    # print(f)

    thresh = 5/f+0.0001

    masked = mask(input_file, 0, thresh=thresh, save_mask=False)


    # optionally, save output file
    # output_file = input_file.replace('.mp4', '_mask.png')
    # plt.imsave(output_file, masked, cmap='gray')

    # return output_file

    return masked

# test cases

# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-122/BEDLUS_122_011.mp4
# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-103/BEDLUS_103_001.mp4
# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-052/test.mp4
# input_file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-013/BEDLUS_013_011.mp4"

# plt.imshow(get_mask(input_file), cmap='gray')
# plt.show()

# main()