"""

Convert to right folder structure for patient folders.

Input: masked_clips folder
Output: LUS_videos_full (with ~200 patient folders)
"""

import os
 
source = 'C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/masked_clips/'
destination = 'C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos_full/'
 
allfiles = os.listdir(source)

for f in allfiles:
    
    # split by '_'
    folder_name, index = f.split('_')
    # print(folder_name)
    # print(source + f)

    if not os.path.exists(destination + folder_name):
        os.makedirs(destination + folder_name)

    os.rename(source + f, destination + folder_name + '/' + f)