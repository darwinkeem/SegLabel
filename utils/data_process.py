import os
import shutil
from cv2 import cv2


image_list = os.listdir('../data/video_image')
mask_list = os.listdir('../data/video_mask')
print(image_list)
print(mask_list)

for name in mask_list:
    if name != '.gitkeep':
        fold = os.path.join('../data/video_mask/', name)
        image_list = os.listdir(fold)
        print(image_list)
        for i in range(len(image_list)):
            with open("mask_files.txt", "a") as text_file:
                text_file.write('../data/video_mask/'+name+'/'+image_list[i]+'\n')
            with open("vid_files.txt", "a") as text_file:
                text_file.write('../data/video_image/'+name+'/'+image_list[i]+'\n')
