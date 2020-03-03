import os
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--train', type=int
)
parser.add_argument(
    '--val', type=int
)
parser.add_argument(
    '--test', type=int
)

cfg = parser.parse_args()
print(cfg)

mask_list = os.listdir('../data/video_mask/')
mask_list.remove('.gitkeep')

sub_image = []
sub_mask = []

for name in mask_list:
    if name != '.gitkeep':
        fold = os.path.join('../data/video_mask/', name)
        image_list = os.listdir(fold)
        print(image_list)
        for i in range(len(image_list)):
            with open("mask_files.txt", "a") as text_file:
                sub_image.append('../data/video_mask/'+name+'/'+image_list[i]+'\n')
            with open("vid_files.txt", "a") as text_file:
                sub_mask.append('../data/video_image/'+name+'/'+image_list[i]+'\n')

train_image = []
val_image = []
test_image = []

random.shuffle(sub_mask)
print(len(sub_mask))

train_split = int(len(sub_mask) * (cfg.train / 100))
val_split = int(len(sub_mask) * (cfg.val / 100))
test_split = int(len(sub_mask) * (cfg.test / 100))

for i in range(0, len(sub_mask)):
    if i < train_split:
        train_image.append(sub_mask[i])
    elif i >= train_split and i < train_split + val_split:
        val_image.append(sub_mask[i])
    else:
        test_image.append(sub_mask[i])

with open('../data/split/train_file.txt', 'w') as o:
    for filename in train_image:
        tmp = filename.split("/")[-1]
        tmp = tmp[:-1]
        tmp2 = filename.split("/")[-2]
        print(f'./data/video_image/{tmp2}/{tmp} ./data/video_mask/{tmp2}/{tmp}', file=o)

with open('../data/split/val_file.txt', 'w') as o:
    for filename in val_image:
        tmp = filename.split("/")[-1]
        tmp = tmp[:-1]
        tmp2 = filename.split("/")[-2]
        print(f'./data/video_image/{tmp2}/{tmp} ./data/video_mask/{tmp2}/{tmp}', file=o)

with open('../data/split/test_file.txt', 'w') as o:
    for filename in test_image:
        tmp = filename.split("/")[-1]
        tmp = tmp[:-1]
        tmp2 = filename.split("/")[-2]
        print(f'./data/video_image/{tmp2}/{tmp} ./data/video_mask/{tmp2}/{tmp}', file=o)

print(len(train_image))
print(len(val_image))
print(len(test_image))
print(len(train_image) + len(val_image) + len(test_image))