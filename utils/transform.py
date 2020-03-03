import torchvision.transforms as transforms
from cv2 import cv2
from PIL import Image


def resize(im, w, h):
    im = cv2.resize(im, dsize=(int(w), int(h)), interpolation=cv2.INTER_AREA)
    return im

def preprocessing(image, mask):
    image = image / 255.0
    image, mask = resize(image, 512, 256), resize(mask, 512, 256)
    # image, mask = Image.fromarray(image), Image.fromarray(mask)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # image_transform.transforms.insert(0, RandAugment(2, 14))
    
    return image_transform(image).float(), image_transform(mask).float()

def r_preprocessing(image):
    image = image / 255.0
    image = resize(image, 512, 256)
    # image, mask = Image.fromarray(image), Image.fromarray(mask)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # image_transform.transforms.insert(0, RandAugment(2, 14))
    
    return image_transform(image).float()
