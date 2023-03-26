from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image, ImageDraw

from xml.dom import minidom
import os 

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class DLIB(Dataset):
    data_dir = 'data/DLIB/'
    data_url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"

    def __init__(self, data_dir: Optional[str] = None, data_url: Optional[str] = None, 
                    root: str = 'ibug_300W_large_face_landmark_dataset/') -> None:
        super().__init__()

        self.root = self.data_dir + root

        if data_url:
            self.data_url = data_url
        if data_dir:
            self.data_dir = data_dir

        self.data = minidom.parse(self.root + 'labels_ibug_300W.xml')
        self.data = self.data.getElementsByTagName('image')

    def __getitem__(self, idx: int):
        image = self.data[idx]
        image_path = image.getAttribute('file')
        image_width = int(image.getAttribute('width'))
        image_height = int(image.getAttribute('height'))

        keypoints = []
        for points in image.getElementsByTagName('part'):
            keypoints.append((int(points.getAttribute('x')), int(points.getAttribute('y'))))

        bbox = image.getElementsByTagName('box')[0]
        x_min = int(bbox.getAttribute('left'))
        y_min = int(bbox.getAttribute('top'))
        x_max = x_min + int(bbox.getAttribute('width'))
        y_max = y_min + int(bbox.getAttribute('height'))

        # image = read_image(path=self.root + image_path, mode=ImageReadMode.RGB) # channel x height x width
        
        # image = image.permute(1, 2, 0).numpy() # height x width x channel

        # # in case if bounding box is outside image
        # # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
        # # np.pad(image, [(top, bot), (left, right), (front, back)])
        # # print("Before padded:", image.shape)
        # if x_max > image_width:
        #     image = np.pad(image, [(0, 0), (0, x_max - image_width), (0, 0)], mode='constant', constant_values=255)
        # if y_max > image_height:
        #     image = np.pad(image, [(0, y_max - image_height),(0, 0), (0, 0)], mode='constant', constant_values=255)
        # if x_min < 0:
        #     image = np.pad(image, [(0, 0), (x_min * -1, 0), (0, 0)], mode='constant', constant_values=255)
        #     keypoints = keypoints - np.array([x_min, 0]) 
        #     x_max -= x_min
        #     x_min = 0
        # if y_min < 0:
        #     image = np.pad(image, [(y_min * -1, 0),(0, 0), (0, 0)], mode='constant', constant_values=255)
        #     keypoints = keypoints - np.array([0, y_min])
        #     y_max -= y_min
        #     y_min = 0

        # transform = A.Compose([A.Crop(x_min=x_min, y_min=y_min, 
        #                             x_max=x_max, y_max=y_max),
        #                         A.Resize(224, 224),
        #                         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #                         ToTensorV2()
        #                         ],
        #                             keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # transformed = transform(image=image, keypoints=keypoints)
        # image = transformed['image']
        # keypoints = transformed['keypoints']

        image: Image = Image.open(self.root + image_path).convert('RGB')
        image: Image = image.crop((x_min, y_min, x_max, y_max))
        image = np.array(image)

        keypoints = (keypoints / np.array([224, 224]) - 0.5).astype(np.float32) # range [-0.5; 0.5]  

        return image, keypoints
    
    def __len__(self):
        return len(self.data)


class DLIBTransform(Dataset):
    def __init__(self, data: DLIB, transform: Optional[A.Compose] = None):
        self.data = data
        
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                                A.Resize(224, 224),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ],
                                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __getitem__(self, idx):
        image, keypoints = self.data[idx]
        keypoints = ((keypoints + 0.5) * np.array([224, 224])).astype(np.uint16) # range [0; 224]

        
        # image = image.permute(1, 2, 0).numpy() # height x width x channel

        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints'] 

        keypoints = keypoints / np.array([224, 224]) - 0.5 # range [-0.5; 0.5] 

        return image, keypoints.astype(np.float32)

    def __len__(self):
        return len(self.data)


def saveImage(image, keypoints):
    toPil = T.ToPILImage()

    image = toPil(image)

    # w, h = image.size
    w, h = (224, 224)
    keypoints = ((keypoints + 0.5) * np.array([w, h])).astype(np.uint16) # scale up
    # print(keypoints)

    draw = ImageDraw.Draw(image)

    for x, y in keypoints:
        # print(x, y)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0))

    image.save('foo.jpg')


def main():
    data = DLIB()
    transform = A.Compose([
        A.Rotate(limit=45), # [-45; 45]
        A.RandomBrightnessContrast(),
        A.RGBShift(),
        ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    data = DLIBTransform(data=data, transform=transform)
    image, keypoints = data[30]
    saveImage(image, keypoints)


if __name__ == "__main__":
    main()

    