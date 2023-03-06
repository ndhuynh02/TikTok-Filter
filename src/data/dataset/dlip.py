from torch.utils.data import Dataset
from torchvision.io import read_image

import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image, ImageDraw

from xml.dom import minidom
import os 

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class DLIP(Dataset):
    data_dir = 'data/DLIP'
    data_url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"

    def __init__(self, root: str = 'data/DLIP/ibug_300W_large_face_landmark_dataset/') -> None:
        super().__init__()

        self.root = root

        self.data = minidom.parse(self.root + 'labels_ibug_300W.xml')
        self.data = self.data.getElementsByTagName('image')

    def __getitem__(self, idx: int):
        image = self.data[idx]
        image_path = image.getAttribute('file')
        # image_width = image.getAttribute('width')
        # image_height = image.getAttribute('height')

        keypoints = []
        for points in image.getElementsByTagName('part'):
            keypoints.append((int(points.getAttribute('x')), int(points.getAttribute('y'))))

        bbox = image.getElementsByTagName('box')[0]
        x_min = int(bbox.getAttribute('left'))
        y_min = int(bbox.getAttribute('top'))
        bbox_width = int(bbox.getAttribute('width'))
        bbox_height = int(bbox.getAttribute('height'))

        image = read_image(self.root + image_path)
        image = image.permute(1, 2, 0).numpy()
        
        transform = A.Compose([A.Crop(x_min=x_min, y_min=y_min, 
                                x_max=x_min + bbox_width, y_max=y_min + bbox_height),
                                ToTensorV2()],
                                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        keypoints = keypoints / np.array([bbox_width, bbox_height]) - 0.5 # range [-0.5; 0.5]        

        return image, keypoints.astype(np.float64)
    
    def __len__(self):
        return len(self.data)


class DLIPTransform(Dataset):
    def __init__(self, data: DLIP, transform: Optional[A.Compose]):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image, keypoints = self.data[idx]

        if transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints'] 

        return image, keypoints.astype(np.float64)

    def __len__(self):
        return len(self.data)


def saveImage(image, keypoints):
    toPil = T.ToPILImage()

    image = toPil(image)

    w, h = image.size
    keypoints = ((keypoints + 0.5) * np.array([w, h])).astype(np.uint16)

    draw = ImageDraw.Draw(image)

    for x, y in keypoints:
        # print(x, y)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 0, 0))

    image.save('foo.jpg')


if __name__ == "__main__":
    data = DLIP()
    image, keypoints = data[1]
    saveImage(image, keypoints)

    