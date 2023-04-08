from torch.utils.data import Dataset
import torchvision
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

        keypoints = []
        for points in image.getElementsByTagName('part'):
            keypoints.append((int(points.getAttribute('x')), int(points.getAttribute('y'))))

        bbox = image.getElementsByTagName('box')[0]
        x_min = int(bbox.getAttribute('left'))
        y_min = int(bbox.getAttribute('top'))
        x_max = x_min + int(bbox.getAttribute('width'))
        y_max = y_min + int(bbox.getAttribute('height'))

        image: Image = Image.open(self.root + image_path).convert('RGB')
        image: Image = image.crop((x_min, y_min, x_max, y_max))
        
        keypoints -= np.array([x_min, y_min])  # crop

        return image, keypoints
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def annotate_image(image: Image, landmarks: np.ndarray) -> Image:
        draw = ImageDraw.Draw(image)
        for i in range(landmarks.shape[0]):
            draw.ellipse((landmarks[i, 0] - 2, landmarks[i, 1] - 2,
                          landmarks[i, 0] + 2, landmarks[i, 1] + 2), fill=(0, 255, 0))
        return image


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
        image = np.array(image)
        h, w, _ = image.shape

        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints'] 

        _, h, w = image.shape
        keypoints = keypoints / np.array([w, h]) - 0.5 # range [-0.5; 0.5] 

        return image, keypoints.astype(np.float32)

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def annotate_tensor(image: torch.Tensor, landmarks: np.ndarray) -> Image:

        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

        images = denormalize(image)
        images_to_save = []
        for lm, img in zip(landmarks, images):
            img = img.permute(1, 2, 0).numpy() * 255
            h, w, _ = img.shape
            lm = (lm + 0.5) * np.array([w, h]) # convert to image pixel coordinates
            img = DLIB.annotate_image(Image.fromarray(img.astype(np.uint8)), lm)
            images_to_save.append(torchvision.transforms.ToTensor()(img) )

        return torch.stack(images_to_save)
    

def main():
    data = DLIB()
    image, keypoints = data[0]
    annotated_image = DLIB.annotate_image(image, keypoints)
    annotated_image.save("foo.jpg")


if __name__ == "__main__":
    main()