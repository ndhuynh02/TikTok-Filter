from torch.utils.data import Dataset
from torchvision.io import read_image
import albumentations as A

from xml.dom import minidom
import os 

import numpy as np

class DLIP(Dataset):
    data_dir = 'data/DLIP'
    data_url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
    def __init__(self, root: str = 'data/DLIP/ibug_300W_large_face_landmark_dataset/', augmentation: bool = True) -> None:
        super().__init__()

        self.root = root
        self.augmentation = augmentation

        self.data = minidom.parse(self.root + 'labels_ibug_300W.xml')
        self.data = self.data.getElementsByTagName('image')

        self.transform = None
        if augmentation:
            self.transform = A.Compose([
                    A.Resize(height=224, width=224)
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.ToTensorV2(),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    def __getitem__(self, idx: int):
        image = self.data[idx]
        image_path = image.getAttribute('file')
        image_width = image.getAttribute('width')
        image_height = image.getAttribute('height')

        keypoints = []
        for points in image.getElementsByTagName('part'):
            keypoints.append((int(points.getAttribute('x')), int(points.getAttribute('y'))))

        image = read_image(self.root + image_path)

        if self.transform:
            transformed = transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints'] / np.array([image_width, image_height]) - 0.5 # range [-0.5; 0.5]

        return image, keypoints.astype(np.float)
    
    def __len__(self):
        return len(self.data)
        
    # def prepare_data(self) -> None:
    #     file_name = 'dlip.tar.gz'
    #     file_path = self.data_dir + '/' + file_name
    #     if (os.path.exists(file_path)):
    #         print("File is downloaded")
    #         return

    #     # Streaming, so we can iterate over the response.
    #     print("Downloading")
    #     self.download(self.data_url, file_path)

    #     print("Extracting")
    #     file = tarfile.open(file_path)
    #     # extracting file to current directory
    #     file.extractall('.')
    #     file.close()


    # def download(self, url, filename):
    #     import functools
    #     import pathlib
    #     import shutil
    #     import requests
    #     from tqdm.auto import tqdm
        
    #     r = requests.get(self.data_url, stream=True, allow_redirects=True)
    #     if r.status_code != 200:
    #         r.raise_for_status()  # Will only raise for 4xx codes, so...
    #         raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    #     file_size = int(r.headers.get('Content-Length', 0))

    #     path = pathlib.Path(filename).expanduser().resolve()
    #     path.parent.mkdir(parents=True, exist_ok=True)

    #     desc = "(Unknown total file size)" if file_size == 0 else ""
    #     r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    #     with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
    #         with path.open("wb") as f:
    #             shutil.copyfileobj(r_raw, f)

if __name__ == "__main__":

    data = DLIP()
    image, keypoints = data[0]

    plt.figure(1)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()