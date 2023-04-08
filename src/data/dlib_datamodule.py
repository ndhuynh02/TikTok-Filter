from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A

import requests
from pathlib import Path
import tarfile
from tqdm import tqdm
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.dlib import DLIB, DLIBTransform


class DLIBDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[A.Compose] = None,
        transform_val: Optional[A.Compose] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

        folder = f'{self.hparams.data_dir}DLIB/'
        file_name = 'dlib.tar.gz'
        file_path = folder + file_name

        if (os.path.exists(file_path)):
            print("Data is downloaded")
            return

        # creating a new directory 
        Path(folder).mkdir(parents=True, exist_ok=True)

        url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
        # Streaming, so we can iterate over the response.
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        print("Extracting")
        file = tarfile.open(file_path)
        # extracting file to current directory
        file.extractall('.')
        file.close()

    def setup(self, stage: Optional[str] = None):
        import numpy as np
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DLIB()
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths= self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = DLIBTransform(self.data_train, self.hparams.transform_train)
            self.data_val = DLIBTransform(self.data_val, self.hparams.transform_val)
            self.data_test = DLIBTransform(self.data_test, self.hparams.transform_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    # def draw_batch(self):
    #     import math
    #     import matplotlib.pyplot as plt

    #     images, landmarks = next(iter(self.train_dataloader()))
    #     batch_size = len(images)
    #     grid_size = math.sqrt(batch_size)
    #     grid_size = math.ceil(grid_size)
    #     fig = plt.figure(figsize=(10, 10))
    #     fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
    #     for i in range(batch_size):
    #         ax = fig.add_subplot(grid_size, grid_size, i+1, xticks=[], yticks=[])
    #         image, landmark = images[i], landmarks[i]
    #         image = image.squeeze().permute(1,2,0)
    #         plt.imshow(image)
    #         kpt = []
    #         for j in range(68):
    #             kpt.append(plt.plot(landmark[j][0], landmark[j][1], 'g.'))
    #     plt.tight_layout()
    #     plt.savefig('batch.png')

def draw_batch(images, keypoints):
    # images: torch.float32 (batch x channel x height x width)
    _, _, h, w = images.shape
    
    import matplotlib.pyplot as plt

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
        ten = x.clone().permute(1, 2, 3, 0) # channel x height x width x batch
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2) # batch x channel x height x width

    fig = plt.figure(figsize=(8,8))

    images = denormalize(images)
    for i in range(len(images)):
        ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        img = images[i]
        assert len(keypoints[i]) == 68
        for j in range(68):
            plt.scatter((keypoints[i][j][0] + 0.5) * w, (keypoints[i][j][1] + 0.5) * h, s=10, marker='.', c='r')
        plt.imshow(img.permute(1, 2, 0))
    plt.savefig('batch.png')


@hydra.main(config_path='../../configs/data', config_name='dlib', version_base=None)
def main(cfg: DictConfig):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torchvision

    transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=45), # [-45; 45]
        A.RandomBrightnessContrast(),
        A.RGBShift(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    
    dlib = hydra.utils.instantiate(cfg, train_transform=transform)
    dlib.setup()

    batch = next(iter(dlib.train_dataloader()))
    image, keypoints = batch
    # print("Batch shape:", len(batch))
    # print('Image shape:', image.shape) # batch * 3 * 244 * 244
    # print('Keypoints shape:', keypoints.shape) # batch * 68 * 2
    annotated_batch = DLIBTransform.annotate_tensor(image, keypoints)
    torchvision.utils.save_image(annotated_batch, "batch.png")


if __name__ == "__main__":
    main()