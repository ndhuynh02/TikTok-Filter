from typing import Any, Dict, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

import requests
from pathlib import Path
import tarfile
from tqdm import tqdm
import os

from dataset.dlip import DLIP, DLIPTransform

import hydra
from omegaconf import DictConfig, OmegaConf

class DLIPDataModule(LightningDataModule):
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

        folder = f'{self.hparams.data_dir}DLIP/'
        file_name = 'dlip.tar.gz'
        file_path = folder + file_name

        if (os.path.exists(file_path)):
            print("File is downloaded")
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

    def setup(self):
        import numpy as np
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DLIP()
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths= self.hparams.train_val_test_split,
                # generator=torch.Generator().manual_seed(42),
            )

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

def draw_batch(images, keypoints):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,8))
    #return print(key_points.shape)

    for i in range(len(images)):
        ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        image = images[i]
        for j in range(68):
            plt.scatter((keypoints[i][j][0] + 0.5) *224, (keypoints[i][j][1]+0.5)*224, s=10, marker='.', c='r')
        # plt.scatter(key_points[:,:1,i],key_points[:,:2,i],s=10, marker='.', c='r')
        plt.imshow(image.permute(1, 2, 0))
    plt.savefig('batch.png')


@hydra.main(config_path='../../configs/data', config_name='dlip', version_base=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    dlip = hydra.utils.instantiate(cfg)
    dlip.setup()

    transform = A.Compose([
        A.Rotate(limit=45), # [-45; 45]
        A.RandomBrightnessContrast(),
        A.RGBShift(),
        ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    dlip.data_train = DLIPTransform(dlip.data_train, transform)

    batch = next(iter(dlip.train_dataloader()))
    image, keypoints = batch
    # print("Batch shape:", len(batch))
    # print('Image shape:', image.shape) # batch * 3 * 244 * 244
    # print('Keypoints shape:', keypoints.shape) # batch * 68 * 2
    draw_batch(image, keypoints)


if __name__ == "__main__":
    main()