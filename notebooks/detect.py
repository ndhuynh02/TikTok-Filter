from typing import Optional
import hydra
from omegaconf import DictConfig
import pyrootutils

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image, ImageDraw
import numpy as np

import cv2
from deepface.detectors import FaceDetector

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.dlib_module import DLIBLitModule
from src.data.components.dlib import DLIB
from src.models.components.simple_cnn import SimpleCNN


def detect(cfg: DictConfig, option: Optional[str] = None):
    net = hydra.utils.instantiate(cfg.net)

    model = DLIBLitModule.load_from_checkpoint('logs/train/runs/2023-04-08_15-12-10/checkpoints/epoch_000.ckpt', 
                                               net=net)
    
    transform = A.Compose([A.Resize(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])
    
    videoCapture = cv2.VideoCapture(option)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outputVideo = cv2.VideoWriter('outputVideo.mp4', fourcc, videoCapture.get(cv2.CAP_PROP_FPS),
                                  (width, height))
    

    # choose one of these:
    #         "opencv"
    #         "ssd"
    #         "dlib"
    #         "mtcnn"
    #         "retinaface"
    #         "mediapipe"
    # retinaface and mtcnn give best score but slow
    detector_name = "ssd"

    while (True):
        ret, frame = videoCapture.read()
  
        if ret == True: 
            detector = FaceDetector.build_model(detector_name)
            detector = FaceDetector.detect_faces(detector, detector_name, frame)

            for face in detector:
                # face: list = [cropped image (h x w x c, np.array), bounding-box, confidence]
                bbox = face[1] 
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))

                image = face[0]
                transformed = transform(image=image)
                transformed_image = torch.unsqueeze(transformed['image'], dim=0)

                keypoints = model(transformed_image).detach().numpy()[0]

                h, w, _ = image.shape
                keypoints = ((keypoints + 0.5) * np.array([w, h]) + np.array(bbox[:2])).astype(np.uint16)

                for point in keypoints:
                    frame = cv2.circle(frame, point, 3, (0, 255, 0), thickness=-1)
            outputVideo.write(frame)
    
            # Press S on keyboard to stop the process
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
    
        # Break the loop
        else:
            break
    
    # image =  Image.open(option).convert('RGB')

    # transformed_image = np.array(image)
    # transformed = transform(image=transformed_image)
    # transformed_image = transformed['image']

    # transformed_image = torch.unsqueeze(transformed_image, dim=0)

    # keypoints = model(transformed_image)

    # h, w = image.size
    # keypoints = keypoints.detach().numpy()[0]
    # keypoints = (keypoints + 0.5) * np.array([w, h]) # convert to image pixel coordinates

    # output = DLIB.annotate_image(image, keypoints)
    # output.save('output.png')


if __name__ == "__main__":
    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        detect(cfg=cfg, option='testVideo.mp4')

    main()