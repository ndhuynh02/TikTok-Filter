from xml.dom import minidom
from PIL import Image

from data.dlip_datamodule import DLIPDataModule

# # parse an xml file by name
# file = minidom.parse('data/DLIP/ibug_300W_large_face_landmark_dataset/labels_ibug_300W.xml')

# #use getElementsByTagName() to get tag
# image0 = file.getElementsByTagName('image')[0]

# print(image0.getAttribute('file'))

# parts = image0.getElementsByTagName('part')
# for part in parts:
#     print((int(part.getAttribute('x')), int(part.getAttribute('y'))))


# im = Image.open('data/DLIP/ibug_300W_large_face_landmark_dataset/helen/trainset/146827737_1.jpg')
# print(im.size)

dm = DLIPDataModule()

dm.setup()
assert dm.data_train and dm.data_val and dm.data_test
assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

# num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
# print("Len(data) = ",num_datapoints)

# visualize an image
batch = next(iter(dm.train_dataloader()))
image, keypoints = batch

print(image.size)