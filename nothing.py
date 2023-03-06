from xml.dom import minidom

data = minidom.parse('data/DLIP/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml')
data = data.getElementsByTagName('image')[0]

bbox = data.getElementsByTagName('box')[1]

print(bbox.getAttribute('top'))