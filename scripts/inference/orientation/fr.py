import cv2
import face_recognition
import os
from PIL import Image


def load_image_cv(file_path):
    img = cv2.imread(file_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


face1 = load_image_cv("/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/orientation/Screenshot 2024-03-01 at 12.52.38.png")
face2 = load_image_cv("/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/orientation/Screenshot 2024-03-01 at 18.53.29.png")


img_encoding = face_recognition.face_encodings(face1)[0]
img_encoding2 = face_recognition.face_encodings(face2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)
print(face_recognition.face_distance([img_encoding], img_encoding2))