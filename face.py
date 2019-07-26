import face_alignment
from skimage import io
import numpy as np
import cv2

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

input = io.imread('1.jpg')
preds = fa.get_landmarks(input)
# preds = np.asarray(preds)
# print(preds.shape)

frame = cv2.imread("1.jpg")
for axis in preds[0]:
    cv2.circle(frame, tuple(axis), 1, (0, 0, 255), -1)
cv2.imshow('WebCam', frame)
cv2.waitKey(10000)