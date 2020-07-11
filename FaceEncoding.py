#loading and updating pickle with backup

import dlib
import scipy.misc
import numpy as np
import os
from os.path import exists
import pickle
import cv2

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

TOLERANCE = 0.6

#This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    image = cv2.imread(path_to_image)
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def encodings(vid):
    detected_faces=face_detector(vid,1)
    shapes_faces = [shape_predictor(vid, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(vid, face_pose, 1)) for face_pose in shapes_faces]



# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)


def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return 'Not Found'

