"""
rPPG-Toolbox: Deep Remote PPG Toolbox
arXiv preprint arXiv:2210.00716
2022
"""

import torch
import os
from scipy import signal
from scipy import sparse
import math
import cv2
import numpy as np
import pandas as pd


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the
    time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(
        diffnormalized_data, 
        diffnormalized_data_padding, axis=0
    )
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the 
    time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def standardized_label(label):
    """Z-score standardization for label signal."""
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label

def face_detection(frame, backend='HC', use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        backend(str): backend to utilize for face detection.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    if backend == "HC":
        # Use OpenCV's Haar Cascade algorithm implementation for face detection
        # This should only utilize the CPU
        detector = cv2.CascadeClassifier(
        'checkpoints/haarcascade_frontalface_default.xml')

        # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
        # (x,y) corresponds to the top-left corner of the zone to define using
        # the computed width and height.
        face_zone = detector.detectMultiScale(frame)

        if len(face_zone) < 1:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) >= 2:
            # Find the index of the largest face zone
            # The face zones are boxes, so the width and height are the same
            max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
            face_box_coor = face_zone[max_width_index]
            print("Warning: More than one faces are detected. Only cropping the biggest one.")
        else:
            face_box_coor = face_zone[0]
    elif backend == "RF":
        # Use a TensorFlow-based RetinaFace implementation for face detection
        # This utilizes both the CPU and GPU
        res = RetinaFace.detect_faces(frame)

        if len(res) > 0:
            # Pick the highest score
            highest_score_face = max(res.values(), key=lambda x: x['score'])
            face_zone = highest_score_face['facial_area']

            # This implementation of RetinaFace returns a face_zone in the
            # form [x_min, y_min, x_max, y_max] that corresponds to the 
            # corners of a face zone
            x_min, y_min, x_max, y_max = face_zone

            # Convert to this toolbox's expected format
            # Expected format: [x_coord, y_coord, width, height]
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min

            # Find the center of the face zone
            center_x = x + width // 2
            center_y = y + height // 2
                
            # Determine the size of the square (use the maximum of width and height)
            square_size = max(width, height)
                
            # Calculate the new coordinates for a square face zone
            new_x = center_x - (square_size // 2)
            new_y = center_y - (square_size // 2)
            face_box_coor = [new_x, new_y, square_size, square_size]
        else:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    else:
        raise ValueError("Unsupported face detection backend!")

    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor

def crop_face_resize(frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection, 
                    detection_freq, use_median_box, width, height):
    """Crop face and resize frames.

    Args:
        frames(np.array): Video frames.
        use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                     and resizing.
                                     If True, it performs face detection every "detection_freq" frames.
        detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
        width(int): Target width for resizing.
        height(int): Target height for resizing.
        use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
        use_face_detection(bool):  Whether crop the face.
        larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
    Returns:
        resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        if use_face_detection:
            face_region_all.append(face_detection(frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))
        else:
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region_all, dtype='int')
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')

    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
            reference_index = i // detection_freq
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if use_face_detection:
            if use_median_box:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames

def preprocess(frames, bvps, config):
    """Preprocesses a pair of data.

    Args:
        frames(np.array): Frames in a video.
        bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
        config_preprocess(CfgNode): preprocessing settings(ref:config.py).
    Returns:
        frame_clips(np.array): processed video data by frames
        bvps_clips(np.array): processed bvp (ppg) labels by frames
    """
    # resize frames and crop for face region
    frames = crop_face_resize(
        frames,
        config['DO_CROP_FACE'],
        config['BACKEND'],
        config['USE_LARGE_FACE_BOX'],
        config['LARGE_BOX_COEF'],
        config['DO_DYNAMIC_DETECTION'],
        config['DYNAMIC_DETECTION_FREQUENCY'],
        config['USE_MEDIAN_FACE_BOX'],
        config['W'],
        config['H'])
    # Check data transformation type
    data = list()  # Video data
    for data_type in config['DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw":
            data.append(f_c)
        elif data_type == "DiffNormalized":
            data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized":
            data.append(standardized_data(f_c))
        else:
            raise ValueError("Unsupported data type!")
    data = np.concatenate(data, axis=-1)  # concatenate all channels
    if config['LABEL_TYPE'] == "Raw":
        pass
    elif config['LABEL_TYPE'] == "DiffNormalized":
        bvps = diff_normalize_label(bvps)
    elif config['LABEL_TYPE'] == "Standardized":
        bvps = standardized_label(bvps)
    else:
        raise ValueError("Unsupported label type!")

    frames_clips = np.array([data])
    bvps_clips = np.array([bvps])

    return frames_clips, bvps_clips