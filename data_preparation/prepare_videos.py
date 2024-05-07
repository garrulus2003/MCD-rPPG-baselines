import os
import numpy as np
import pandas as pd
import skvideo.io
import mediapipe as mp
import cv2

from tqdm.auto import tqdm


class MediaPipeFaceExtractor:
    def __init__(
        self, 
        max_num_faces=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ):
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces, 
            min_detection_confidence=self.min_detection_confidence, 
            min_tracking_confidence=self.min_tracking_confidence
        )
        
    def process_frame(self, frame):
        face_results = self.face_mesh.process(frame)
        if face_results.multi_face_landmarks is None:
            return None, None
        points, bbox = points_and_bbox(
            face_results, 
            frame.shape[0], 
            frame.shape[1]
        )
        return points, bbox


def pick_bbox(video_bboxes, expand=1.2):
    min_y1 = min([y1 for x1, x2, y1, y2 in video_bboxes])
    max_y2 = max([y2 for x1, x2, y1, y2 in video_bboxes])

    min_x1 = min([x1 for x1, x2, y1, y2 in video_bboxes])
    max_x2 = max([x2 for x1, x2, y1, y2 in video_bboxes])

    W = max_x2 - min_x1
    H = max_y2 - min_y1
    
    crop_size = max(W, H) * expand

    cen_x = min_x1 + W//2
    cen_y = min_y1 + H//2

    new_x1 = int(cen_x - crop_size/2)
    new_x2 = int(cen_x + crop_size/2)

    new_y1 = int(cen_y - crop_size/2)
    new_y2 = int(cen_y + crop_size/2)

    return new_x1, new_x2, new_y1, new_y2
    

def points_and_bbox(face_results, frame_height, frame_width):
    points = list()
    for face_landmarks in face_results.multi_face_landmarks:
        x1 = y1 = 1
        x2 = y2 = 0
        for lm in face_landmarks.landmark:
            cx, cy = lm.x, lm.y
            ccx = int(cx * frame_width)
            ccy = int(cy * frame_height)
            points.append((ccx, ccy))
            if cx < x1:
                x1 = cx
            if cy < y1:
                y1 = cy
            if cx > x2:
                x2 = cx
            if cy > y2:
                y2 = cy
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        x1, x2 = int(x1 * frame_width), int(x2 * frame_width)
        y1, y2 = int(y1 * frame_height), int(y2 * frame_height)
    return points, (x1, x2, y1, y2)


def prepare_video(
    path_to_video, 
    output_path, 
    detector, 
    targets,
    n_samples,
    video_length=300
):
    np.float = np.float64
    np.int = np.int_
    
    videodata = skvideo.io.vread(path_to_video)
    indices = np.random.choice(len(videodata) - video_length, n_samples, replace=False)
    filenames = []

    for i, index in enumerate(indices):
        bboxes = []
        for j in range(index, index + video_length):
            landmarks, bbox = detector.process_frame(videodata[j])
            if landmarks is not None:
                bboxes.append(bbox)
        if len(bboxes)==0:
            extracted_video = videodata[index: index + video_length]
        else:
            bbox = pick_bbox(bboxes)
            extracted_video = videodata[index:(index+video_length), bbox[2]:bbox[3], bbox[0]:bbox[1]]

        filename = '_'.join(
            [
                str(targets['patient_id']), 
                targets['camera'], 
                targets['step'],
                str(i).zfill(len(str(n_samples)))
            ]
        )
        
        save_path = os.path.join(output_path, filename)
        np.save(save_path, extracted_video)
        filenames.append(save_path + '.npy')
    
    return pd.DataFrame({
        'begin': indices,
        'end': indices + video_length,
        'video': filenames,
    })


def prepare_videos(
    path_to_dataframe,
    output_path_files,
    output_path_dataframe,
    n_samples=20,
    only_hd=False,
    video_length=300
):
    df = pd.read_csv(path_to_dataframe)

    extractor = MediaPipeFaceExtractor()

    list_of_dataframes = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if only_hd and 'HD' not in row['video']:
            continue
            
        extracted_images_df = prepare_video(
            row['video'],
            output_path_files,
            extractor, 
            row,
            n_samples,
            video_length
        )

        extracted_images_df['patient_id'] = row['patient_id']
        extracted_images_df['step'] = row['step']
        extracted_images_df['camera'] = row['camera']
        list_of_dataframes.append(extracted_images_df)
        
    new_df = pd.concat(list_of_dataframes).reset_index(drop='True')

    new_df = new_df.merge(
        df.drop(columns=['video']), 
        on=['patient_id', 'camera', 'step']
    ).reset_index(drop=True)

    new_df.to_csv(output_path_dataframe, index=False)
        