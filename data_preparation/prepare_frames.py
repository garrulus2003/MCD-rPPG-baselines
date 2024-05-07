import os
import numpy as np
import pandas as pd
import skvideo.io
import mediapipe as mp
import cv2

from tqdm.auto import tqdm


def prepare_frames_video(
    path_to_video, 
    output_path, 
    detector, 
    targets,
    n_samples
):
    
    np.float = np.float64
    np.int = np.int_
    
    videodata = skvideo.io.vread(path_to_video)
    indices = np.random.choice(len(videodata), n_samples, replace=False)
    filenames = []

    for i, index in enumerate(indices):
        frame = videodata[index]
        detections = detector.process(cv2.cvtColor(
            frame, 
            cv2.COLOR_BGR2RGB
        )).detections
        
        if detections is None:
            extracted_img = frame
            
        else:
            bbox = detections[0].location_data.relative_bounding_box

            xmin = int(bbox.xmin * frame.shape[1])
            ymin = int(bbox.ymin * frame.shape[0])
            xmax = xmin + int(bbox.width * frame.shape[1])
            ymax = ymin + int(bbox.height * frame.shape[0])
            
            extracted_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

        filename = '_'.join(
            [
                str(targets['patient_id']), 
                targets['camera'], 
                targets['step'],
                str(i).zfill(len(str(n_samples)))
            ]
        )
        
        save_path = os.path.join(output_path, filename)
        np.save(save_path, extracted_img)
        filenames.append(save_path + '.npy')
    
    return pd.DataFrame({
        'begin': indices,
        'end': indices + 1,
        'video': filenames,
    })


def prepare_frames(
    path_to_dataframe,
    output_path_files,
    output_path_dataframe,
    n_samples=20,
    only_hd=False
):
    df = pd.read_csv(path_to_dataframe)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.7
    )

    list_of_dataframes = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if only_hd and 'HD' not in row['video']:
            continue
            
        extracted_images_df = prepare_frames_video(
            row['video'], 
            output_path_files,
            face_detection, 
            row,
            n_samples
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
        