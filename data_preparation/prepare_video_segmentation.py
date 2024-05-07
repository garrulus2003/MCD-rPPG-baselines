import os
import numpy as np
import pandas as pd
import skvideo.io
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tqdm.auto import tqdm

def get_bbox(skin):
    rows = skin.sum(axis=1)
    x_min = (rows != 0).argmax()
    x_max = len(rows) - 1 - (np.flip(rows) != 0).argmax()
    
    columns = skin.sum(axis=0)
    y_min = (columns!=0).argmax()
    y_max = len(columns) - 1 - (np.flip(columns!=0)).argmax()
    
    return (x_min, x_max, y_min, y_max)


def segment_photo(photo_np, segmenter):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=photo_np)
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    category_mask = category_mask.numpy_view()
    skin = (category_mask == 3)
    bbox = get_bbox(skin)
    return category_mask, bbox


def get_one_bbox(bboxes):
    x1 = np.min(np.array([b[0] for b in bboxes]))
    x2 = np.max(np.array([b[1] for b in bboxes]))
    y1 = np.min(np.array([b[2] for b in bboxes]))
    y2 = np.max(np.array([b[3] for b in bboxes]))

    x_delta = (x2 - x1)//10
    y_delta = (y2 - y1)//10

    x1 = max(0, x1 - x_delta)
    y1 = max(0, y1 - y_delta)
    x2 += x_delta
    y2 += y_delta
    return x1, x2, y1, y2

def segment_video(video, segmenter):
    masks = np.zeros(video.shape[:3])
    bboxes = []
    for i in range(len(video)):
        mask, bbox = segment_photo(video[i], segmenter)
        masks[i] = mask
        bboxes.append(bbox)
    (x1, x2, y1, y2) = get_one_bbox(bboxes)
    return video[:, x1:x2, y1:y2], masks[:, x1:x2, y1:y2]


def prepare_video(
    path_to_video, 
    output_path, 
    segmenter, 
    targets,
    n_samples,
    path_to_data,
    video_length=300,
):
    np.float = np.float64
    np.int = np.int_
    
    videodata = skvideo.io.vread(path_to_video)
    
    indices = np.random.choice(
        len(videodata) - video_length, 
        n_samples, 
        replace=False
    )
    
    filenames_video = []
    filenames_mask = []

    for i, index in enumerate(indices):
        
        video_cropped, mask = segment_video(
            videodata[index : index+video_length], 
            segmenter
        )

        filename = '_'.join(
            [
                str(targets['patient_id']), 
                targets['camera'], 
                targets['step'],
                str(i).zfill(len(str(n_samples)))
            ]
        )
        
        save_path_video = os.path.join(output_path, 'video', filename)
        np.save(save_path_video, video_cropped)
        filenames_video.append(save_path_video + '.npy')

        save_path_mask = os.path.join(output_path, 'mask', filename)
        np.save(save_path_mask, mask)
        filenames_mask.append(save_path_mask + '.npy')
    
    return pd.DataFrame({
        'begin': indices,
        'end': indices + video_length,
        'video': filenames_video,
        'mask': filenames_mask
    })


def prepare_video_segmentation(
    path_to_dataframe,
    output_path_files,
    output_path_dataframe,
    n_samples=20,
    only_hd=False,
    video_length=300
):
    df = pd.read_csv(path_to_dataframe)

    base_options = python.BaseOptions(
        model_asset_path='checkpoints/selfie_multiclass_256x256.tflite'
    )
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)

    list_of_dataframes = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if only_hd and 'HD' not in row['video']:
            continue
            
        extracted_images_df = prepare_video(
            row['video'],
            output_path_files,
            segmenter, 
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