import argparse
import os
import skvideo.io

from data_preparation.prepare_frames import prepare_frames
from data_preparation.prepare_videos import prepare_videos
from data_preparation.prepare_video_segmentation import prepare_video_segmentation


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--preparation_class',
        type=str,
        required=True,
        choices=['frame', 'video', 'video_segmentation'],
        help="Choose according to the  model class you intend to use"
    )

    parser.add_argument(
        '--only_hd', 
        action='store_true',
        help="To use only frontal Full HD videos"
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help="Number of samples from each video"
    )
    
    parser.add_argument(
        '--dataframe',
        type=str,
        required=True,
        help="Name of gt file (.csv)"
    )
    
    parser.add_argument(
        '--output_directory',
        type=str,
        required=True,
        help="Name of directory to store files"
    )

    parser.add_argument(
        '--output_dataframe',
        type=str,
        required=True,
        help="Name of the new dataframe"
    )

    parser.add_argument(
        '--path_to_data',
        type=str,
        default='data',
        help="Path to all the data storage"
    )
    
    parser.add_argument(
        '--video_length',
        type=int,
        default=300,
        help="Length of video samples (in frames)"
    )
    
    return parser.parse_args(args)


def run(args=None):
    args = parse_args(args)

    args.dataframe = os.path.join(args.path_to_data, args.dataframe)
    args.output_directory = os.path.join(args.path_to_data, args.output_directory)
    args.output_dataframe = os.path.join(args.path_to_data, args.output_dataframe)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.preparation_class == 'frame':
        prepare_frames(
            args.dataframe,
            args.output_directory,
            args.output_dataframe,
            args.n_samples,
            args.only_hd
        )
        return
        
    if args.preparation_class == 'video':
        prepare_videos(
            args.dataframe,
            args.output_directory,
            args.output_dataframe,
            args.n_samples,
            args.only_hd,
            args.video_length
        )
        return

    if args.preparation_class == 'video_segmentation':
        video_path = os.path.join(args.output_directory, 'video')
        mask_path = os.path.join(args.output_directory, 'mask')
        
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
            
        prepare_video_segmentation(
            args.dataframe,
            args.output_directory,
            args.output_dataframe,
            args.n_samples,
            args.only_hd,
            args.video_length
        )
        return
    
    
if __name__ == '__main__':
    run()