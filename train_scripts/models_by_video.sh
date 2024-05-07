#! /bin/bash

python prepare_data.py --preparation_class video \
--n_samples 3 \
--only_hd \
--dataframe db.csv \
--video_length 300 \
--output_directory video_fragments \
--output_dataframe video_fragments_df.csv


python split_data.py --dataframe data/video_fragments_df.csv


python train.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_train.csv \
--num_epochs 100 --lr 1e-4 --unfreeze \
--exp_name video_exps/full_size


python train.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_train.csv \
--num_epochs 100 --lr 1e-4 --unfreeze \
--frequency 8 --crop 16 \
--exp_name video_exps/each_eight


python train.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_train.csv \
--num_epochs 100 --lr 1e-4 --unfreeze \
--frequency 8 --crop 16 --avg_over_frames \
--exp_name video_exps/each_eight_avg


python evaluate.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_test.csv \
--path_to_ckpt video_exps/full_size/state_dict.pt \ 
--exp_name video_exps/full_size


python evaluate.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_test.csv \
--path_to_ckpt video_exps/each_eight/state_dict.pt \
--frequency 8 --crop 16 \
--exp_name video_exps/each_eight


python evaluate.py --model_class by_video --model_name r2plus1d --target all \
--dataframe data/video_fragments_df_test.csv \
--path_to_ckpt video_exps/each_eight_avg/state_dict.pt \
--frequency 8 --crop 16 --avg_over_frames \
--exp_name video_exps/each_eight_avg