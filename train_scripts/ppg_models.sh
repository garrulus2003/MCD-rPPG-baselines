#! /bin/bash

python prepare_data.py \
--preparation_class video_segmentation \
--n_samples 3 \
--only_hd \
--dataframe db.csv \
--video_length 300 \
--output_directory video_segmentation \
--output_dataframe video_segmentation_df.csv

python split_data.py --dataframe data/video_segmentation_df.csv

python train.py --model_class ppg \
--dataframe data/video_segmentation_df_train.csv \
--batch_size 4 --num_epochs 50 --lr 9e-3 \
--exp_name ppg_exps/tscan

python evaluate.py --model_class ppg --model_name TSCAN \
--dataframe data/video_segmentation_df_test.csv \
--exp_name ppg_exps/tscan \
--path_to_ckpt ppg_exps/tscan/sate_dict

for model in POS_WANG POS_WANG_mask LGI LGI_mask

python evaluate.py --model_class ppg --model_name $model \
--dataframe data/video_segmentation_df_test.csv \
--exp_name ppg_exps/$model \