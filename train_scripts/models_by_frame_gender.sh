#! /bin/bash

python prepare_data.py \
--preparation_class frame \
--n_samples 20 \
--dataframe db.csv \
--output_directory frames \
--output_dataframe frames_df.csv

python split_data.py --dataframe data/frames_df.csv

for model in resnet50 vit swin rexnet150_pretrained enet2_pretrained
do
python train.py --model_class by_frame --target gender --model_name $model \
--dataframe data/frames_df_train.csv \
--exp_name gender_exp/$model --num_epochs 100

python evaluate.py --model_class by_frame --target gender --model_name $model \
--dataframe data/frames_df_test.csv \
--exp_name gender_exp/$model \
--path_to_ckpt gender_exp/$model/state_dict.pt
done

