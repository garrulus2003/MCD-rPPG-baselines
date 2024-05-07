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
python train.py --model_class by_frame --target all --model_name $model \
--dataframe data/frames_df_train.csv \
--exp_name regression_exp/$model/all \
--num_epochs 100 --lr 1e-5 --unfreeze

python evaluate.py --model_class by_frame --target all --model_name $model \
--dataframe data/frames_df_test.csv \
--exp_name regression_exp/$model/all \
--path_to_ckpt regression_exp/$model/all/state_dict.pt

declare -A normalizations=( ["age"]=100 ["bmi"]=40 ["lower_ap"]=100 ["upper_ap"]=100 ["saturation"]=100 \
["temperature"]=40 ["stress"]=10 ["hemoglobin"]=20 ["glycated_hemoglobin"]=20 ["cholesterol"]=10 ["respiratory"]=100 \
["rigidity"]=40 ["pulse"]=100)

for target in "${!normalizations[@]}"
do 
python train.py --model_class by_frame --target $target --model_name $model \
--dataframe data/frames_df_train.csv \
--exp_name regression_exp/$model/$target \
--num_epochs 100 --lr 1e-5 --unfreeze

python evaluate.py --model_class by_frame --target $target --model_name $model \
--dataframe data/frames_df_test.csv \
--exp_name regression_exp/$model/$target \
--path_to_ckpt regression_exp/$model/$target/state_dict.pt

done

done
