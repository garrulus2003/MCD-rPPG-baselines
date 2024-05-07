# :wave: Introduction

This repository contains baselines and data preparation code for Multi-Camera Dataset rPPG (будет ссылкой). 
It is designed to facilitate the beginning of work with MCD-rPPG.

There are three types of models: 
* Models taking separate frames as input and predicting various medical parameters
* Models taking videos as input and predicting various medical parameters
* Supervised and unsuppervised algorithms for rPPG

To learn more about the **dataset** see (ссылка на статью про датасет).

To learn more about the **baselines** see (ссылка на статью на Хабр) (in russian)

# :city_sunset: Dataset

The dataset contains data about 600 patients. Each patient was filmed twice (before and after physical activity). Every time a video was recorded from three different camera angles simultaneously. Each video is assosiated with a synchronised PPG-signal, an ECG and various medical parameters. After downloading the dataset make sure to organize as shown below. Also make sure that paths in the `db.csv` file point to true locationы of the dataset files.

    -----------------
         data/
         |   |-- db.csv
         |   |-- video/
         |       |-- 1020_FullHDwebcam_after.avi	
         |       |-- 1020_USBVideo_after.avi
         |       |-- 1020_IriunWebcam_after.avi	
         |       |-- 1020_FullHDwebcam_before.avi	
         |       |-- 1020_USBVideo_before.avi	
         |       |-- 1020_IriunWebcam_before.avi
         |       |-- ...
         |   |-- ppg/
         |       |-- 1020_FullHDwebcam_after.PW	
         |       |-- 1020_USBVideo_after.PW
         |       |-- 1020_IriunWebcam_after.PW	
         |       |-- 1020_FullHDwebcam_before.PW	
         |       |-- 1020_USBVideo_before.PW	
         |       |-- 1020_IriunWebcam_before.PW
         |       |-- ...
         |   |-- ppg_sync/
         |       |-- 1020_FullHDwebcam_after.txt	
         |       |-- 1020_USBVideo_after.txt
         |       |-- 1020_IriunWebcam_after.txt	
         |       |-- 1020_FullHDwebcam_before.txt	
         |       |-- 1020_USBVideo_before.txt	
         |       |-- 1020_IriunWebcam_before.txt
         |       |-- ...
         |   |-- ecg/
         |       |-- 1020_FullHDwebcam_after.json	
         |       |-- 1020_USBVideo_after.json
         |       |-- 1020_IriunWebcam_after.json	
         |       |-- 1020_FullHDwebcam_before.json	
         |       |-- 1020_USBVideo_before.json	
         |       |-- 1020_IriunWebcam_before.json
         |       |-- ...
         |   |-- meta/
         |       |-- 1020_FullHDwebcam_after.avi	
         |       |-- 1020_USBVideo_after.avi
         |       |-- 1020_IriunWebcam_after.avi	
         |       |-- 1020_FullHDwebcam_before.avi	
         |       |-- 1020_USBVideo_before.avi	
         |       |-- 1020_IriunWebcam_before.avi
         |       |-- ...
    -----------------
   

# :wrench: Setup

To run the baselines follow the steps:

STEP 1: Clone this repository

STEP 2: Install [Mediapipe](https://developers.google.com/mediapipe/solutions/setup_python) and [torch](https://pytorch.org/get-started/locally/) version compatible with your cuda version and packet manager. 

STEP 3: Insall other packages by running `pip install -r requirements.txt`

# :pizza: Data preparation
## Frames preparation
Separate frames are needed to run models that take frames as input. The code below samples 20 frames from each video of provided dataframe. It also detects the face in each frame and cuts every frame in with resepct to the faces bounding box. 

```
python prepare_data.py \
--preparation_class frame \
--n_samples 20 \
--dataframe db.csv \
--output_directory frames \
--output_dataframe frames_df.csv
```

## Videos preparation
Since provided videos are very long it is better to create their shorter vresions for training and evaluation. Code below samples 3 10-second fragments from all frontal videos in the given dataframe, detects faces and cuts videos with respect to detection bounding boxes.

```
python prepare_data.py \
--preparation_class video \
--n_samples 3 \
--only_hd \
--dataframe db.csv \
--video_length 300 \
--output_directory video_fragments \
--output_dataframe video_fragments_df.csv
```

## Segmentation and skin-mask extraction
Code below also provides 7-class segmentation by Mediapipe Selfie Segmentor and stores videos as well as segmentation masks.

```
python prepare_data.py \
--preparation_class video_segmentation \
--n_samples 3 \
--only_hd \
--dataframe db.csv \
--video_length 300 \
--output_directory video_segmentation \
--output_dataframe video_segmentation_df.csv
```

## Train-Test split
To split the data into to samples one should run with hyperparameters of their choice. One can also specify names of ouput folders using arguments `--output_dataframe_train --output_dataframe_test`

```
 python split_data.py --dataframe data/db.csv
```
# :camera: Models with frame input
There are 5 available models.
* ResNet50 initialized with weights pretrained on ImageNet
* ViT initialized with weights pretrained on ImageNet
* Swin initialized with weights pretrained on ImageNet
* EficientNet_b2 initialized with weights from [face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/pretrained_faces) To use it, download sile manually and put it in `checkpoints` folder under name `pretrained_enet2`.
* ReXNet150 initialized with weights from [face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/pretrained_faces) To use it, download sile manually and put it in `checkpoints` folder under name `pretrained_rexnet`.

See below for custom training and evaluation of models with frame input. Trainig options such as learning rate, batch size, number of epochs etc could be varied.

We report results achivied by ViT and RexNet150 trained in two settings: having exactly one argument as target or having all regression targets at the same time. To reproduce those results run
```
bash train_scripts/models_by_frame_regression.sh
```

MAE on the test subset is presened below.

| target | ReXNet | ViT | ReXNet all argets | ViT all argets |
|----------|----------|----------|----------|----------|
| age | 4.386 | 1.689 | 1.669 | 1.921 | 4.461 |
| bmi | 3.277 | 2.516 | 2.534 | 2.454 | 2.803 |
|lower_ap| 7.423 | 8.581 | 8.146 | 8.013 | 7.870 |
|upper_ap| 13.606 | 14.049 | 13.788 | 13.178 | 13.299 |
|saturation| 0.872 | 1.814 | 0.946 | 0.855 | 0.876 |
|temperature| 0.094 | 0.614 | 0.128 | 0.110 | 0.103 |
|sress| 1.144 | 1.198 | 1.210 | 1.117 | 1.156 |
|hemoglobin| 1.342 | 1.223 | 1.154 | 1.110 | 1.385 |
|glyc. hemogloin| 0.418 | 0.533 | 0.526 | 0.513 | 0.487 |
|cholesterol| 0.647 | 0.692 | 0.710 | 0.629 | 0.680 |
|respiratory| 1.399 | 1.453 | 1.488 | 1.483 | 1.455 |
|rigidity| 2.21| 2.308 | 2.320 |2.284 | 2.201 |
|pulse| 15.043 | 16.564 | 16.121 | 14.685 | 15.316 |




We also report results achieved by all five models on gender prediction task. To reproduce run

```
bash train_scripts/models_by_frame_gender.sh
```
|model|	accuracy |F1 |
|-----|----------|---|
|resnet50|0.952|0.961|
|swin | 0.957 | 0.964|
|rexnet150_pretrained | 0.989 | 0.991 |
|vit | 0.951 | 0.96 |
|enet2_pretrained | 0.979 | 0.983 |

# :video_camera: Models with video input
Pipeline hear is similar to the one by frames.
Four models are available
* Video ResNet_18 3D (3-dimenaional convolutions)
* Video ResNet_18 MC3 (mixed convolutions)
* Video ResNet_18 R2Plus1D (2D spatial and 1D temporal convolutions)
* MViT

See below for custom training and evaluation of models with frame input. Trainig options such as learning rate, batch size, number of epochs etc could be varied. Also there are options to crop a video, skip some frames or group frames and then average.

We report results achivied by Video ResNet R2Plus1D_18 taking 10-second videos as input trained to predict all regression parameters simultaneously. This model os train using only frontal camera views.
To reproduce run 

```
bash train_scripts/models_by_video.sh
```

| target | R2Plus1D_18 | R2Plus1D_18 every 8-th frame | R2Plus1D_18 every 8-th + avg|
|----------|----------| --- | --- |
| age | 1.706 | 1.866 | 1.853 |
| bmi | 2.336 | 2.365 | 2.405 |
|lower_ap| 8.279 | 7.728 | 7.960 |
|upper_ap| 12.540 | 13.050 | 12.727 |
|saturation| 1.274 | 1.057 | 0.968 |
|temperature| 0.258 | 0.199 | 0.190 |
|sress| 1.124 | 1.137 | 1.219 | 1.219 |
|hemoglobin| 1.158 | 1.199 |  1.139 |
|glyc. hemogloin| 0.502 | 0.498 | 0.494 |
|cholesterol| 0.689 | 0.662 |  0.679 |
|respiratory| 1.597 | 1.409 | 1.524 |
|rigidity| 2.170 | 2.241 | 2.226 |
|pulse| 15.176 | 14.161 | 14.675 |

# :chart_with_downwards_trend: rPPG models

This part is deeply based on [rppg-toolbox](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/README.md). 
We provide several methods to predict photoplethysmogram from video. 
* POS_WANG statistical algorithm
* POS_WANG that accounts only facial skin pixels
* LGI statistical algorithm
* POS_WANG that accounts only facial skin pixels
* TSCAN supervised neural model

To reproduce results run

```
bash train_scripts/ppg_models.sh
```

| model | POS_WANG | POS_WANG_mask | LGI | LGI_mask | TS_CAN |
| ---- | ---- | ----| ----| ----| ---- |
| MAE heart rate (beats / min ) | 4.463 | 2.968 | 7.368 | 5.418 | 2.841 |



# :runner: Custom training and evaluation
## Training

To train one of presented models with custom hyperparameters run the following code. `--model_class` should be one of `by_frame`, `by_video`, `ppg`.

```
python train.py --dataframe your_train_dataframe --model_class your_model_class
```
If `--model_class` is `by_frame` or `by_video` it is also necessary to specify `--model_name` and `--target`. Model name can be chosen from list `['r3d', 'mc3', 'r2plus1d', 'mvit']` in case of video and from list `['resnet50', 'vit', 'swin', 'rexnet150_pretrained', 'enet2_pretrained']` in case of frames. Target can be on of the names of medical parameters in the dataset or `'all'`. The results of running this command will be a trained model and training losses stored in directory specified in `--exp_name`, default is `test_exp`.

For any of the model classes you can also specify

* `--batch_size`
* `--lr`
* `--num_epochs`
* `--device` can be cuda or cpu, however for majority of algorithms it wil take too long to converge on cpu
* `--exp_name` path to folder where training logs and trained model will be stores
* `--seed` random seed for reproducibility

For models working with frames or videos you can also vary it'sstructure and training procedure

* `--two_layers` set to `True` (which is default) if one wants the model head to consist out of two linear layers instead of one
* `--hidden_dim` is the size of hidden dimension if `--two_layers` is set to `True` and ignored otherwise
* `--criterion` deermines whether `MAE` or `MSE` is used for training, default is `MAE`
* `--scale` scales target dividing it by some constant
* `--unfreeze` is a flag that allows to train whole model and not only head
* `--from_scratch` is a flag that initializes model with random weights instead of pretrined ones

There also are several hyperparameers used only for video case, in other cases they are ignored.
* `--crop` integer parameter, crops all input videos to specific length
* `--frequency` when set to $n$ makes model consider only every $n$-th frame of the video
* `--avg_over_frames` is a flag that makes the model divide input into blocks of size `--frequency` and average frames in each block
  


## Evaluation
To train one of presented models with custom hyperparameters run the following code. `--model_class` should be one of `by_frame`, `by_video`, `ppg`.
Model name should be on of `['resnet50', 'vit', 'swin', 'rexnet150_pretrained', 'enet2_pretrained']` for case of frame-wise models, on of `['r3d', 'mc3', 'r2plus1d', 'mvit']` for video-wise models and one of `[TSCAN', 'POS_WANG', 'LGI', 'POS_WANG_mask', 'LGI_mask']` for models retrieving PPG. 

```
python train.py --dataframe your_train_dataframe --model_class your_model_class --model_name your_model_name
```
If model is supervised i. e. not one of `['POS_WANG', 'LGI', 'POS_WANG_mask', 'LGI_mask']`, it is also necessary to provide a path to checkpoint in argument `--path_to_ckpt`. 
For evaluation of any model one can also define `--device` and `--exp_name` as in training.

Result of this programme are a file `predictions.csv` with model predictions of arget and `metrics.json`, both in the folder specified in `--exp_name`.
When evaluating a model by frames or by video it is also necessary to specify model and input structure like in training by providing parameters `--two_layers`, `--hidden_dim`, `--crop`, `--frequency`, `--avg_over_frames` with same values.


# :pray: Citations

```
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```

```
@inproceedings{savchenko2023facial,
  title = 	 {Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction},
  author =       {Savchenko, Andrey},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  pages = 	 {30119--30129},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  url={https://proceedings.mlr.press/v202/savchenko23a.html}
}
```

```
@inproceedings{savchenko2021facial,
  title={Facial expression and attributes recognition based on multi-task learning of lightweight neural networks},
  author={Savchenko, Andrey V.},
  booktitle={Proceedings of the 19th International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={119--124},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2103.17107}
}
```

```
@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url={https://arxiv.org/abs/2103.17107}
}
```

```
@inproceedings{Savchenko_2022_ECCVW,
  author    = {Savchenko, Andrey V.},
  title     = {{MT-EmotiEffNet} for Multi-task Human Affective Behavior Analysis and Learning from Synthetic Data},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV 2022) Workshops},
  pages={45--59},
  year={2023},
  organization={Springer},
  url={https://arxiv.org/abs/2207.09508}
}
```

```
@article{savchenko2022classifying,
  title={Classifying emotions and engagement in online learning based on a single facial expression recognition neural network},
  author={Savchenko, Andrey V and Savchenko, Lyudmila V and Makarov, Ilya},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/9815154}
}
```
