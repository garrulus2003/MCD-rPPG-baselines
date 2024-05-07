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

The dataset contains data about 600 patients. Each patient was filmed twice (before and after physical activity). Each time video was recorded from three different camera angles simultaniosly. Each video is assosiated with a synchronised PPG-signal, an ECG and various other medical parameters. After downloading the dataset make sure to organize as shown below. Also make sure that paths in the `db.csv` file point to true location of the dataset files.

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

STEP 2: Install Mediapipe(https://developers.google.com/mediapipe/solutions/setup_python) and [torch](https://pytorch.org/get-started/locally/) version compatible with your cuda version and packet manager. 

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
* ResNet50
* ViT
* Swin
* EficientNet_b2 initialized with weights from [face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/pretrained_faces) To use it, download sile manually and put it in `checkpoints` folder under name `pretrained_enet2`.
* ReXNet150 initialized with weights from [face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/pretrained_faces) To use it, download sile manually and put it in `checkpoints` folder under name `pretrained_enet2`.
  
To train them to predict all targets simultaneously run

```
python train.py \
--model_class by_frame \
--model_name vit \
--target all \
--dataframe data/frames_df.csv \
--exp_name exps/frames_all \
--num_epochs 100 \
--lr 1e-3 \
--unfreeze
```

To train then to predict one target but not all run with option `--target your_target`.

To train only head remove `--unfreeze` parameter. To use random initialization instead of pretrained weights add `--from_scratch`

To evaluate model on the subdataset of your choice run code similar to the one below but with your trainig parameters. Results will be in the directory named like `--test_exp` parameter.

```
python train.py \
--model_class by_frame \
--model_name vit \
--target age \
--dataframe data/frames_df.csv \
--exp_name exps/frames_all \
--path_to_ckpt fexps/frames_all/state_dict.pt
```

Results with hyperparameters as listed above are:

# :video_camera: Models with video input
Pipeline hear is similar to the one by frames.
Four models are available
*
*
*
*

The results are

# :chart_with_downwards_trend: rPPG models

# :pray: Citations

