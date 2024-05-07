картинка из статьи

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
картинка из статьи

To 

## Videos preparation

## Segmentation and skin-mask extraction

## Train-Test split

# :camera: Models with frame input

# :video_camera: Models with video input

# :chart_with_downwards_trend: rPPG models

# :pray: Citations

