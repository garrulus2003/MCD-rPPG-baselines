import random
import os
import torch
import torch.nn as nn
import numpy as np
import timm

from torchvision.models import resnet50, vit_b_16, swin_v2_b, ResNet50_Weights
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18, mvit_v2_s
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights, R2Plus1D_18_Weights, MViT_V2_S_Weights

CKPT_PATH = 'checkpoints'

TARGETS_TO_CLASSES = {'sex': 2}

POS_CLASS = {'sex' : 'F'}

MODEL_DIMS = {
    'resnet50': 2048,
    'vit': 768,
    'swin': 1024,
    'rexnet150_pretrained': 1920,
    'enet2_pretrained': 1408,
    'r3d': 512,
    'mc3': 512,
    'r2plus1d': 512,
    'mvit': 768
}

TARGETS = [
    'age', 'bmi', 'lower_ap', 'upper_ap', 'saturation', 'temperature',
    'stress', 'hemoglobin', 'glycated_hemoglobin', 'cholesterol', 
    'respiratory', 'rigidity', 'pulse'
]

SCALINGS = {
    "age" : 100,
    "bmi" : 40,
    "lower_ap" : 100,
    "upper_ap" : 100,
    "saturation" : 100,
    "temperature" : 40,
    "stress" : 10,
    "hemoglobin" : 20,
    "glycated_hemoglobin" : 20,
    "cholesterol" : 10,
    "respiratory" : 100,
    "rigidity" : 40,
    "pulse" : 100
}


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_criterion(is_regression, criterion_name=None):
    if is_regression:
        if criterion_name=='mse':
            return nn.MSELoss()
        elif criterion_name=='mae':
            return nn.L1Loss()
        else:
            assert False, 'We do not support this criterion yet'
    return nn.CrossEntropyLoss()


def model_type(target):
    return target not in ['sex']


def rescale(y):
    for i, target in enumerate(TARGETS):
        y[:, i] *= SCALINGS[target]
    return y


def get_weights(model_name):
    if model_name ==  'r3d':
        return R3D_18_Weights.DEFAULT
        
    if model_name ==  'mc3':
        return MC3_18_Weights.DEFAULT
        
    if model_name ==  'r2plus1d':
        return R2Plus1D_18_Weights.DEFAULT

    if model_name ==  'mvit':
        return MViT_V2_S_Weights.DEFAULT
        
    return ResNet50_Weights.DEFAULT


def get_model(model_name, unfreeze, from_scratch):

    weights = None if from_scratch else 'DEFAULT'
    if model_name == 'resnet50':
        model = resnet50(weights=weights)
        model.fc = nn.Identity()

    if model_name == 'vit':
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Identity()

    if model_name == 'swin':
        model = swin_v2_b(weights=weights)
        model.head = nn.Identity()
            
    if model_name == 'rexnet150_pretrained':
        model = timm.create_model('rexnet_150', pretrained=False)
        model.head.fc=torch.nn.Identity()
        if not from_scratch:
            model.load_state_dict(torch.load(os.path.join(CKPT_PATH, 'pretrained_rexnet.pt')))

    if model_name == 'enet2_pretrained':
        model = timm.create_model('efficientnet_b2', pretrained=False)
        model.classifier = nn.Identity()
        if not from_scratch:
            model.load_state_dict(torch.load(os.path.join(CKPT_PATH, 'pretrained_enet2.pt')))
    
    if model_name == 'r3d':
        model = r3d_18(weights=weights)
        model.fc = nn.Identity()
        
    if model_name ==  'mc3':
        model = mc3_18(weights=weights)
        model.fc = nn.Identity()
        
    if model_name ==  'r2plus1d':
        model = r2plus1d_18(weights=weights)
        model.fc = nn.Identity()
        
    if model_name ==  'mvit':
        model = mvit_v2_s(weights=weights)
        model.head[1] = nn.Identity()

    for child in list(model.children()):
        for param in child.parameters():
            param.requires_grad = unfreeze

    return model


def get_crop(model_name, crop):
    if model_name == "mvit":
        return 16
    return crop
    

def ppg_from_txt(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    ppg = [int(x.split()[0]) for x in lines]
    return np.array(ppg)
    