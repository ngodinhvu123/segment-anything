from functools import partial

import torch, torchvision
import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from box import Box
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder, TwoWayTransformer
from segment_anything.modeling import ImageEncoderViT

# from . import build_sam, SamAutomaticMaskGenerator, sam_model_registry
# from segment_anything.modeling import MaskDecoder, TwoWayTransformer
# from segment_anything.utils.transforms import ResizeLongestSide


print(torch.__version__, torchvision.__version__)
'''
Path: Image
'''
data_path = 'C:\AB\segment-anything\images\\'
folder = os.listdir(data_path)

# check first image
img_id = folder[0][:-4]
sample_path = data_path + img_id + '.jpg'
img = cv2.imread(data_path + img_id + '.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
# show image sample
# plt.show()

'''
Config traing paramaster
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


'''Setiing Tranign Start'''
MODEL_TYPE = 'Vux'
CHECKPOINT_PATH = 'C:\AB\segment-anything\model\\'



# load the model on the gpu
sam = sam.to(device)

# code for freezing parameters
if cfg.model.freeze.image_encoder:
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
if cfg.model.freeze.prompt_encoder:
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
if cfg.model.freeze.mask_decoder:
    for param in sam.mask_decoder.parameters():
        param.requires_grad = False
