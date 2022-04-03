from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from os import listdir
import os
import copy
import cv2

model_conv = torch.load('model_conv.pt')



model_conv.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir='fvc2000_final_test'


for img in listdir(data_dir):
    img_path = os.path.join(data_dir, img)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img.reshape(1, 3, 224, 224)
    img = torch.from_numpy(img)
    img = img.float()
    img = img.to(device)
    outputs = model_conv(img)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted + outputs.data)