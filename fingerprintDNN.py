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
import os
import copy

model_conv = torch.load('model_conv.pth')



model_conv.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir='fvc2000_final_test'

for img in os.listdir(data_dir):
    if img.endswith(".bmp"):
        img_path = os.path.join(data_dir, img)
        img = torch.from_numpy(cv2.imread(img_path)).float()
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model_conv(img)
        _, preds = torch.max(output, 1)
        print(class_names[preds])

