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
import PIL

model_conv = torch.load('model_conv.pt')



model_conv.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir='fvc2000_final_test'

class_names = [x for x in range(0,10)]

score=0

for img in listdir(data_dir):
    img_path = os.path.join(data_dir, img)
    img = PIL.Image.open(img_path)
    img = img.convert('RGB')
    img = transforms.Resize(256)(img)
    img = transforms.CenterCrop(224)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.float()
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model_conv(img)
    _, predicted = torch.max(outputs, 1)

    if(class_names[predicted.item()]==int(img_path[-5])):
        score+=1
    print("Predicted: ", class_names[predicted.item()], "Actual: ", img_path[-5])
print("Accuracy:",score/len(listdir(data_dir)))
    