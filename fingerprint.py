from genericpath import isfile
import cv2 as cv
from cv2 import NORM_L2
from cv2 import NORM_HAMMING
import numpy as np
from os.path import join
from os import listdir
from matplotlib import pyplot as plt



filename_list = [f for f in listdir('DB') if isfile(join('DB',f))]

features_base={}
features_suspects={}

for filename in filename_list:
    image = cv.imread('DB/'+filename)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift=cv.SIFT_create()
    sift_des=sift.detect(image,None)

    orb = cv.ORB_create()
    orb_des = orb.detect(image,None)
    orb_des, des = orb.compute(image, orb_des)


    feat_dict = {'sift': sift_des, 'orb': orb_des}

    img_id = filename.split('.')[0]
    if img_id.split('_')[-1] == '1':
        features_base[img_id.split('_')[0]] = feat_dict
    else:
        features_suspects[img_id] = feat_dict

#BFmatcher

sift_matcher = cv.BFMatcher_create(cv.NORM_L2)
orb_matcher = cv.BFMatcher_create(cv.NORM_HAMMING)

sift_test=[]
sift_pred=[]

orb_test=[]
orb_pred=[]

for descriptor in features_suspects:

    

    for base in features_base:







