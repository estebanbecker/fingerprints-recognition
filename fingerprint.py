from genericpath import isfile
import cv2 as cv
from cv2 import NORM_L2
from cv2 import NORM_HAMMING
import numpy as np
from os.path import join
from os import listdir
from sklearn.metrics import classification_report as report


print("Preparing data...")

filename_list = [f for f in listdir('DB') if isfile(join('DB',f))]

features_base={}
features_suspects={}

for filename in filename_list:
    image = cv.imread('DB/'+filename)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift=cv.SIFT_create()
    sift_des=sift.detectAndCompute(image,None)


    orb = cv.ORB_create()
    orb_des = orb.detectAndCompute(image,None)


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

i=0


for descriptor in features_suspects:
    i+=1

    print("Analysing image: "+str(i)+"/"+str(len(features_suspects)))

    match_len = {}

    for base in features_base:

        sift_match=sift_matcher.knnMatch(features_suspects[descriptor]['sift'][1],features_base[base]['sift'][1],k=2)

        good_match=[]

        for m,n in sift_match:   
            if m.distance < 0.75*n.distance:
                good_match.append(m)
        
        match_len[base]=len(good_match)

    sift_pred.append(max(match_len, key=match_len.get))
    sift_test.append(descriptor.split('_')[0])

    #Dooing the same but with ORB

    match_len = {}

    for base in features_base:

        orb_match=orb_matcher.knnMatch(features_suspects[descriptor]['orb'][1],features_base[base]['orb'][1],k=2)

        good_match=[]

        for m,n in orb_match:   
            if m.distance < 0.75*n.distance:
                good_match.append(m)
        
        match_len[base]=len(good_match)

    orb_pred.append(max(match_len, key=match_len.get))
    orb_test.append(descriptor.split('_')[0])


print(report(sift_test,sift_pred))
print(report(orb_test,orb_pred))