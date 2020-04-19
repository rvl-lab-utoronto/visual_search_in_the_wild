"""
Feature tracking with ORB features

Code from this module is based on code courtesy of Jimmy Li (MRL, mcgill, jimmyli@cim.mcgill.ca).

"""


import os
import cv2
import numpy as np
from helpers import *

orb = cv2.ORB_create()


class Frame(object):
    def __init__(self, imp, mask=None, bboxes=None):
        if type(imp) == str:
            self.imp = imp
            self.img = cv2.imread(imp,cv2.IMREAD_GRAYSCALE)
            self.objects = []
        else:
            self.img = imp
            self.objects = bboxes

        if type(mask) is not type(None):
            self.kp, self.descriptors = detect(self.img, mask)
        else:
            self.kp, self.descriptors = detect(self.img)

def draw_objects(f,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.5,
                 lineType=2,
                 color=None,
                 top_k_matches=10,
                 kp1=None,
                 kp2=None,
                 matches=None,
                 visualize=True):
    buf = np.copy(f.img)

    if f.objects:
        for obj in f.objects:
            for line in obj.lines():
                p1 = (int(line.p1[0]), int(line.p1[1]))
                p2 = (int(line.p2[0]), int(line.p2[1]))
                if color:
                    cv2.line(buf,p1,p2,color,2)
                else:
                    cv2.line(buf,p1,p2,line.color,2)
                cv2.putText(buf,
                           str(obj.label),
                           (int(obj.left+3), int(obj.top+15)),
                           font,
                           fontScale,
                           color,
                           lineType)

    if matches and kp1 and kp2:
        for m in matches[:top_k_matches]:
            cv2.circle(buf, tup2int(kp2[m.trainIdx].pt), 3, (255,0,0), 1)
            cv2.line(buf,tup2int(kp1[m.queryIdx].pt),tup2int(kp2[m.trainIdx].pt),color,1)

    if visualize:
        cv2.imshow("Objects", buf)

    return buf

def detect(img, mask=None):
    kp, des = orb.detectAndCompute(img, mask)
    return kp, des

def match(kp1, des1, kp2, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    try:
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    except cv2.error as e:
        print(e)
        matches = []

    return matches

def display_matches(img1, img2, kp1, kp2, matches, best_k=10, window_name='img', visualize=True):
    best_matches = matches[:best_k]

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,best_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if visualize:
        cv2.imshow(window_name, img3)

    best_matches = sorted(best_matches, key=lambda m: kp1[m.queryIdx].pt[1])

def compute_displacement(kp1, kp2, matches, best_k=10):
    if len(matches) > 0:
        x_disp = [ kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in matches[:best_k]]
        y_disp = [ kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in matches[:best_k]]
        return sum(x_disp) / len(x_disp), sum(y_disp) / len(y_disp)
    else:
        x_disp = 0
        y_disp = 0
        return x_disp, y_disp
