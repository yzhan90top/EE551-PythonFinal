# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64

def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    
    h, w, _ = image.shape
    
  
    longest_edge = max(h, w)    
    
    
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    
    BLACK = [0, 0, 0]
    

    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    
    return cv2.resize(constant, (height, width))

#read train
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    
            read_path(full_path)
        else:   
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                #
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(path_name)                                
                    
    return images,labels
    


def load_dataset(path_name):
    images,labels = read_path(path_name)    
    

    images = np.array(images)
    print(images.shape)    
    
    
    labels = np.array([0 if label.endswith('yuhangzhang') else 1 for label in labels])    
    
    return images, labels

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, labels = load_dataset(sys.argv[1])
