import cv2
import numpy as np
import os
import pandas as pd
import pathlib
import tensorflow as tf
from PIL import Image
from core.utils import get_datapoint_list



def predict(model):

    """Performing inference real time on webcam using the trained model 
       Display of full frame from webcam and predicted classes in it's
       class names in another window ROI"""

    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()
        #Convert the captured frame into RGB 
        x = frame.shape[0]
        y = frame.shape[1]
        roi = frame[100:(100+512),400:(400+x)]
        gray =cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        im = Image.fromarray(gray,'L')

        #Converting to array.
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 380x240x1 into 1x380x240x1 
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array,axis= -1)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (00, 185)
        # fontScale
        fontScale = 1
        # Red color in BGR
        color = (0, 0, 255)
        # Line thickness of 2 px
        thickness = 2
        if prediction.argmax(axis=-1)==0:
            cv2.putText(gray,'neutral',org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)
        if prediction.argmax(axis=-1)==1:
            cv2.putText(gray,'forward',org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)
        if prediction.argmax(axis=-1)==2:
            cv2.putText(gray,'backward',org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)
        #if num_classes = 4
        if prediction.argmax(axis=-1)==3:
            cv2.putText(gray,'waiting for gesture',org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)


        cv2.imshow("Capturing", frame)
        cv2.imshow("predict", gray)
        #cv2.destroyAllWindows()
        key=cv2.waitKey(1)
    
        if 0xFF == ord('q'):
            break